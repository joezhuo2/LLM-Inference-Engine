#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

#include "cuda_check.h"
#include "engine/kernels/swiglu.h"

namespace engine::fast {

namespace {

constexpr int kThreads = 256;
constexpr int kVecWidth = 8;

int sm_count_of_current_device() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    int sm_count = 1;
    CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
    return sm_count;
}

__device__ __forceinline__ __nv_bfloat16 swiglu_value(__nv_bfloat16 gate_bf16,
                                                      __nv_bfloat16 up_bf16) {
    float gate_float = __bfloat162float(gate_bf16);
    float up_float = __bfloat162float(up_bf16);

    float silu_float = gate_float / (1.0f + expf(-gate_float));
    float silu_rounded_float = __bfloat162float(__float2bfloat16(silu_float));

    return __float2bfloat16(silu_rounded_float * up_float);
}

}  // namespace

__global__ void swiglu_kernel(__nv_bfloat16* __restrict__ output_data,
                              const __nv_bfloat16* __restrict__ gate_up_data, int64_t total,
                              int64_t hidden_dim) {
    int64_t idx = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;

    if (idx < total) {
        int64_t row_idx = idx / hidden_dim;
        int64_t col_idx = idx % hidden_dim;
        const __nv_bfloat16* gate_ptr = gate_up_data + row_idx * (2 * hidden_dim) + col_idx;
        output_data[idx] = swiglu_value(*gate_ptr, gate_ptr[hidden_dim]);
    }
}

__global__ void swiglu_vec_kernel(__nv_bfloat16* __restrict__ output_data,
                                  const __nv_bfloat16* __restrict__ gate_up_data, int64_t total,
                                  int64_t vecs_per_row) {
    int64_t idx = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;

    if (idx < total) {
        int64_t row_idx = idx / vecs_per_row;
        int64_t vec_idx = idx % vecs_per_row;
        const uint4* gate_ptr =
            reinterpret_cast<const uint4*>(gate_up_data) + row_idx * (2 * vecs_per_row) + vec_idx;
        uint4 gate_vec = gate_ptr[0];
        uint4 up_vec = gate_ptr[vecs_per_row];

        const __nv_bfloat16* gate_values = reinterpret_cast<const __nv_bfloat16*>(&gate_vec);
        const __nv_bfloat16* up_values = reinterpret_cast<const __nv_bfloat16*>(&up_vec);

        uint4 out_vec;
        __nv_bfloat16* out_values = reinterpret_cast<__nv_bfloat16*>(&out_vec);
#pragma unroll
        for (int i = 0; i < kVecWidth; ++i) {
            out_values[i] = swiglu_value(gate_values[i], up_values[i]);
        }

        reinterpret_cast<uint4*>(output_data)[idx] = out_vec;
    }
}

void swiglu(const Tensor& out, const Tensor& gate_up, Stream stream) {
    if (gate_up.dtype != DType::BF16) {
        throw std::invalid_argument("[swiglu] gate_up tensor data type must be BF16.");
    }

    if (gate_up.ndim != 2) {
        throw std::invalid_argument("[swiglu] gate_up tensor must be exactly 2-dimensional.");
    }

    if (out.dtype != DType::BF16) {
        throw std::invalid_argument("[swiglu] out tensor data type must be BF16.");
    }

    if (out.ndim != 2) {
        throw std::invalid_argument("[swiglu] out tensor must be exactly 2-dimensional.");
    }

    int64_t batch_size_t = gate_up.shape[0];

    if (out.shape[0] != batch_size_t) {
        throw std::invalid_argument("[swiglu] out tensor row count must match gate_up row count.");
    }

    int64_t gate_up_total_width = gate_up.shape[1];

    if (gate_up_total_width % 2 != 0) {
        throw std::invalid_argument("[swiglu] gate_up tensor width must be even (2 * ff).");
    }

    int64_t hidden_dim_ff = gate_up_total_width / 2;

    if (out.shape[1] != hidden_dim_ff) {
        throw std::invalid_argument(
            "[swigl]: out tensor width must be half of gate_up tensor width (ff).");
    }

    if (batch_size_t == 0 || hidden_dim_ff == 0) {
        return;
    }

    __nv_bfloat16* output_device_ptr = reinterpret_cast<__nv_bfloat16*>(out.data);
    const __nv_bfloat16* gate_up_device_ptr = reinterpret_cast<const __nv_bfloat16*>(gate_up.data);

    bool aligned = hidden_dim_ff % kVecWidth == 0 &&
                   reinterpret_cast<uintptr_t>(output_device_ptr) % sizeof(uint4) == 0 &&
                   reinterpret_cast<uintptr_t>(gate_up_device_ptr) % sizeof(uint4) == 0;

    int64_t half_gpu_vectors = static_cast<int64_t>(sm_count_of_current_device()) * kThreads / 2;
    bool vectorizable = aligned && batch_size_t * (hidden_dim_ff / kVecWidth) >= half_gpu_vectors;

    int64_t work_per_row = vectorizable ? hidden_dim_ff / kVecWidth : hidden_dim_ff;
    int64_t total_work = batch_size_t * work_per_row;
    unsigned int grid_blocks = static_cast<unsigned int>((total_work + kThreads - 1) / kThreads);

    if (vectorizable) {
        swiglu_vec_kernel<<<grid_blocks, kThreads, 0, stream>>>(
            output_device_ptr, gate_up_device_ptr, total_work, work_per_row);
    } else {
        swiglu_kernel<<<grid_blocks, kThreads, 0, stream>>>(output_device_ptr, gate_up_device_ptr,
                                                            total_work, work_per_row);
    }

    KERNEL_CHECK(stream);
}

}  // namespace engine::fast
