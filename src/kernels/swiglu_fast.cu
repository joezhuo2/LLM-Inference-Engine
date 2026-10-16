#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include "cuda_check.h"
#include "engine/kernels/swiglu.h"

namespace engine::fast {

__global__ void swiglu_kernel(__nv_bfloat16* __restrict__ output_data,
                              const __nv_bfloat16* __restrict__ gate_up_data, int64_t num_rows,
                              int64_t hidden_dim) {
    int64_t col_idx = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    int64_t row_idx = (static_cast<int64_t>(blockIdx.y) * blockDim.y) + threadIdx.y;

    if (row_idx < num_rows && col_idx < hidden_dim) {
        int64_t gate_up_row_offset = row_idx * (2 * hidden_dim);

        const __nv_bfloat16* gate_ptr = gate_up_data + gate_up_row_offset + col_idx;
        const __nv_bfloat16* up_ptr = gate_up_data + gate_up_row_offset + col_idx + hidden_dim;

        int64_t output_row_offset = row_idx * hidden_dim;
        __nv_bfloat16* output_ptr = output_data + output_row_offset + col_idx;

        __nv_bfloat16 gate_bf16 = *gate_ptr;
        __nv_bfloat16 up_bf16 = *up_ptr;

        float gate_float = __bfloat162float(gate_bf16);
        float up_float = __bfloat162float(up_bf16);

        float neg_gate_float = -gate_float;
        float exp_val = expf(neg_gate_float);
        float denom_val = 1.0f + exp_val;

        float silu_float = gate_float / denom_val;
        __nv_bfloat16 silu_bf16 = __float2bfloat16(silu_float);
        float silu_rounded_float = __bfloat162float(silu_bf16);

        float final_product_float = silu_rounded_float * up_float;
        __nv_bfloat16 final_output_bf16 = __float2bfloat16(final_product_float);

        *output_ptr = final_output_bf16;
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

    dim3 block_threads(256, 1);

    dim3 grid_blocks(
        static_cast<unsigned int>((hidden_dim_ff + block_threads.x - 1) / block_threads.x),
        static_cast<unsigned int>((batch_size_t + block_threads.y - 1) / block_threads.y));

    swiglu_kernel<<<grid_blocks, block_threads, 0, stream>>>(output_device_ptr, gate_up_device_ptr,
                                                             batch_size_t, hidden_dim_ff);

    KERNEL_CHECK(stream);
}

}  // namespace engine::fast
