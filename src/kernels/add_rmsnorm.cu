#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

#include "cuda_check.h"
#include "engine/kernels/add_rmsnorm.h"

namespace engine::fast {

constexpr int kWarpSize = 32;

__device__ __forceinline__ float warpReduceSum(float val) {
    for (int mask = kWarpSize / 2; mask > 0; mask /= 2) {
        val += __shfl_down_sync(0xffffffff, val, mask);
    }

    return val;
}

__device__ __forceinline__ float blockReduceSum(float val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x % kWarpSize;
    int warp_id = threadIdx.x / kWarpSize;

    val = warpReduceSum(val);

    if (lane == 0) {
        shared[warp_id] = val;
    }

    __syncthreads();

    int num_warps = blockDim.x / kWarpSize;
    val = (lane < num_warps) ? shared[lane] : 0.0f;

    if (warp_id == 0) {
        val = warpReduceSum(val);
    }
    return val;
}

__global__ void add_rmsnorm_kernel(__nv_bfloat16* sum, __nv_bfloat16* __restrict__ norm,
                                   const __nv_bfloat16* x, const __nv_bfloat16* residual,
                                   const __nv_bfloat16* __restrict__ weight, float eps,
                                   int64_t total_batch_size, int64_t hidden) {
    int row = blockIdx.x;

    if (row >= total_batch_size) return;

    const __nv_bfloat16* x_row = x + (row * hidden);
    const __nv_bfloat16* res_row = residual + (row * hidden);
    __nv_bfloat16* sum_row = sum + (row * hidden);
    __nv_bfloat16* norm_row = norm + (row * hidden);

    float local_sq_sum = 0.0f;

    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float xv = __bfloat162float(x_row[i]);
        float rv = __bfloat162float(res_row[i]);

        const __nv_bfloat16 s_bf = __float2bfloat16(xv + rv);
        sum_row[i] = s_bf;
        const float s = __bfloat162float(s_bf);
        local_sq_sum += s * s;
    }

    float block_sq_sum = blockReduceSum(local_sq_sum);

    __shared__ float s_rstd;

    if (threadIdx.x == 0) {
        s_rstd = rsqrtf(block_sq_sum / static_cast<float>(hidden) + eps);
    }

    __syncthreads();

    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float s_val = __bfloat162float(sum_row[i]);
        float w_val = __bfloat162float(weight[i]);

        const __nv_bfloat16 normed = __float2bfloat16(s_val * s_rstd);
        norm_row[i] = __float2bfloat16(w_val * __bfloat162float(normed));
    }
}

constexpr int kVecWidth = 8;
constexpr int kMaxVecsPerThread = 4;
constexpr int kVecThreads = 256;

__device__ __forceinline__ float blockAllReduceSum(float val) {
    __shared__ float shared[kVecThreads / kWarpSize];
    const int lane = threadIdx.x % kWarpSize;
    const int warp_id = threadIdx.x / kWarpSize;

    for (int mask = kWarpSize / 2; mask > 0; mask /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }

    if (lane == 0) shared[warp_id] = val;
    __syncthreads();

    val = (lane < blockDim.x / kWarpSize) ? shared[lane] : 0.0f;
    for (int mask = kWarpSize / 2; mask > 0; mask /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

__global__ void add_rmsnorm_vec_kernel(__nv_bfloat16* sum, __nv_bfloat16* __restrict__ norm,
                                       const __nv_bfloat16* x, const __nv_bfloat16* residual,
                                       const __nv_bfloat16* __restrict__ weight, float eps,
                                       int64_t hidden) {
    const int64_t row_offset = int64_t(blockIdx.x) * hidden;
    const uint4* x_row = reinterpret_cast<const uint4*>(x + row_offset);
    const uint4* res_row = reinterpret_cast<const uint4*>(residual + row_offset);
    uint4* sum_row = reinterpret_cast<uint4*>(sum + row_offset);
    uint4* norm_row = reinterpret_cast<uint4*>(norm + row_offset);
    const uint4* weight_vec = reinterpret_cast<const uint4*>(weight);
    const int vecs = int(hidden / kVecWidth);

    uint4 held[kMaxVecsPerThread];
    float local_sq_sum = 0.0f;

#pragma unroll
    for (int k = 0; k < kMaxVecsPerThread; ++k) {
        const int v = threadIdx.x + k * kVecThreads;
        if (v >= vecs) break;
        uint4 xv = x_row[v];
        const uint4 rv = res_row[v];
        __nv_bfloat16* xe = reinterpret_cast<__nv_bfloat16*>(&xv);
        const __nv_bfloat16* re = reinterpret_cast<const __nv_bfloat16*>(&rv);
#pragma unroll
        for (int e = 0; e < kVecWidth; ++e) {
            xe[e] = __float2bfloat16(__bfloat162float(xe[e]) + __bfloat162float(re[e]));
            const float s = __bfloat162float(xe[e]);
            local_sq_sum += s * s;
        }
        held[k] = xv;
        sum_row[v] = xv;
    }

    const float block_sq_sum = blockAllReduceSum(local_sq_sum);
    const float rstd = rsqrtf(block_sq_sum / static_cast<float>(hidden) + eps);

#pragma unroll
    for (int k = 0; k < kMaxVecsPerThread; ++k) {
        const int v = threadIdx.x + k * kVecThreads;
        if (v >= vecs) break;
        const uint4 wv = weight_vec[v];
        uint4 out = held[k];
        const __nv_bfloat16* we = reinterpret_cast<const __nv_bfloat16*>(&wv);
        __nv_bfloat16* oe = reinterpret_cast<__nv_bfloat16*>(&out);
#pragma unroll
        for (int e = 0; e < kVecWidth; ++e) {
            const __nv_bfloat16 normed = __float2bfloat16(__bfloat162float(oe[e]) * rstd);
            oe[e] = __float2bfloat16(__bfloat162float(we[e]) * __bfloat162float(normed));
        }
        norm_row[v] = out;
    }
}

bool aligned16(const void* p) {
    return reinterpret_cast<uintptr_t>(p) % 16 == 0;
}

void add_rmsnorm(const Tensor& sum, const Tensor& norm, const Tensor& x, const Tensor& residual,
                 const Tensor& weight, float eps, Stream stream) {
    if (sum.dtype != DType::BF16 || norm.dtype != DType::BF16 || x.dtype != DType::BF16 ||
        residual.dtype != DType::BF16 || weight.dtype != DType::BF16) {
        throw std::invalid_argument("[add_rmsnorm] all tensors must be BF16");
    }

    if (x.ndim != 2) {
        throw std::invalid_argument("[add_rmsnorm] x must be 2-dimensional");
    }

    if (residual.shape != x.shape || sum.shape != x.shape || norm.shape != x.shape) {
        throw std::invalid_argument(
            "[add_rmsnorm] sum, norm, and residual must have the same shape as x");
    }

    if (weight.ndim != 1) {
        throw std::invalid_argument("[add_rmsnorm] weight must be 1-dimensional");
    }

    const int64_t seq_len = x.shape[0];
    const int64_t hidden = x.shape[1];

    if (weight.shape[0] != hidden) {
        throw std::invalid_argument("[add_rmsnorm] weight length must equal hidden dimension");
    }

    if (seq_len == 0) return;

    const dim3 grid(static_cast<unsigned int>(seq_len));

    if (hidden % kVecWidth == 0 && hidden / kVecWidth <= kVecThreads * kMaxVecsPerThread &&
        aligned16(sum.data) && aligned16(norm.data) && aligned16(x.data) &&
        aligned16(residual.data) && aligned16(weight.data)) {
        add_rmsnorm_vec_kernel<<<grid, kVecThreads, 0, stream>>>(
            static_cast<__nv_bfloat16*>(sum.data), static_cast<__nv_bfloat16*>(norm.data),
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(residual.data),
            static_cast<const __nv_bfloat16*>(weight.data), eps, hidden);
        KERNEL_CHECK(stream);
        return;
    }

    const int threads_per_block = 256;

    add_rmsnorm_kernel<<<grid, threads_per_block, 0, stream>>>(
        static_cast<__nv_bfloat16*>(sum.data), static_cast<__nv_bfloat16*>(norm.data),
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(residual.data),
        static_cast<const __nv_bfloat16*>(weight.data), eps, seq_len, hidden);

    KERNEL_CHECK(stream);
}
}  // namespace engine::fast
