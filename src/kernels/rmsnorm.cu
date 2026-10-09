#include <cuda_bf16.h>

#include <cstdint>

#include "cuda_check.h"
#include "engine/kernels/rmsnorm.h"
#include "require.h"

namespace engine::naive {

namespace {

constexpr int kThreads = 256;

__global__ void rmsnorm_kernel(__nv_bfloat16* out, const __nv_bfloat16* x,
                               const __nv_bfloat16* weight, int hidden, float eps) {
    __shared__ float partial[kThreads];
    const __nv_bfloat16* src = x + int64_t(blockIdx.x) * hidden;
    __nv_bfloat16* dst = out + int64_t(blockIdx.x) * hidden;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += kThreads) {
        const float v = __bfloat162float(src[i]);
        sum += v * v;
    }
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float rstd = rsqrtf(partial[0] / float(hidden) + eps);

    for (int i = threadIdx.x; i < hidden; i += kThreads) {
        const __nv_bfloat16 normed = __float2bfloat16(__bfloat162float(src[i]) * rstd);
        dst[i] = __float2bfloat16(__bfloat162float(weight[i]) * __bfloat162float(normed));
    }
}

}  // namespace

void rmsnorm(const Tensor& out, const Tensor& x, const Tensor& weight, float eps, Stream stream) {
    require(x.ndim == 2 && x.dtype == DType::BF16, "rmsnorm", "x must be BF16 [T, hidden]");
    require(weight.ndim == 1 && weight.dtype == DType::BF16 && weight.shape[0] == x.shape[1],
            "rmsnorm", "weight must be BF16 [hidden]");
    require(out.ndim == 2 && out.dtype == DType::BF16 && out.shape[0] == x.shape[0] &&
                out.shape[1] == x.shape[1],
            "rmsnorm", "out must be BF16 [T, hidden], like x");

    const int64_t tokens = x.shape[0];
    if (tokens == 0) return;
    rmsnorm_kernel<<<unsigned(tokens), kThreads, 0, stream>>>(
        static_cast<__nv_bfloat16*>(out.data), static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const __nv_bfloat16*>(weight.data), int(x.shape[1]), eps);
    KERNEL_CHECK(stream);
}

}  // namespace engine::naive
