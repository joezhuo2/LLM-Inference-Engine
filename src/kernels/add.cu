#include <cuda_bf16.h>

#include <cstdint>

#include "cuda_check.h"
#include "engine/kernels/add.h"
#include "require.h"

namespace engine::naive {

namespace {

constexpr int kThreads = 256;

__global__ void add_kernel(__nv_bfloat16* out, const __nv_bfloat16* a, const __nv_bfloat16* b,
                           int64_t n) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16(__bfloat162float(a[i]) + __bfloat162float(b[i]));
}

bool same_shape(const Tensor& x, const Tensor& y) {
    return x.ndim == y.ndim && x.shape == y.shape;
}

}  // namespace

void add(const Tensor& out, const Tensor& a, const Tensor& b, Stream stream) {
    require(a.dtype == DType::BF16 && b.dtype == DType::BF16 && out.dtype == DType::BF16, "add",
            "out, a and b must be BF16");
    require(same_shape(a, b) && same_shape(out, a), "add", "out, a and b must have the same shape");

    const int64_t n = a.numel();
    if (n == 0) return;
    add_kernel<<<unsigned((n + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
        static_cast<__nv_bfloat16*>(out.data), static_cast<const __nv_bfloat16*>(a.data),
        static_cast<const __nv_bfloat16*>(b.data), n);
    KERNEL_CHECK(stream);
}

}  // namespace engine::naive
