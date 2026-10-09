#include <cuda_bf16.h>

#include <cstdint>

#include "cuda_check.h"
#include "engine/kernels/swiglu.h"
#include "require.h"

namespace engine::naive {

namespace {

constexpr int kThreads = 256;

__global__ void swiglu_kernel(__nv_bfloat16* out, const __nv_bfloat16* gate_up, int64_t n, int ff) {
    const int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const int64_t t = idx / ff;
    const int64_t j = idx % ff;
    const float g = __bfloat162float(gate_up[t * 2 * ff + j]);
    const float u = __bfloat162float(gate_up[t * 2 * ff + ff + j]);
    const float silu = __bfloat162float(__float2bfloat16(g / (1.0f + expf(-g))));
    out[idx] = __float2bfloat16(silu * u);
}

}  // namespace

void swiglu(const Tensor& out, const Tensor& gate_up, Stream stream) {
    require(gate_up.ndim == 2 && gate_up.dtype == DType::BF16 && gate_up.shape[1] % 2 == 0,
            "swiglu", "gate_up must be BF16 [T, 2 * ff]");
    require(out.ndim == 2 && out.dtype == DType::BF16 && out.shape[0] == gate_up.shape[0] &&
                out.shape[1] * 2 == gate_up.shape[1],
            "swiglu", "out must be BF16 [T, ff]");

    const int64_t n = out.numel();
    if (n == 0) return;
    swiglu_kernel<<<unsigned((n + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
        static_cast<__nv_bfloat16*>(out.data), static_cast<const __nv_bfloat16*>(gate_up.data), n,
        int(out.shape[1]));
    KERNEL_CHECK(stream);
}

}  // namespace engine::naive
