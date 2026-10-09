#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>

#include "cuda_check.h"
#include "engine/kernels/rope.h"
#include "require.h"

namespace engine::naive {

namespace {

constexpr int kThreads = 256;
constexpr int kMaxHeadDim = 256;

struct InvFreq {
    float v[kMaxHeadDim / 2];
};

__global__ void rope_kernel(__nv_bfloat16* qkv, const int32_t* positions, int width,
                            int rotated_heads, int head_dim, InvFreq inv_freq) {
    const int half = head_dim / 2;
    const float pos = float(positions[blockIdx.x]);
    __nv_bfloat16* row = qkv + int64_t(blockIdx.x) * width;
    for (int p = threadIdx.x; p < rotated_heads * half; p += kThreads) {
        const int i = p % half;
        __nv_bfloat16* x = row + (p / half) * head_dim;
        const float angle = pos * inv_freq.v[i];
        const float c = __bfloat162float(__float2bfloat16(cosf(angle)));
        const float s = __bfloat162float(__float2bfloat16(sinf(angle)));
        const float lo = __bfloat162float(x[i]);
        const float hi = __bfloat162float(x[i + half]);
        const float lo_c = __bfloat162float(__float2bfloat16(lo * c));
        const float hi_c = __bfloat162float(__float2bfloat16(hi * c));
        const float lo_s = __bfloat162float(__float2bfloat16(lo * s));
        const float hi_s = __bfloat162float(__float2bfloat16(-hi * s));
        x[i] = __float2bfloat16(lo_c + hi_s);
        x[i + half] = __float2bfloat16(hi_c + lo_s);
    }
}

}  // namespace

void rope(const Tensor& qkv, const Tensor& positions, int heads, int kv_heads, float theta,
          Stream stream) {
    require(heads > 0 && kv_heads > 0, "rope", "heads and kv_heads must be positive");
    require(qkv.ndim == 2 && qkv.dtype == DType::BF16, "rope",
            "qkv must be BF16 [T, (heads + 2 * kv_heads) * head_dim]");
    require(
        positions.ndim == 1 && positions.dtype == DType::I32 && positions.shape[0] == qkv.shape[0],
        "rope", "positions must be I32 [T]");
    const int64_t width = qkv.shape[1];
    const int64_t slots = heads + 2 * int64_t(kv_heads);
    require(width % slots == 0 && (width / slots) % 2 == 0, "rope",
            "qkv width must be (heads + 2 * kv_heads) * head_dim with an even head_dim");
    const int head_dim = int(width / slots);
    require(head_dim <= kMaxHeadDim, "rope", "head_dim must be at most 256");

    InvFreq inv_freq{};
    for (int i = 0; i < head_dim / 2; ++i)
        inv_freq.v[i] = 1.0f / std::pow(theta, float(2 * i) / float(head_dim));

    const int64_t tokens = qkv.shape[0];
    if (tokens == 0) return;
    rope_kernel<<<unsigned(tokens), kThreads, 0, stream>>>(
        static_cast<__nv_bfloat16*>(qkv.data), static_cast<const int32_t*>(positions.data),
        int(width), heads + kv_heads, head_dim, inv_freq);
    KERNEL_CHECK(stream);
}

}  // namespace engine::naive
