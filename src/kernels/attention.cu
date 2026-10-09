#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>

#include "cuda_check.h"
#include "engine/kernels/attention.h"
#include "require.h"

namespace engine::naive {

namespace {

constexpr int kThreads = 256;
constexpr size_t kMaxSharedBytes = 48 * 1024;

__global__ void attention_kernel(__nv_bfloat16* out, const __nv_bfloat16* qkv,
                                 const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache,
                                 int64_t start, int width, int heads, int kv_heads, int head_dim,
                                 float scale) {
    extern __shared__ float smem[];
    float& max_score = smem[0];
    float& sum = smem[1];
    float* q = smem + 2;
    float* p = q + head_dim;

    const int64_t t = blockIdx.x;
    const int h = blockIdx.y;
    const int kv_width = kv_heads * head_dim;
    const int kv_offset = (h / (heads / kv_heads)) * head_dim;
    const int64_t context = start + t + 1;

    for (int d = threadIdx.x; d < head_dim; d += kThreads)
        q[d] = __bfloat162float(qkv[t * width + h * head_dim + d]);
    __syncthreads();

    for (int64_t j = threadIdx.x; j < context; j += kThreads) {
        const __nv_bfloat16* k = k_cache + j * kv_width + kv_offset;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) dot = fmaf(q[d], __bfloat162float(k[d]), dot);
        p[j] = __bfloat162float(__float2bfloat16(__bfloat162float(__float2bfloat16(dot)) * scale));
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float m = p[0];
        for (int64_t j = 1; j < context; ++j) m = fmaxf(m, p[j]);
        max_score = m;
    }
    __syncthreads();

    for (int64_t j = threadIdx.x; j < context; j += kThreads) p[j] = expf(p[j] - max_score);
    __syncthreads();

    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int64_t j = 0; j < context; ++j) s += p[j];
        sum = s;
    }
    __syncthreads();

    for (int64_t j = threadIdx.x; j < context; j += kThreads)
        p[j] = __bfloat162float(__float2bfloat16(p[j] / sum));
    __syncthreads();

    for (int d = threadIdx.x; d < head_dim; d += kThreads) {
        float acc = 0.0f;
        for (int64_t j = 0; j < context; ++j)
            acc = fmaf(p[j], __bfloat162float(v_cache[j * kv_width + kv_offset + d]), acc);
        out[t * heads * head_dim + h * head_dim + d] = __float2bfloat16(acc);
    }
}

}  // namespace

void store_kv(const Tensor& k_cache, const Tensor& v_cache, const Tensor& qkv, int64_t start,
              Stream stream) {
    require(qkv.ndim == 2 && qkv.dtype == DType::BF16, "store_kv",
            "qkv must be BF16 [T, (heads + 2 * kv_heads) * head_dim]");
    require(k_cache.ndim == 2 && k_cache.dtype == DType::BF16 && v_cache.ndim == 2 &&
                v_cache.dtype == DType::BF16 && k_cache.shape == v_cache.shape,
            "store_kv", "k_cache and v_cache must both be BF16 [S, kv_heads * head_dim]");
    const int64_t tokens = qkv.shape[0];
    const int64_t width = qkv.shape[1];
    const int64_t kv_width = k_cache.shape[1];
    require(kv_width > 0 && width > 2 * kv_width, "store_kv",
            "qkv width must exceed twice the cache width");
    require(start >= 0 && start + tokens <= k_cache.shape[0], "store_kv",
            "rows start to start + T must fit in the cache");
    if (tokens == 0) return;

    const size_t elem = sizeof(__nv_bfloat16);
    const auto* src = static_cast<const __nv_bfloat16*>(qkv.data) + (width - 2 * kv_width);
    auto* k_dst = static_cast<__nv_bfloat16*>(k_cache.data) + start * kv_width;
    auto* v_dst = static_cast<__nv_bfloat16*>(v_cache.data) + start * kv_width;
    CUDA_CHECK(cudaMemcpy2DAsync(k_dst, kv_width * elem, src, width * elem, kv_width * elem,
                                 size_t(tokens), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpy2DAsync(v_dst, kv_width * elem, src + kv_width, width * elem,
                                 kv_width * elem, size_t(tokens), cudaMemcpyDeviceToDevice,
                                 stream));
}

void attention(const Tensor& out, const Tensor& qkv, const Tensor& k_cache, const Tensor& v_cache,
               int64_t start, int heads, int kv_heads, Stream stream) {
    require(heads > 0 && kv_heads > 0 && heads % kv_heads == 0, "attention",
            "heads must be a positive multiple of kv_heads");
    require(qkv.ndim == 2 && qkv.dtype == DType::BF16, "attention",
            "qkv must be BF16 [T, (heads + 2 * kv_heads) * head_dim]");
    const int64_t tokens = qkv.shape[0];
    const int64_t width = qkv.shape[1];
    const int64_t slots = heads + 2 * int64_t(kv_heads);
    require(width > 0 && width % slots == 0, "attention",
            "qkv width must be (heads + 2 * kv_heads) * head_dim");
    const int head_dim = int(width / slots);
    require(out.ndim == 2 && out.dtype == DType::BF16 && out.shape[0] == tokens &&
                out.shape[1] == int64_t(heads) * head_dim,
            "attention", "out must be BF16 [T, heads * head_dim]");
    require(k_cache.ndim == 2 && k_cache.dtype == DType::BF16 && v_cache.ndim == 2 &&
                v_cache.dtype == DType::BF16 && k_cache.shape == v_cache.shape &&
                k_cache.shape[1] == int64_t(kv_heads) * head_dim,
            "attention", "k_cache and v_cache must both be BF16 [S, kv_heads * head_dim]");
    require(start >= 0 && start + tokens <= k_cache.shape[0], "attention",
            "positions start to start + T - 1 must be in the cache");
    const size_t shared = size_t(2 + head_dim + start + tokens) * sizeof(float);
    require(shared <= kMaxSharedBytes, "attention",
            "head_dim plus the context length must fit in 48 KB of shared memory");
    if (tokens == 0) return;

    const dim3 grid{unsigned(tokens), unsigned(heads)};
    attention_kernel<<<grid, kThreads, shared, stream>>>(
        static_cast<__nv_bfloat16*>(out.data), static_cast<const __nv_bfloat16*>(qkv.data),
        static_cast<const __nv_bfloat16*>(k_cache.data),
        static_cast<const __nv_bfloat16*>(v_cache.data), start, int(width), heads, kv_heads,
        head_dim, float(1.0 / std::sqrt(double(head_dim))));
    KERNEL_CHECK(stream);
}

}  // namespace engine::naive
