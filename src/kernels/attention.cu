#include <cuda_bf16.h>

#include <cstdint>

#include "cuda_check.h"
#include "engine/kernels/attention.h"
#include "require.h"

namespace engine::naive {

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

}  // namespace engine::naive
