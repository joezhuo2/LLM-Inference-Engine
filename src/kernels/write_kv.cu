#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include "engine/core/tensor.h"
#include "engine/kernels/write_kv.h"
#include "kernels/cuda_check.h"

namespace engine {

__global__ void write_kv_kernel(const __nv_bfloat16* __restrict__ qkv,
                                const int32_t* __restrict__ slot_mapping,
                                __nv_bfloat16* __restrict__ k_cache,
                                __nv_bfloat16* __restrict__ v_cache, int64_t num_tokens,
                                int64_t kv_heads, int64_t block_size, int64_t head_dim,
                                int64_t width, int64_t kv_width) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total_elements = num_tokens * kv_heads * head_dim;

    if (idx >= total_elements) return;

    int64_t dim_idx = idx % head_dim;
    int64_t token_head_rem = idx / head_dim;
    int64_t kv_head_idx = token_head_rem % kv_heads;
    int64_t token_idx = token_head_rem / kv_heads;
    int32_t physical_slot = slot_mapping[token_idx];

    if (physical_slot < 0) return;

    int64_t block_index = physical_slot / block_size;
    int64_t block_offset = physical_slot % block_size;

    const __nv_bfloat16* qkv_row = qkv + token_idx * width;
    int64_t col_k = width - (2 * kv_width) + (kv_head_idx * head_dim) + dim_idx;
    int64_t col_v = width - kv_width + (kv_head_idx * head_dim) + dim_idx;
    __nv_bfloat16 k_val = qkv_row[col_k];
    __nv_bfloat16 v_val = qkv_row[col_v];

    int64_t cache_idx =
        (((((block_index * kv_heads) + kv_head_idx) * block_size) + block_offset) * head_dim) +
        dim_idx;
    k_cache[cache_idx] = k_val;
    v_cache[cache_idx] = v_val;
}

void write_kv(const Tensor& k_cache, const Tensor& v_cache, const Tensor& qkv,
              const Tensor& slot_mapping, Stream stream) {
    if (k_cache.dtype != DType::BF16 || k_cache.ndim != 4) {
        throw std::invalid_argument("[write_kv] k_cache must be BF16 and 4-dimensional");
    }

    if (v_cache.dtype != DType::BF16 || v_cache.ndim != 4) {
        throw std::invalid_argument("[write_kv] v_cache must be BF16 and 4-dimensional");
    }

    if (qkv.dtype != DType::BF16 || qkv.ndim != 2) {
        throw std::invalid_argument("[write_kv] qkv must be BF16 and 2-dimensional");
    }

    if (slot_mapping.dtype != DType::I32 || slot_mapping.ndim != 1) {
        throw std::invalid_argument("[write_kv] slot_mapping must be INT32 and 1-dimensional");
    }

    for (int i = 0; i < Tensor::kMaxDims; ++i) {
        if (k_cache.shape[i] != v_cache.shape[i]) {
            throw std::invalid_argument("[write_kv] k_cache and v_cache shapes must match");
        }
    }

    int64_t num_tokens = qkv.shape[0];

    if (num_tokens == 0) {
        return;
    }

    if (slot_mapping.shape[0] != num_tokens) {
        throw std::invalid_argument("[write_kv] slot_mapping size must match num_tokens");
    }

    int64_t width = qkv.shape[1];

    int64_t kv_heads = k_cache.shape[1];
    int64_t block_size = k_cache.shape[2];
    int64_t head_dim = k_cache.shape[3];

    int64_t kv_width = kv_heads * head_dim;

    if (width <= 2 * kv_width) {
        throw std::invalid_argument("[write_kv] qkv width must be larger than 2 * kv_width");
    }

    if (num_tokens == 0) {
        return;
    }

    int64_t total_elements = num_tokens * kv_heads * head_dim;
    const int threads_per_block = 256;
    const int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    write_kv_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(qkv.data), static_cast<const int32_t*>(slot_mapping.data),
        static_cast<__nv_bfloat16*>(k_cache.data), static_cast<__nv_bfloat16*>(v_cache.data),
        num_tokens, kv_heads, block_size, head_dim, width, kv_width);
    KERNEL_CHECK(stream);
}

}  // namespace engine
