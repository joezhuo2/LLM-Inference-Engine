#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "engine/kernels/flash_prefill.h"

namespace engine {

__global__ void flash_attention_prefill_kernel(
    nv_bfloat16* __restrict__ out, const nv_bfloat16* __restrict__ qkv,
    const nv_bfloat16* __restrict__ k_cache, const nv_bfloat16* __restrict__ v_cache,
    const int32_t* __restrict__ block_tables, const int32_t* __restrict__ chunk_starts,
    const int32_t* __restrict__ cached_lens, int num_chunks, int total_step_rows, int query_heads,
    int kv_heads, int head_dim, int block_size, int blocks_per_chunk) {
}

void flash_attention_prefill(const Tensor& out, const Tensor& qkv, const Tensor& k_cache,
                             const Tensor& v_cache, const Tensor& block_tables,
                             const Tensor& chunk_starts, const Tensor& cached_lens, Stream stream) {

}

}  // namespace engine
