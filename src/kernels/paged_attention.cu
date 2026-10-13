#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include "engine/kernels/paged_attention.h"
#include "kernels/cuda_check.h"

namespace engine {

__device__ __inline__ nv_bfloat16 float_to_bf16(float val) {
    return __float2bfloat16(val);
}

__global__ void paged_attention_decode_kernel(
    nv_bfloat16* __restrict__ out, const nv_bfloat16* __restrict__ qkv,
    const nv_bfloat16* __restrict__ k_cache, const nv_bfloat16* __restrict__ v_cache,
    const int* __restrict__ block_tables, const int* __restrict__ context_lens, int batch_size,
    int heads, int kv_heads, int head_dim, int block_size, int max_blocks, int num_blocks) {
    int attention_head_index = blockIdx.x;
    int batch_row_index = blockIdx.y;

    if (batch_row_index >= batch_size || attention_head_index >= heads) return;

    int context_length = context_lens[batch_row_index];
    if (context_length <= 0) return;

    int kv_head_index = attention_head_index / (heads / kv_heads);
    int qkv_total_width = (heads + (2 * kv_heads)) * head_dim;

    const nv_bfloat16* query_pointer =
        qkv + (batch_row_index * qkv_total_width) + (attention_head_index * head_dim);

    extern __shared__ char dynamic_shared_memory[];
    nv_bfloat16* shared_query = reinterpret_cast<nv_bfloat16*>(dynamic_shared_memory);

    for (int dim_idx = threadIdx.x; dim_idx < head_dim; dim_idx += blockDim.x) {
        shared_query[dim_idx] = query_pointer[dim_idx];
    }

    __syncthreads();

    float attention_scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    float* shared_scores = reinterpret_cast<float*>(dynamic_shared_memory);

    const int* sequence_block_table = block_tables + batch_row_index * max_blocks;

    for (int token_idx = threadIdx.x; token_idx < context_length; token_idx += blockDim.x) {
        int physical_block_idx = sequence_block_table[token_idx / block_size];
        int block_offset = token_idx % block_size;
        int k_cache_offset = physical_block_idx * (kv_heads * block_size * head_dim) +
                             (kv_head_index * block_size * head_dim) + (block_offset * head_dim);

        float dot_product_accumulator = 0.0f;
        for (int dim_idx = 0; dim_idx < head_dim; ++dim_idx) {
            dot_product_accumulator += static_cast<float>(shared_query[dim_idx]) *
                                       static_cast<float>(k_cache[k_cache_offset + dim_idx]);
        }
        nv_bfloat16 bf16_dot = __float2bfloat16(dot_product_accumulator);
        nv_bfloat16 bf16_scaled = __float2bfloat16(static_cast<float>(bf16_dot) * attention_scale);
        shared_scores[token_idx] = static_cast<float>(bf16_scaled);
    }

    __syncthreads();

    float max_attention_score = -1e20f;
    for (int token_idx = threadIdx.x; token_idx < context_length; token_idx += blockDim.x) {
        max_attention_score = fmaxf(max_attention_score, shared_scores[token_idx]);
    }

    float* shared_reduction_buffer = reinterpret_cast<float*>(shared_scores + context_length);
    shared_reduction_buffer[threadIdx.x] = max_attention_score;
    __syncthreads();

    for (int reduction_stride = blockDim.x / 2; reduction_stride > 0; reduction_stride /= 2) {
        if (threadIdx.x < reduction_stride) {
            shared_reduction_buffer[threadIdx.x] =
                fmaxf(shared_reduction_buffer[threadIdx.x],
                      shared_reduction_buffer[threadIdx.x + reduction_stride]);
        }
        __syncthreads();
    }

    float global_max_attention_score = shared_reduction_buffer[0];
    __syncthreads();

    float attention_exponential_sum = 0.0f;
    for (int token_idx = threadIdx.x; token_idx < context_length; token_idx += blockDim.x) {
        float exponential_score = __expf(shared_scores[token_idx] - global_max_attention_score);
        shared_scores[token_idx] = exponential_score;
        attention_exponential_sum += exponential_score;
    }

    shared_reduction_buffer[threadIdx.x] = attention_exponential_sum;
    __syncthreads();

    for (int reduction_stride = blockDim.x / 2; reduction_stride > 0; reduction_stride /= 2) {
        if (threadIdx.x < reduction_stride) {
            shared_reduction_buffer[threadIdx.x] +=
                shared_reduction_buffer[threadIdx.x + reduction_stride];
        }
        __syncthreads();
    }

    float global_exponential_sum = shared_reduction_buffer[0];

    for (int dim_idx = threadIdx.x; dim_idx < head_dim; dim_idx += blockDim.x) {
        float output_accumulator = 0.0f;

        for (int token_idx = 0; token_idx < context_length; ++token_idx) {
            float attention_probability = shared_scores[token_idx] / global_exponential_sum;

            int physical_block_idx = sequence_block_table[token_idx / block_size];
            int block_offset = token_idx % block_size;

            int v_cache_offset = physical_block_idx * (kv_heads * block_size * head_dim) +
                                 (kv_head_index * block_size * head_dim) +
                                 (block_offset * head_dim) + dim_idx;
            output_accumulator +=
                attention_probability * static_cast<float>(v_cache[v_cache_offset]);
        }
        out[batch_row_index * (heads * head_dim) + attention_head_index * head_dim + dim_idx] =
            __float2bfloat16(output_accumulator);
    }
}

void paged_attention_decode(const Tensor& out, const Tensor& qkv, const Tensor& k_cache,
                            const Tensor& v_cache, const Tensor& block_tables,
                            const Tensor& context_lens, Stream stream) {
    if (out.ndim != 3) {
        throw std::invalid_argument("paged_attention_decode: out tensor must have ndim == 3");
    }
    if (qkv.ndim != 2 && qkv.ndim != 3) {
        throw std::invalid_argument("paged_attention_decode: qkv tensor must have ndim == 2 or 3");
    }
    if (k_cache.ndim != 4 || v_cache.ndim != 4) {
        throw std::invalid_argument(
            "paged_attention_decode: k_cache and v_cache tensors must have ndim == 4");
    }
    if (block_tables.ndim != 2) {
        throw std::invalid_argument(
            "paged_attention_decode: block_tables tensor must have ndim == 2");
    }
    if (context_lens.ndim != 1) {
        throw std::invalid_argument(
            "paged_attention_decode: context_lens tensor must have ndim == 1");
    }

    int batch_size = static_cast<int>(qkv.shape[0]);

    if (batch_size == 0) {
        return;
    }

    int heads = static_cast<int>(out.shape[1]);
    int head_dim = static_cast<int>(out.shape[2]);
    int kv_heads = static_cast<int>(k_cache.shape[1]);
    int block_size = static_cast<int>(k_cache.shape[2]);
    int max_blocks = static_cast<int>(block_tables.shape[1]);
    int num_blocks = static_cast<int>(k_cache.shape[0]);

    auto* d_out = static_cast<nv_bfloat16*>(out.data);
    auto* d_qkv = static_cast<const nv_bfloat16*>(qkv.data);
    auto* d_k_cache = static_cast<const nv_bfloat16*>(k_cache.data);
    auto* d_v_cache = static_cast<const nv_bfloat16*>(v_cache.data);
    auto* d_block_tables = static_cast<const int*>(block_tables.data);
    auto* d_context_lens = static_cast<const int*>(context_lens.data);

    int max_context_length = 0;
    const int* h_context_lens = static_cast<const int*>(context_lens.data);
    int num_batch_rows = static_cast<int>(context_lens.shape[0]);
    for (int i = 0; i < num_batch_rows; ++i) {
        max_context_length = std::max(max_context_length, h_context_lens[i]);
    }

    int thread_block_size = 256;

    dim3 grid_dim(heads, batch_size);
    dim3 block_dim(thread_block_size);

    size_t shared_mem_bytes =
        (max_context_length * sizeof(float)) + (head_dim * sizeof(nv_bfloat16));

    paged_attention_decode_kernel<<<grid_dim, block_dim, shared_mem_bytes, stream>>>(
        d_out, d_qkv, d_k_cache, d_v_cache, d_block_tables, d_context_lens, batch_size, heads,
        kv_heads, head_dim, block_size, max_blocks, num_blocks);

    KERNEL_CHECK(stream);
}

}  // namespace engine
