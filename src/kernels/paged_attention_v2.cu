#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "engine/kernels/paged_attention_v2.h"
#include "kernels/cuda_check.h"

namespace engine {

__global__ void paged_attention_decode_v2_kernel(
    nv_bfloat16* __restrict__ out, const nv_bfloat16* __restrict__ qkv,
    const nv_bfloat16* __restrict__ k_cache, const nv_bfloat16* __restrict__ v_cache,
    const int* __restrict__ block_tables, const int* __restrict__ context_lens, int batch_size,
    int heads, int kv_heads, int head_dim, int block_size, int max_blocks) {
    int attention_head_idx = blockIdx.x;
    int batch_row_idx = blockIdx.y;

    if (batch_row_idx >= batch_size || attention_head_idx >= heads) return;

    int context_len = context_lens[batch_row_idx];

    if (context_len <= 0) return;

    int kv_head_idx = attention_head_idx / (heads / kv_heads);
    int qkv_total_width = (heads + (2 * kv_heads)) * head_dim;

    const nv_bfloat16* query_ptr =
        qkv + (batch_row_idx * qkv_total_width) + (attention_head_idx * head_dim);

    extern __shared__ char dynamic_shared_memory[];
    float* shared_reduction = reinterpret_cast<float*>(dynamic_shared_memory);
    float* shared_probs = shared_reduction + blockDim.x;
    float* shared_acc = shared_probs + blockDim.x;
    nv_bfloat16* shared_query = reinterpret_cast<nv_bfloat16*>(shared_acc + head_dim);

    for (int dim_idx = threadIdx.x; dim_idx < head_dim; dim_idx += blockDim.x) {
        shared_query[dim_idx] = query_ptr[dim_idx];
        shared_acc[dim_idx] = 0.0f;
    }

    __syncthreads();

    float attention_scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    const int* seq_block_table = block_tables + (batch_row_idx * max_blocks);

    float running_max = -1e20f;
    float running_sum = 0.0f;
    int chunk_size = blockDim.x;

    for (int chunk_start = 0; chunk_start < context_len; chunk_start += chunk_size) {
        int token_idx = chunk_start + threadIdx.x;
        float local_score = -1e20f;

        if (token_idx < context_len) {
            int physical_block_idx = seq_block_table[token_idx / block_size];
            int block_offset = token_idx % block_size;

            size_t k_cache_offset =
                static_cast<size_t>(physical_block_idx) * kv_heads * block_size * head_dim +
                static_cast<size_t>(kv_head_idx) * block_size * head_dim +
                static_cast<size_t>(block_offset) * head_dim;

            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += static_cast<float>(shared_query[d]) *
                       static_cast<float>(k_cache[k_cache_offset + d]);
            }

            nv_bfloat16 dot_bf16 = __float2bfloat16(dot);
            nv_bfloat16 scaled_bf16 =
                __float2bfloat16(static_cast<float>(dot_bf16) * attention_scale);
            local_score = static_cast<float>(scaled_bf16);
        }

        shared_reduction[threadIdx.x] = local_score;

        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (threadIdx.x < stride) {
                shared_reduction[threadIdx.x] =
                    fmaxf(shared_reduction[threadIdx.x], shared_reduction[threadIdx.x + stride]);
            }

            __syncthreads();
        }

        float chunk_max = shared_reduction[0];

        __syncthreads();
        float new_running_max = fmaxf(running_max, chunk_max);

        float exp_old = __expf(running_max - new_running_max);
        running_sum *= exp_old;

        float thread_exp = (token_idx < context_len) ? __expf(local_score - new_running_max) : 0.0f;
        shared_probs[threadIdx.x] = thread_exp;
        shared_reduction[threadIdx.x] = thread_exp;

        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (threadIdx.x < stride) {
                shared_reduction[threadIdx.x] += shared_reduction[threadIdx.x + stride];
            }

            __syncthreads();
        }

        running_sum += shared_reduction[0];

        int chunk_tokens = min(chunk_size, context_len - chunk_start);

        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            float acc = shared_acc[d] * exp_old;

            for (int t = 0; t < chunk_tokens; ++t) {
                int token = chunk_start + t;
                size_t v_cache_offset =
                    static_cast<size_t>(seq_block_table[token / block_size]) * kv_heads *
                        block_size * head_dim +
                    static_cast<size_t>(kv_head_idx) * block_size * head_dim +
                    static_cast<size_t>(token % block_size) * head_dim;
                acc += shared_probs[t] * static_cast<float>(v_cache[v_cache_offset + d]);
            }

            shared_acc[d] = acc;
        }

        __syncthreads();

        running_max = new_running_max;
    }

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float final_val = (running_sum > 0.0f) ? (shared_acc[d] / running_sum) : 0.0f;

        out[batch_row_idx * (heads * head_dim) + (attention_head_idx * head_dim) + d] =
            __float2bfloat16(final_val);
    }
}

void paged_attention_decode_v2(const Tensor& out, const Tensor& qkv, const Tensor& k_cache,
                               const Tensor& v_cache, const Tensor& block_tables,
                               const Tensor& context_lens, Stream stream) {
    if (k_cache.dtype != DType::BF16 || k_cache.ndim != 4) {
        throw std::invalid_argument(
            "[paged_attention_decode_v2] k_cache must be BF16 and 4-dimensional");
    }

    if (v_cache.dtype != DType::BF16 || v_cache.ndim != 4) {
        throw std::invalid_argument(
            "[paged_attention_decode_v2] v_cache must be BF16 and 4-dimensional");
    }

    for (int i = 0; i < Tensor::kMaxDims; ++i) {
        if (k_cache.shape[i] != v_cache.shape[i]) {
            throw std::invalid_argument(
                "[paged_attention_decode_v2] k_cache and v_cache shapes must match");
        }
    }

    if (out.dtype != DType::BF16 || out.ndim != 2) {
        throw std::invalid_argument(
            "[paged_attention_decode_v2] out must be BF16 and 2-dimensional");
    }

    if (qkv.dtype != DType::BF16 || qkv.ndim != 2) {
        throw std::invalid_argument(
            "[paged_attention_decode_v2] qkv must be BF16 and 2-dimensional");
    }

    if (block_tables.dtype != DType::I32 || block_tables.ndim != 2) {
        throw std::invalid_argument(
            "[paged_attention_decode_v2] block_tables must be I32 and 2-dimensional");
    }

    if (context_lens.dtype != DType::I32 || context_lens.ndim != 1) {
        throw std::invalid_argument(
            "[paged_attention_decode_v2] context_lens must be I32 and 1-dimensional");
    }

    int64_t rows = out.shape[0];
    int64_t head_dim64 = k_cache.shape[3];
    int64_t kv_heads64 = k_cache.shape[1];
    int64_t block_size64 = k_cache.shape[2];

    if (head_dim64 <= 0 || kv_heads64 <= 0 || block_size64 <= 0 || k_cache.shape[0] <= 0) {
        throw std::invalid_argument(
            "[paged_attention_decode_v2] cache dimensions must be positive");
    }

    if (out.shape[1] <= 0 || out.shape[1] % head_dim64 != 0) {
        throw std::invalid_argument(
            "[paged_attention_decode_v2] out width must be a positive multiple of head_dim");
    }

    int64_t heads64 = out.shape[1] / head_dim64;

    if (heads64 % kv_heads64 != 0) {
        throw std::invalid_argument(
            "[paged_attention_decode_v2] heads must be a multiple of kv_heads");
    }

    if (qkv.shape[0] != rows) {
        throw std::invalid_argument("[paged_attention_decode_v2] qkv rows must match out rows");
    }

    if (qkv.shape[1] != (heads64 + 2 * kv_heads64) * head_dim64) {
        throw std::invalid_argument(
            "[paged_attention_decode_v2] qkv width must be (heads + 2 * kv_heads) * head_dim");
    }

    if (block_tables.shape[0] != rows || block_tables.shape[1] <= 0) {
        throw std::invalid_argument(
            "[paged_attention_decode_v2] block_tables must have one row per batch row and at least "
            "one column");
    }

    if (context_lens.shape[0] != rows) {
        throw std::invalid_argument(
            "[paged_attention_decode_v2] context_lens length must match the batch size");
    }

    if (rows == 0) {
        return;
    }

    int batch_size = static_cast<int>(rows);
    int head_dim = static_cast<int>(head_dim64);
    int heads = static_cast<int>(heads64);
    int kv_heads = static_cast<int>(kv_heads64);
    int block_size = static_cast<int>(block_size64);
    int max_blocks = static_cast<int>(block_tables.shape[1]);

    auto* d_out = static_cast<nv_bfloat16*>(out.data);
    auto* d_qkv = static_cast<const nv_bfloat16*>(qkv.data);
    auto* d_k_cache = static_cast<const nv_bfloat16*>(k_cache.data);
    auto* d_v_cache = static_cast<const nv_bfloat16*>(v_cache.data);
    auto* d_block_tables = static_cast<const int*>(block_tables.data);
    auto* d_context_lens = static_cast<const int*>(context_lens.data);

    int thread_block_size = 256;
    dim3 grid_dim(heads, batch_size);
    dim3 block_dim(thread_block_size);

    size_t shared_mem_bytes = (static_cast<size_t>(2 * thread_block_size) * sizeof(float)) +
                              (static_cast<size_t>(head_dim) * sizeof(float)) +
                              (static_cast<size_t>(head_dim) * sizeof(nv_bfloat16));

    paged_attention_decode_v2_kernel<<<grid_dim, block_dim, shared_mem_bytes, stream>>>(
        d_out, d_qkv, d_k_cache, d_v_cache, d_block_tables, d_context_lens, batch_size, heads,
        kv_heads, head_dim, block_size, max_blocks);

    KERNEL_CHECK(stream);
}

}  // namespace engine
