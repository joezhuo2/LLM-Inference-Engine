#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "engine/kernels/paged_attention_split.h"
#include "kernels/cuda_check.h"

namespace engine {

__global__ void paged_attention_split_partial_kernel(
    float* __restrict__ scratch, const nv_bfloat16* __restrict__ qkv,
    const nv_bfloat16* __restrict__ k_cache, const nv_bfloat16* __restrict__ v_cache,
    const int* __restrict__ block_tables, const int* __restrict__ context_lens, int batch_size,
    int num_splits, int heads, int kv_heads, int head_dim, int block_size, int max_blocks) {
    int split_idx = blockIdx.x;
    int attention_head_idx = blockIdx.y;
    int batch_row_idx = blockIdx.z;

    if (batch_row_idx >= batch_size || attention_head_idx >= heads) return;

    int context_len = context_lens[batch_row_idx];

    if (context_len <= 0) return;

    int total_blocks = (context_len + block_size - 1) / block_size;
    int blocks_per_split = (total_blocks + num_splits - 1) / num_splits;
    int range_start = min(context_len, split_idx * blocks_per_split * block_size);
    int range_end = min(context_len, (split_idx + 1) * blocks_per_split * block_size);

    if (range_start >= range_end) return;

    int kv_head_idx = attention_head_idx / (heads / kv_heads);
    int qkv_total_width = (heads + (2 * kv_heads)) * head_dim;

    const nv_bfloat16* query_ptr =
        qkv + (batch_row_idx * qkv_total_width) + (attention_head_idx * head_dim);

    extern __shared__ char dynamic_shared_memory[];
    float* shared_reduction = reinterpret_cast<float*>(dynamic_shared_memory);
    float* shared_probs = shared_reduction + blockDim.x;
    float* shared_partial = shared_probs + blockDim.x;
    float* shared_acc = shared_partial + blockDim.x;
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

    for (int chunk_start = range_start; chunk_start < range_end; chunk_start += chunk_size) {
        int token_idx = chunk_start + threadIdx.x;
        float local_score = -1e20f;

        int lane_in_group = threadIdx.x % 8;
        int group_idx = threadIdx.x / 8;
        int group_count = blockDim.x / 8;
        int chunk_tokens = min(chunk_size, range_end - chunk_start);

        for (int slot = group_idx; slot < chunk_size; slot += group_count) {
            int slot_token = chunk_start + slot;
            float dot = 0.0f;

            if (slot < chunk_tokens) {
                size_t k_cache_offset =
                    static_cast<size_t>(seq_block_table[slot_token / block_size]) * kv_heads *
                        block_size * head_dim +
                    static_cast<size_t>(kv_head_idx) * block_size * head_dim +
                    static_cast<size_t>(slot_token % block_size) * head_dim;

                for (int d = lane_in_group; d < head_dim; d += 8) {
                    dot += static_cast<float>(shared_query[d]) *
                           static_cast<float>(k_cache[k_cache_offset + d]);
                }
            }

            dot += __shfl_xor_sync(0xffffffffu, dot, 4);
            dot += __shfl_xor_sync(0xffffffffu, dot, 2);
            dot += __shfl_xor_sync(0xffffffffu, dot, 1);

            if (lane_in_group == 0) {
                nv_bfloat16 dot_bf16 = __float2bfloat16(dot);
                nv_bfloat16 scaled_bf16 =
                    __float2bfloat16(static_cast<float>(dot_bf16) * attention_scale);
                shared_probs[slot] = static_cast<float>(scaled_bf16);
            }
        }

        __syncthreads();

        if (token_idx < range_end) {
            local_score = shared_probs[threadIdx.x];
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

        float thread_exp = (token_idx < range_end) ? __expf(local_score - new_running_max) : 0.0f;
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

        int slice_count = max(1, static_cast<int>(blockDim.x) / head_dim);
        int active_threads = slice_count * head_dim;

        if (static_cast<int>(threadIdx.x) < active_threads) {
            int d = threadIdx.x % head_dim;
            int slice = threadIdx.x / head_dim;
            float partial = 0.0f;

            for (int t = slice; t < chunk_tokens; t += slice_count) {
                int token = chunk_start + t;
                size_t v_cache_offset = static_cast<size_t>(seq_block_table[token / block_size]) *
                                            kv_heads * block_size * head_dim +
                                        static_cast<size_t>(kv_head_idx) * block_size * head_dim +
                                        static_cast<size_t>(token % block_size) * head_dim;
                partial += shared_probs[t] * static_cast<float>(v_cache[v_cache_offset + d]);
            }

            shared_partial[threadIdx.x] = partial;
        }

        __syncthreads();

        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            float acc = shared_acc[d] * exp_old;

            for (int slice = 0; slice < slice_count; ++slice) {
                acc += shared_partial[slice * head_dim + d];
            }

            shared_acc[d] = acc;
        }

        __syncthreads();

        running_max = new_running_max;
    }

    float* partial_out =
        scratch + ((static_cast<size_t>(batch_row_idx) * heads + attention_head_idx) * num_splits +
                   split_idx) *
                      (head_dim + 2);

    if (threadIdx.x == 0) {
        partial_out[0] = running_max;
        partial_out[1] = running_sum;
    }

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        partial_out[2 + d] = shared_acc[d];
    }
}

__global__ void paged_attention_split_merge_kernel(nv_bfloat16* __restrict__ out,
                                                   const float* __restrict__ scratch,
                                                   const int* __restrict__ context_lens,
                                                   int num_splits, int heads, int head_dim,
                                                   int block_size) {
    int attention_head_idx = blockIdx.x;
    int batch_row_idx = blockIdx.y;
    int context_len = context_lens[batch_row_idx];

    if (context_len <= 0) return;

    int total_blocks = (context_len + block_size - 1) / block_size;
    int blocks_per_split = (total_blocks + num_splits - 1) / num_splits;
    int used_splits = (total_blocks + blocks_per_split - 1) / blocks_per_split;

    const float* partials =
        scratch + (static_cast<size_t>(batch_row_idx) * heads + attention_head_idx) * num_splits *
                      (head_dim + 2);
    int stride = head_dim + 2;

    float global_max = partials[0];

    for (int split = 1; split < used_splits; ++split) {
        global_max = fmaxf(global_max, partials[split * stride]);
    }

    float total_sum = 0.0f;

    for (int split = 0; split < used_splits; ++split) {
        total_sum += __expf(partials[split * stride] - global_max) * partials[split * stride + 1];
    }

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;

        for (int split = 0; split < used_splits; ++split) {
            acc += __expf(partials[split * stride] - global_max) * partials[split * stride + 2 + d];
        }

        float final_val = (total_sum > 0.0f) ? (acc / total_sum) : 0.0f;

        out[batch_row_idx * (heads * head_dim) + (attention_head_idx * head_dim) + d] =
            __float2bfloat16(final_val);
    }
}

void paged_attention_decode_split(const Tensor& out, const Tensor& qkv, const Tensor& k_cache,
                                  const Tensor& v_cache, const Tensor& block_tables,
                                  const Tensor& context_lens, const Tensor& scratch,
                                  Stream stream) {
    const auto fail = [](const char* message) {
        throw std::invalid_argument(std::string("[paged_attention_decode_split] ") + message);
    };

    if (k_cache.dtype != DType::BF16 || k_cache.ndim != 4) {
        fail("k_cache must be BF16 and 4-dimensional");
    }

    if (v_cache.dtype != DType::BF16 || v_cache.ndim != 4) {
        fail("v_cache must be BF16 and 4-dimensional");
    }

    for (int i = 0; i < Tensor::kMaxDims; ++i) {
        if (k_cache.shape[i] != v_cache.shape[i]) fail("k_cache and v_cache shapes must match");
    }

    if (out.dtype != DType::BF16 || out.ndim != 2) fail("out must be BF16 and 2-dimensional");
    if (qkv.dtype != DType::BF16 || qkv.ndim != 2) fail("qkv must be BF16 and 2-dimensional");

    if (block_tables.dtype != DType::I32 || block_tables.ndim != 2) {
        fail("block_tables must be I32 and 2-dimensional");
    }

    if (context_lens.dtype != DType::I32 || context_lens.ndim != 1) {
        fail("context_lens must be I32 and 1-dimensional");
    }

    if (scratch.dtype != DType::F32 || scratch.ndim != 4) {
        fail("scratch must be F32 and 4-dimensional");
    }

    int64_t rows = out.shape[0];
    int64_t head_dim64 = k_cache.shape[3];
    int64_t kv_heads64 = k_cache.shape[1];
    int64_t block_size64 = k_cache.shape[2];

    if (head_dim64 <= 0 || kv_heads64 <= 0 || block_size64 <= 0 || k_cache.shape[0] <= 0) {
        fail("cache dimensions must be positive");
    }

    if (out.shape[1] <= 0 || out.shape[1] % head_dim64 != 0) {
        fail("out width must be a positive multiple of head_dim");
    }

    int64_t heads64 = out.shape[1] / head_dim64;

    if (heads64 % kv_heads64 != 0) fail("heads must be a multiple of kv_heads");
    if (qkv.shape[0] != rows) fail("qkv rows must match out rows");

    if (qkv.shape[1] != (heads64 + 2 * kv_heads64) * head_dim64) {
        fail("qkv width must be (heads + 2 * kv_heads) * head_dim");
    }

    if (block_tables.shape[0] != rows || block_tables.shape[1] <= 0) {
        fail("block_tables must have one row per batch row and at least one column");
    }

    if (context_lens.shape[0] != rows) fail("context_lens length must match the batch size");

    if (scratch.shape[0] != rows || scratch.shape[1] != heads64 ||
        scratch.shape[3] != head_dim64 + 2) {
        fail("scratch must be [batch, heads, num_splits, head_dim + 2]");
    }

    if (scratch.shape[2] <= 0) fail("scratch must hold at least one split");

    if (rows == 0) {
        return;
    }

    int batch_size = static_cast<int>(rows);
    int head_dim = static_cast<int>(head_dim64);
    int heads = static_cast<int>(heads64);
    int kv_heads = static_cast<int>(kv_heads64);
    int block_size = static_cast<int>(block_size64);
    int max_blocks = static_cast<int>(block_tables.shape[1]);
    int num_splits = static_cast<int>(scratch.shape[2]);

    auto* d_out = static_cast<nv_bfloat16*>(out.data);
    auto* d_scratch = static_cast<float*>(scratch.data);
    auto* d_qkv = static_cast<const nv_bfloat16*>(qkv.data);
    auto* d_k_cache = static_cast<const nv_bfloat16*>(k_cache.data);
    auto* d_v_cache = static_cast<const nv_bfloat16*>(v_cache.data);
    auto* d_block_tables = static_cast<const int*>(block_tables.data);
    auto* d_context_lens = static_cast<const int*>(context_lens.data);

    int thread_block_size = 256;
    dim3 partial_grid(num_splits, heads, batch_size);

    size_t shared_mem_bytes = (static_cast<size_t>(3 * thread_block_size) * sizeof(float)) +
                              (static_cast<size_t>(head_dim) * sizeof(float)) +
                              (static_cast<size_t>(head_dim) * sizeof(nv_bfloat16));

    paged_attention_split_partial_kernel<<<partial_grid, thread_block_size, shared_mem_bytes,
                                           stream>>>(
        d_scratch, d_qkv, d_k_cache, d_v_cache, d_block_tables, d_context_lens, batch_size,
        num_splits, heads, kv_heads, head_dim, block_size, max_blocks);

    KERNEL_CHECK(stream);

    dim3 merge_grid(heads, batch_size);
    int merge_block_size = std::min(head_dim, 128);

    paged_attention_split_merge_kernel<<<merge_grid, merge_block_size, 0, stream>>>(
        d_out, d_scratch, d_context_lens, num_splits, heads, head_dim, block_size);

    KERNEL_CHECK(stream);
}

}  // namespace engine
