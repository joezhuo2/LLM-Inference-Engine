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
    int token_row_idx = blockIdx.y;
    int head_idx = blockIdx.x;

    if (head_idx >= query_heads || token_row_idx >= total_step_rows) {
        return;
    }

    int low = 0;
    int high = num_chunks - 1;
    int chunk_idx = -1;

    while (low <= high) {
        int mid = low + (high - low) / 2;

        int start = chunk_starts[mid];
        int next = chunk_starts[mid + 1];

        if (token_row_idx >= start && token_row_idx < next) {
            chunk_idx = mid;
            break;
        } else if (token_row_idx < start) {
            high = mid - 1;
        } else {
            low = mid + 1;
        }
    }

    if (chunk_idx < 0) {
        return;
    }
    
    int lane = threadIdx.x;
    int query_heads_per_kv_head = query_heads / kv_heads;
    int kv_head_idx = head_idx / query_heads_per_kv_head;

    int chunk_relative_token_idx = token_row_idx - chunk_starts[chunk_idx];
    int context_cached_length = cached_lens[chunk_idx];
    int seq_casual_boundary = context_cached_length + chunk_relative_token_idx;

    float softmax_scale_factor = 1.0f / sqrtf(static_cast<float>(head_dim));
    int qkv_row_stride = (query_heads + (2 * kv_heads)) * head_dim;
    int query_buffer_offset = (token_row_idx * qkv_row_stride) + (head_idx * head_dim);

    nv_bfloat16 query_element = __float2bfloat16(0.0f);

    if (lane < head_dim) {
        query_element = qkv[query_buffer_offset + lane];
    }

    float running_max = -1e30f;
    float running_sum = 0.0f;
    float running_accumulated_val = 0.0f;

    __shared__ float warp_reductions[4];
    __shared__ float final_score;

    for (int context_pos = 0; context_pos <= seq_casual_boundary; ++context_pos) {
        nv_bfloat16 key = __float2bfloat16(0.0f);
        nv_bfloat16 val = __float2bfloat16(0.0f);

        if (lane < head_dim) {
            if (context_pos < context_cached_length) {
                int logical_block_idx = context_pos / block_size;
                int physical_block_idx =
                    block_tables[(chunk_idx * blocks_per_chunk) + logical_block_idx];
                int block_token_os = context_pos % block_size;

                int block_head_idx = (physical_block_idx * kv_heads) + kv_head_idx;
                int token_slot_idx = (block_head_idx * block_size) + block_token_os;
                int cache_os = (token_slot_idx * head_dim) + lane;

                key = k_cache[cache_os];
                val = v_cache[cache_os];
            } else {
                int source_chunk_os = context_pos - context_cached_length;
                int source_global_row = chunk_starts[chunk_idx] + source_chunk_os;
                int row_start_os = source_global_row * qkv_row_stride;

                int k_section_os = row_start_os + (query_heads * head_dim);
                int k_head_os = k_section_os + (kv_head_idx * head_dim);
                int k_os = k_head_os + lane;

                int v_section_os = k_section_os + (kv_heads * head_dim);
                int v_head_os = v_section_os + (kv_head_idx * head_dim);
                int v_os = v_head_os + lane;

                key = qkv[k_os];
                val = qkv[v_os];
            }
        }

        float partial_dot_product =
            (lane < head_dim) ? __bfloat162float(query_element) * __bfloat162float(key) : 0.0f;
        
        for (int shuffle_os = 16; shuffle_os > 0; shuffle_os /= 2) {
            partial_dot_product += __shfl_down_sync(0xffffffff, partial_dot_product, shuffle_os);
        }

        int warp_id = lane / 32;
        int lane_in_warp = lane % 32;

        if (lane_in_warp == 0) {
            warp_reductions[warp_id] = partial_dot_product;
        }

        __syncthreads();

        if (warp_id == 0) {
            float total_dot_product = (lane_in_warp < 4) ? warp_reductions[lane_in_warp] : 0.0f;

            for (int shuffle_os = 2; shuffle_os > 0; shuffle_os /= 2) {
                total_dot_product += __shfl_down_sync(0xffffffff, total_dot_product, shuffle_os);
            }

            if (lane_in_warp == 0) {
                nv_bfloat16 dot_bf16 = __float2bfloat16(total_dot_product);
                float dot_fp32 = __bfloat162float(dot_bf16);
                nv_bfloat16 scaled_score_bf16 = __float2bfloat16(dot_fp32 * softmax_scale_factor);
                final_score = __bfloat162float(scaled_score_bf16);
            }
        }

        __syncthreads();

        float score_at_pos_j = final_score;

        if (score_at_pos_j > running_max) {
            float new_max = score_at_pos_j;
            float scale_rescale_factor = expf(running_max - new_max);

            running_max = new_max;
            running_sum = (running_sum * scale_rescale_factor) + 1.0f;
            running_accumulated_val = (running_accumulated_val * scale_rescale_factor) + __bfloat162float(val);
        } else {
            float unnormalized_weight = expf(score_at_pos_j - running_max);
            running_sum += unnormalized_weight;
            running_accumulated_val += unnormalized_weight * __bfloat162float(val);
        }
    }

    if (lane < head_dim) {
        float inverse_running_sum = 1.0f / running_sum;
        int output_element_os = token_row_idx * (query_heads * head_dim) + (head_idx * head_dim) + lane;
        out[output_element_os] = __float2bfloat16(running_accumulated_val * inverse_running_sum);
    }
}

void flash_attention_prefill(const Tensor& out, const Tensor& qkv, const Tensor& k_cache,
                             const Tensor& v_cache, const Tensor& block_tables,
                             const Tensor& chunk_starts, const Tensor& cached_lens, Stream stream) {

}

}  // namespace engine
