#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "cuda_check.h"
#include "engine/kernels/rope.h"

namespace engine::fast {

namespace {

constexpr int kMaxHeadDim = 256;

struct InvFreq {
    float v[kMaxHeadDim / 2];
};

}  // namespace

__global__ void rope_kernel(nv_bfloat16* __restrict__ qkv_data,
                            const int* __restrict__ positions_data, int num_query_heads,
                            int num_kv_heads, int head_dim, InvFreq inv_freq_table) {
    int token_idx = blockIdx.x;
    int token_pos = positions_data[token_idx];

    int half_head_dim = head_dim / 2;
    int total_rounded_heads = num_query_heads + num_kv_heads;

    int row_stride_elements = (num_query_heads + (2 * num_kv_heads)) * head_dim;
    nv_bfloat16* row_qkv_ptr = qkv_data + (static_cast<int64_t>(token_idx) * row_stride_elements);

    for (int head_idx = threadIdx.x; head_idx < total_rounded_heads; head_idx += blockDim.x) {
        nv_bfloat16* head_ptr = row_qkv_ptr + static_cast<int64_t>(head_idx) * head_dim;

        for (int i = 0; i < half_head_dim; ++i) {
            float rot_ang = static_cast<float>(token_pos) * inv_freq_table.v[i];

            float comp_cos = cosf(rot_ang);
            float comp_sin = sinf(rot_ang);

            nv_bfloat16 cos_bf16 = __float2bfloat16(comp_cos);
            nv_bfloat16 sin_bf16 = __float2bfloat16(comp_sin);

            nv_bfloat16 low_bf16 = head_ptr[i];
            nv_bfloat16 high_bf16 = head_ptr[i + half_head_dim];

            float lo_float = __bfloat162float(low_bf16);
            float hi_float = __bfloat162float(high_bf16);
            float cos_float = __bfloat162float(cos_bf16);
            float sin_float = __bfloat162float(sin_bf16);

            float prod_lo_cos = lo_float * cos_float;
            float prod_hi_sin = hi_float * sin_float;
            float prod_hi_cos = hi_float * cos_float;
            float prod_lo_sin = lo_float * sin_float;

            nv_bfloat16 term1_bf16 = __float2bfloat16(prod_lo_cos);
            nv_bfloat16 term2_bf16 = __float2bfloat16(prod_hi_sin);
            nv_bfloat16 term3_bf16 = __float2bfloat16(prod_hi_cos);
            nv_bfloat16 term4_bf16 = __float2bfloat16(prod_lo_sin);

            float t1 = __bfloat162float(term1_bf16);
            float t2 = __bfloat162float(term2_bf16);
            float t3 = __bfloat162float(term3_bf16);
            float t4 = __bfloat162float(term4_bf16);

            float rotated_lo_fp32 = t1 - t2;
            float rotated_hi_fp32 = t3 + t4;

            head_ptr[i] = __float2bfloat16(rotated_lo_fp32);
            head_ptr[i + half_head_dim] = __float2bfloat16(rotated_hi_fp32);
        }
    }
}

void rope(const Tensor& qkv, const Tensor& positions, int heads, int kv_heads, float theta,
          Stream stream) {
    if (heads <= 0) {
        throw std::invalid_argument("[rope] number of query heads must be positive");
    }

    if (kv_heads <= 0) {
        throw std::invalid_argument("[rope] number of key/value heads must be positive");
    }

    if (qkv.dtype != DType::BF16) {
        throw std::invalid_argument("[rope] qkv tensor must have BF16 data type");
    }

    if (qkv.ndim != 2) {
        throw std::invalid_argument("[rope] qkv tensor must be 2-dimensional");
    }

    if (positions.dtype != DType::I32) {
        throw std::invalid_argument("[rope] positions tensor must have I32 data type");
    }

    if (positions.ndim != 1) {
        throw std::invalid_argument("[rope] positions tensor must be 1-dimensional");
    }

    int64_t num_tokens = qkv.shape[0];

    if (positions.shape[0] != num_tokens) {
        throw std::invalid_argument("[rope] positions tensor length must match token count T");
    }

    int64_t row_width_elements = qkv.shape[1];
    int total_heads_layout = heads + (2 * kv_heads);

    if (row_width_elements % total_heads_layout != 0) {
        throw std::invalid_argument(
            "[rope] qkv row width is not a multiple of heads + 2 * kv_heads");
    }

    int derived_head_dim = static_cast<int>(row_width_elements / total_heads_layout);

    if (derived_head_dim % 2 != 0 || derived_head_dim > kMaxHeadDim) {
        throw std::invalid_argument("[rope] head_dim must be even and no greater than 256");
    }

    if (num_tokens == 0) {
        return;
    }

    auto* qkv_device_ptr = static_cast<nv_bfloat16*>(qkv.data);
    auto* positions_device_ptr = static_cast<const int*>(positions.data);

    InvFreq inv_freq{};
    for (int i = 0; i < derived_head_dim / 2; ++i) {
        inv_freq.v[i] = 1.0f / std::pow(theta, static_cast<float>(2 * i) /
                                                   static_cast<float>(derived_head_dim));
    }

    int threads_per_block = 128;
    int num_blocks = static_cast<int>(num_tokens);

    rope_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        qkv_device_ptr, positions_device_ptr, heads, kv_heads, derived_head_dim, inv_freq);

    KERNEL_CHECK(stream);
}

}  // namespace engine::fast
