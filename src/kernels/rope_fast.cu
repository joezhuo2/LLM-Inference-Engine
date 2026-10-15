#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
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
                            const int* __restrict__ positions_data, int rotated_heads, int width,
                            int head_dim, InvFreq inv_freq_table) {
    const int half_head_dim = head_dim / 2;
    const int pair = threadIdx.x % half_head_dim;
    const int first_head = threadIdx.x / half_head_dim;
    const int head_stride = blockDim.x / half_head_dim;

    const float angle = static_cast<float>(positions_data[blockIdx.x]) * inv_freq_table.v[pair];
    const float cos_val = __bfloat162float(__float2bfloat16(cosf(angle)));
    const float sin_val = __bfloat162float(__float2bfloat16(sinf(angle)));

    nv_bfloat16* row = qkv_data + static_cast<int64_t>(blockIdx.x) * width;

    for (int head = first_head; head < rotated_heads; head += head_stride) {
        nv_bfloat16* head_ptr = row + static_cast<int64_t>(head) * head_dim;

        const float lo = __bfloat162float(head_ptr[pair]);
        const float hi = __bfloat162float(head_ptr[pair + half_head_dim]);

        const float lo_cos = __bfloat162float(__float2bfloat16(lo * cos_val));
        const float hi_cos = __bfloat162float(__float2bfloat16(hi * cos_val));
        const float lo_sin = __bfloat162float(__float2bfloat16(lo * sin_val));
        const float hi_sin = __bfloat162float(__float2bfloat16(-hi * sin_val));

        head_ptr[pair] = __float2bfloat16(lo_cos + hi_sin);
        head_ptr[pair + half_head_dim] = __float2bfloat16(hi_cos + lo_sin);
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

    int head_groups = std::min(heads + kv_heads, 256 / (derived_head_dim / 2));
    int threads_per_block = (derived_head_dim / 2) * std::max(head_groups, 1);
    int num_blocks = static_cast<int>(num_tokens);

    rope_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        qkv_device_ptr, positions_device_ptr, heads + kv_heads,
        static_cast<int>(row_width_elements), derived_head_dim, inv_freq);

    KERNEL_CHECK(stream);
}

}  // namespace engine::fast
