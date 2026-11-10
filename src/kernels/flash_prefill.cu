#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

#include "engine/core/tensor.h"
#include "engine/kernels/flash_prefill.h"
#include "kernels/cuda_check.h"

namespace engine {

namespace {

constexpr int kTile = 16;
constexpr int kWarpsPerBlock = 4;

__device__ __forceinline__ float half_bf16(uint32_t word, int odd) {
    return __bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(word >> (16 * odd))));
}

__device__ __forceinline__ void mma_bf16(float acc[4], const uint32_t a[4], uint32_t b0,
                                         uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && !defined(ENGINE_FLASH_EMULATE_MMA)
    float d0, d1, d2, d3;
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(acc[0]), "f"(acc[1]),
          "f"(acc[2]), "f"(acc[3]));
    acc[0] = d0;
    acc[1] = d1;
    acc[2] = d2;
    acc[3] = d3;
#else
    const int lane = int(threadIdx.x) % 32;
    const int group = lane / 4;
    const int quad = lane % 4;
    for (int pair = 0; pair < 8; ++pair) {
        const int source = pair % 4;
        const bool upper = pair >= 4;
        const uint32_t a_low = __shfl_sync(0xffffffff, upper ? a[2] : a[0], 4 * group + source);
        const uint32_t a_high = __shfl_sync(0xffffffff, upper ? a[3] : a[1], 4 * group + source);
        const uint32_t b_even = __shfl_sync(0xffffffff, upper ? b1 : b0, 8 * quad + source);
        const uint32_t b_odd = __shfl_sync(0xffffffff, upper ? b1 : b0, 8 * quad + 4 + source);
        for (int odd = 0; odd < 2; ++odd) {
            acc[0] = __fmaf_rn(half_bf16(a_low, odd), half_bf16(b_even, odd), acc[0]);
            acc[1] = __fmaf_rn(half_bf16(a_low, odd), half_bf16(b_odd, odd), acc[1]);
            acc[2] = __fmaf_rn(half_bf16(a_high, odd), half_bf16(b_even, odd), acc[2]);
            acc[3] = __fmaf_rn(half_bf16(a_high, odd), half_bf16(b_odd, odd), acc[3]);
        }
    }
#endif
}

__device__ __forceinline__ float round_bf16(float x) {
    return __bfloat162float(__float2bfloat16(x));
}

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
    return uint32_t(__bfloat16_as_ushort(__float2bfloat16(lo))) |
           (uint32_t(__bfloat16_as_ushort(__float2bfloat16(hi))) << 16);
}

__device__ __forceinline__ float group_max(float x) {
    x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, 1));
    return fmaxf(x, __shfl_xor_sync(0xffffffff, x, 2));
}

__device__ __forceinline__ float group_sum(float x) {
    x += __shfl_xor_sync(0xffffffff, x, 1);
    return x + __shfl_xor_sync(0xffffffff, x, 2);
}

template <int HeadDim>
__global__ void flash_attention_prefill_kernel(
    nv_bfloat16* __restrict__ out, const uint32_t* __restrict__ qkv,
    const uint32_t* __restrict__ k_cache, const uint32_t* __restrict__ v_cache,
    const int32_t* __restrict__ block_tables, const int32_t* __restrict__ chunk_starts,
    const int32_t* __restrict__ cached_lens, int num_chunks, int total_rows, int query_heads,
    int kv_heads, int block_size, int blocks_per_chunk) {
    constexpr int kSteps = HeadDim / 16;
    constexpr int kCols = HeadDim / 8;
    constexpr int kRowWords = HeadDim / 2;

    const int head = int(blockIdx.y) * kWarpsPerBlock + int(threadIdx.x) / 32;
    if (head >= query_heads) return;
    const int lane = int(threadIdx.x) % 32;
    const int group = lane / 4;
    const int quad = lane % 4;
    const int kv_head = head / (query_heads / kv_heads);
    const int64_t row_words = int64_t(query_heads + 2 * kv_heads) * kRowWords;
    const int64_t k_column = int64_t(query_heads + kv_head) * kRowWords;
    const int64_t v_column = k_column + int64_t(kv_heads) * kRowWords;
    const float scale = 1.0f / sqrtf(float(HeadDim));
    const float masked = -INFINITY;

    const int64_t tile_first = int64_t(blockIdx.x) * kTile;
    const int64_t tile_end = min(tile_first + kTile, int64_t(total_rows));
    const int64_t rows[2] = {tile_first + group, tile_first + group + 8};

    int chunk = 0;
    for (int high = num_chunks; chunk < high;) {
        const int mid = chunk + (high - chunk) / 2;
        if (chunk_starts[mid + 1] > tile_first)
            high = mid;
        else
            chunk = mid + 1;
    }

    for (; chunk < num_chunks && chunk_starts[chunk] < tile_end; ++chunk) {
        const int64_t chunk_first = chunk_starts[chunk];
        const int64_t first = max(chunk_first, tile_first);
        const int64_t end = min(int64_t(chunk_starts[chunk + 1]), tile_end);
        const int64_t cached = cached_lens[chunk];
        if (first >= end || cached < 0 || cached > int64_t(blocks_per_chunk) * block_size) continue;
        const int64_t last_position = cached + (end - 1 - chunk_first);
        const int32_t* table = block_tables + int64_t(chunk) * blocks_per_chunk;

        bool active[2];
        int64_t position[2];
        uint32_t q[kSteps][4];
        for (int r = 0; r < 2; ++r) {
            active[r] = rows[r] >= first && rows[r] < end;
            position[r] = cached + (rows[r] - chunk_first);
            const uint32_t* q_row = qkv + rows[r] * row_words + int64_t(head) * kRowWords;
            for (int step = 0; step < kSteps; ++step) {
                q[step][r] = active[r] ? q_row[8 * step + quad] : 0u;
                q[step][2 + r] = active[r] ? q_row[8 * step + quad + 4] : 0u;
            }
        }

        float running_max[2] = {-1e30f, -1e30f};
        float running_sum[2] = {0.0f, 0.0f};
        float acc[kCols][4];
        for (int col = 0; col < kCols; ++col)
            for (int i = 0; i < 4; ++i) acc[col][i] = 0.0f;

        const auto key_rows = [&](int64_t key, const uint32_t*& k_row, const uint32_t*& v_row) {
            k_row = nullptr;
            v_row = nullptr;
            if (key > last_position) return;
            if (key < cached) {
                const uint32_t key32 = uint32_t(key);
                const uint32_t size32 = uint32_t(block_size);
                const int64_t block = table[key32 / size32];
                const int64_t words =
                    ((block * kv_heads + kv_head) * block_size + key32 % size32) * kRowWords;
                k_row = k_cache + words;
                v_row = v_cache + words;
            } else {
                const uint32_t* row = qkv + (chunk_first + key - cached) * row_words;
                k_row = row + k_column;
                v_row = row + v_column;
            }
        };

        for (int64_t tile_key = 0; tile_key <= last_position; tile_key += kTile) {
            const int key_offset[4] = {2 * quad, 2 * quad + 1, 2 * quad + 8, 2 * quad + 9};
            float score[2][4] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
            for (int half = 0; half < 2; ++half) {
                const uint32_t *k_row, *unused;
                key_rows(tile_key + 8 * half + group, k_row, unused);
                for (int step = 0; step < kSteps; ++step) {
                    const uint32_t b0 = k_row ? k_row[8 * step + quad] : 0u;
                    const uint32_t b1 = k_row ? k_row[8 * step + quad + 4] : 0u;
                    mma_bf16(score[half], q[step], b0, b1);
                }
            }

            float weight[2][4];
            float rescale[2];
            for (int r = 0; r < 2; ++r) {
                float s[4];
                float tile_max = masked;
                for (int i = 0; i < 4; ++i) {
                    const float rounded =
                        round_bf16(round_bf16(score[i / 2][2 * r + i % 2]) * scale);
                    const bool visible = active[r] && tile_key + key_offset[i] <= position[r];
                    s[i] = visible ? rounded : masked;
                    tile_max = fmaxf(tile_max, s[i]);
                }
                const float new_max = fmaxf(running_max[r], group_max(tile_max));
                rescale[r] = expf(running_max[r] - new_max);
                float sum = 0.0f;
                for (int i = 0; i < 4; ++i) {
                    weight[r][i] = round_bf16(expf(s[i] - new_max));
                    sum += weight[r][i];
                }
                running_sum[r] = running_sum[r] * rescale[r] + group_sum(sum);
                running_max[r] = new_max;
            }
            const uint32_t p[4] = {
                pack_bf16(weight[0][0], weight[0][1]), pack_bf16(weight[1][0], weight[1][1]),
                pack_bf16(weight[0][2], weight[0][3]), pack_bf16(weight[1][2], weight[1][3])};

            const uint32_t* v_row[4];
            for (int i = 0; i < 4; ++i) {
                const uint32_t* unused;
                key_rows(tile_key + key_offset[i], unused, v_row[i]);
            }
            for (int pair = 0; pair < kCols / 2; ++pair) {
                uint32_t v[4];
                for (int i = 0; i < 4; ++i)
                    v[i] = v_row[i] ? v_row[i][group * (kCols / 2) + pair] : 0u;
                for (int odd = 0; odd < 2; ++odd) {
                    const int col = 2 * pair + odd;
                    const int shift = 16 * odd;
                    const uint32_t b0 = ((v[0] >> shift) & 0xffffu) | ((v[1] >> shift) << 16);
                    const uint32_t b1 = ((v[2] >> shift) & 0xffffu) | ((v[3] >> shift) << 16);
                    acc[col][0] *= rescale[0];
                    acc[col][1] *= rescale[0];
                    acc[col][2] *= rescale[1];
                    acc[col][3] *= rescale[1];
                    mma_bf16(acc[col], p, b0, b1);
                }
            }
        }

        for (int r = 0; r < 2; ++r) {
            if (!active[r]) continue;
            const float inverse = 1.0f / running_sum[r];
            nv_bfloat16* out_row = out + (rows[r] * query_heads + head) * int64_t(HeadDim);
            for (int col = 0; col < kCols; ++col) {
                out_row[2 * quad * kCols + col] = __float2bfloat16(acc[col][2 * r] * inverse);
                out_row[(2 * quad + 1) * kCols + col] =
                    __float2bfloat16(acc[col][2 * r + 1] * inverse);
            }
        }
    }
}

template <int HeadDim>
void launch(dim3 grid, cudaStream_t stream, const Tensor& out, const Tensor& qkv,
            const Tensor& k_cache, const Tensor& v_cache, const Tensor& block_tables,
            const Tensor& chunk_starts, const Tensor& cached_lens, int query_heads) {
    flash_attention_prefill_kernel<HeadDim><<<grid, 32 * kWarpsPerBlock, 0, stream>>>(
        static_cast<nv_bfloat16*>(out.data), static_cast<const uint32_t*>(qkv.data),
        static_cast<const uint32_t*>(k_cache.data), static_cast<const uint32_t*>(v_cache.data),
        static_cast<const int32_t*>(block_tables.data),
        static_cast<const int32_t*>(chunk_starts.data),
        static_cast<const int32_t*>(cached_lens.data), int(cached_lens.shape[0]), int(out.shape[0]),
        query_heads, int(k_cache.shape[1]), int(k_cache.shape[2]), int(block_tables.shape[1]));
}

}  // namespace

void flash_attention_prefill(const Tensor& out, const Tensor& qkv, const Tensor& k_cache,
                             const Tensor& v_cache, const Tensor& block_tables,
                             const Tensor& chunk_starts, const Tensor& cached_lens, Stream stream) {
    if (k_cache.dtype != DType::BF16 || k_cache.ndim != 4) {
        throw std::invalid_argument("[flash_attention_prefill] k_cache must be 4D BF16 tensor");
    }

    if (v_cache.dtype != DType::BF16 || v_cache.ndim != 4) {
        throw std::invalid_argument("[flash_attention_prefill] v_cache must be 4D BF16 tensor");
    }

    if (k_cache.shape != v_cache.shape) {
        throw std::invalid_argument(
            "[flash_attention_prefill] k_cache and v_cache shapes must match");
    }

    int64_t head_dim = k_cache.shape[3];

    if (head_dim <= 0 || head_dim % 16 != 0 || head_dim > 128) {
        throw std::invalid_argument(
            "[flash_attention_prefill] head_dim must be a multiple of 16 and <= 128");
    }

    if (out.dtype != DType::BF16 || out.ndim != 2) {
        throw std::invalid_argument("[flash_attention_prefill] out must be 2D BF16 tensor");
    }

    if (out.shape[1] % head_dim != 0) {
        throw std::invalid_argument(
            "[flash_attention_prefill] out width must be a multiple of head_dim");
    }

    int64_t num_query_heads = out.shape[1] / head_dim;
    int64_t num_kv_heads = k_cache.shape[1];

    if (num_kv_heads <= 0 || num_query_heads <= 0 || num_query_heads % num_kv_heads != 0) {
        throw std::invalid_argument(
            "[flash_attention_prefill] heads must be a positive multiple of kv_heads");
    }

    if (num_query_heads > 65535) {
        throw std::invalid_argument("[flash_attention_prefill] query heads must be at most 65535");
    }

    if (qkv.dtype != DType::BF16 || qkv.ndim != 2) {
        throw std::invalid_argument("[flash_attention_prefill] qkv must be 2D BF16 tensor");
    }

    if (qkv.shape[0] != out.shape[0]) {
        throw std::invalid_argument(
            "[flash_attention_prefill] qkv row count must match out row count");
    }

    int64_t expected_qkv_width = (num_query_heads + (2 * num_kv_heads)) * head_dim;
    if (qkv.shape[1] != expected_qkv_width) {
        throw std::invalid_argument(
            "[flash_attention_prefill] qkv width must be (heads + (2 * kv_heads)) * head_dim");
    }

    if (chunk_starts.dtype != DType::I32 || chunk_starts.ndim != 1 || chunk_starts.numel() == 0) {
        throw std::invalid_argument(
            "[flash_attention_prefill] chunk_starts must be 1D non-empty I32 tensor");
    }

    int64_t num_chunks = chunk_starts.shape[0] - 1;

    if (cached_lens.dtype != DType::I32 || cached_lens.ndim != 1 ||
        cached_lens.shape[0] != num_chunks) {
        throw std::invalid_argument(
            "[flash_attention_prefill] cached_lens must be 1D I32 tensor of length C");
    }

    if (block_tables.dtype != DType::I32 || block_tables.ndim != 2 ||
        block_tables.shape[0] != num_chunks || block_tables.shape[1] <= 0) {
        throw std::invalid_argument(
            "[flash_attention_prefill] block_tables must be 2D I32 tensor with C rows and >0 cols");
    }

    int64_t total_step_rows = out.shape[0];

    if (total_step_rows > INT32_MAX) {
        throw std::invalid_argument("[flash_attention_prefill] row count must fit in int");
    }

    if (num_chunks == 0 || total_step_rows == 0) {
        return;
    }

    for (const Tensor* t : {&qkv, &k_cache, &v_cache}) {
        if (reinterpret_cast<uintptr_t>(t->data) % alignof(uint32_t) != 0) {
            throw std::invalid_argument(
                "[flash_attention_prefill] qkv, k_cache and v_cache must be 4-byte aligned");
        }
    }

    const dim3 grid(
        static_cast<unsigned int>((total_step_rows + kTile - 1) / kTile),
        static_cast<unsigned int>((num_query_heads + kWarpsPerBlock - 1) / kWarpsPerBlock));
    const auto execution_stream = static_cast<cudaStream_t>(stream);
    const int heads = static_cast<int>(num_query_heads);

    switch (head_dim) {
        case 16:
            launch<16>(grid, execution_stream, out, qkv, k_cache, v_cache, block_tables,
                       chunk_starts, cached_lens, heads);
            break;
        case 32:
            launch<32>(grid, execution_stream, out, qkv, k_cache, v_cache, block_tables,
                       chunk_starts, cached_lens, heads);
            break;
        case 48:
            launch<48>(grid, execution_stream, out, qkv, k_cache, v_cache, block_tables,
                       chunk_starts, cached_lens, heads);
            break;
        case 64:
            launch<64>(grid, execution_stream, out, qkv, k_cache, v_cache, block_tables,
                       chunk_starts, cached_lens, heads);
            break;
        case 80:
            launch<80>(grid, execution_stream, out, qkv, k_cache, v_cache, block_tables,
                       chunk_starts, cached_lens, heads);
            break;
        case 96:
            launch<96>(grid, execution_stream, out, qkv, k_cache, v_cache, block_tables,
                       chunk_starts, cached_lens, heads);
            break;
        case 112:
            launch<112>(grid, execution_stream, out, qkv, k_cache, v_cache, block_tables,
                        chunk_starts, cached_lens, heads);
            break;
        default:
            launch<128>(grid, execution_stream, out, qkv, k_cache, v_cache, block_tables,
                        chunk_starts, cached_lens, heads);
            break;
    }

    KERNEL_CHECK(stream);
}

}  // namespace engine
