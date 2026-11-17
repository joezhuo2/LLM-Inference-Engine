#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine/kernels/fp8_kv.h"
#include "kernels/cuda_check.h"
#include "support/bf16.h"
#include "support/device.h"
#include "support/fp8.h"
#include "support/paged_attention_harness.h"

using engine::DeviceBuffer;
using engine::DType;
using engine::paged_attention_decode_fp8;
using engine::Tensor;
using engine::write_kv_fp8;
using engine::test::compare_bf16;
using engine::test::CudaStream;
using engine::test::download;
using engine::test::fp8_is_nan;
using engine::test::from_bf16;
using engine::test::from_fp8;
using engine::test::random_bf16;
using engine::test::to_bf16;
using engine::test::to_fp8;
using engine::test::upload;

namespace {
using namespace engine::test::pa;

struct Case {
    int heads, kv_heads, head_dim, num_blocks, block_size;
    int64_t kv_width() const { return int64_t(kv_heads) * head_dim; }
    int64_t width() const { return int64_t(heads) * head_dim + 2 * kv_width(); }
    int64_t cache_elems() const { return int64_t(num_blocks) * block_size * kv_width(); }
};

std::vector<uint8_t> random_bytes(size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<uint8_t> out(n);
    for (auto& b : out) b = uint8_t(rng());
    return out;
}

std::vector<int32_t> sequence_slots(const std::vector<int>& table, int length, int block_size) {
    std::vector<int32_t> slots;
    for (int pos = 0; pos < length; ++pos)
        slots.push_back(table[size_t(pos / block_size)] * block_size + pos % block_size);
    return slots;
}

std::vector<int> random_blocks(int count, int num_blocks, std::mt19937& rng) {
    std::vector<int> ids(static_cast<size_t>(num_blocks));
    std::iota(ids.begin(), ids.end(), 0);
    std::shuffle(ids.begin(), ids.end(), rng);
    ids.resize(size_t(count));
    return ids;
}

void check_write(const Case& c, const std::vector<int32_t>& slots,
                 const std::vector<uint16_t>& qkv_host, float k_scale, float v_scale,
                 uint32_t seed) {
    const auto tokens = int64_t(slots.size());
    const int64_t cache = c.cache_elems();
    const int64_t k_begin = kGuard, v_begin = 2 * kGuard + cache;
    const int64_t total = 3 * kGuard + 2 * cache;
    ASSERT_EQ(int64_t(qkv_host.size()), std::max<int64_t>(tokens, 1) * c.width());

    const auto memory_host = random_bytes(size_t(total), seed);
    DeviceBuffer qkv_buf = upload(qkv_host);
    DeviceBuffer memory = upload(memory_host);
    DeviceBuffer slot_buf = upload(slots.empty() ? std::vector<int32_t>{7} : slots);

    auto* base = static_cast<uint8_t*>(memory.data());
    const Tensor k(base + k_begin, DType::F8E4M3,
                   {c.num_blocks, c.kv_heads, c.block_size, c.head_dim});
    const Tensor v(base + v_begin, DType::F8E4M3,
                   {c.num_blocks, c.kv_heads, c.block_size, c.head_dim});
    const Tensor qkv(qkv_buf.data(), DType::BF16, {tokens, c.width()});
    const Tensor slot_mapping(slot_buf.data(), DType::I32, {tokens});

    CudaStream stream;
    write_kv_fp8(k, v, qkv, slot_mapping, k_scale, v_scale, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<uint8_t> expected = memory_host;
    for (int64_t t = 0; t < tokens; ++t) {
        const int32_t s = slots[size_t(t)];
        if (s < 0) continue;
        const int64_t block = s / c.block_size, offset = s % c.block_size;
        for (int h = 0; h < c.kv_heads; ++h)
            for (int d = 0; d < c.head_dim; ++d) {
                const int64_t cell =
                    ((block * c.kv_heads + h) * c.block_size + offset) * c.head_dim + d;
                const int64_t col = h * int64_t(c.head_dim) + d;
                const int64_t row = t * c.width() + c.width() - 2 * c.kv_width();
                expected[size_t(k_begin + cell)] =
                    to_fp8(from_bf16(qkv_host[size_t(row + col)]) / k_scale);
                expected[size_t(v_begin + cell)] =
                    to_fp8(from_bf16(qkv_host[size_t(row + c.kv_width() + col)]) / v_scale);
            }
    }

    const auto actual = download<uint8_t>(memory);
    int64_t wrong = 0, first = -1;
    for (int64_t i = 0; i < total; ++i) {
        const uint8_t a = actual[size_t(i)], e = expected[size_t(i)];
        const bool same = a == e || (fp8_is_nan(a) && fp8_is_nan(e));
        if (!same && wrong++ == 0) first = i;
    }
    EXPECT_EQ(wrong, 0) << "first wrong byte " << first << " (K starts at " << k_begin << ", V at "
                        << v_begin << ", each " << cache << " bytes): got "
                        << int(actual[size_t(std::max<int64_t>(first, 0))]) << ", expected "
                        << int(expected[size_t(std::max<int64_t>(first, 0))]);
    EXPECT_EQ(download<uint16_t>(qkv_buf), qkv_host) << "qkv was modified";
    if (!slots.empty()) {
        EXPECT_EQ(download<int32_t>(slot_buf), slots) << "slot_mapping was modified";
    }
}

void check_write(const Case& c, const std::vector<int32_t>& slots, float k_scale, float v_scale,
                 uint32_t seed, float range = 4.0f) {
    const auto rows = size_t(std::max<size_t>(slots.size(), 1));
    check_write(c, slots, random_bf16(rows * size_t(c.width()), seed + 1, -range, range), k_scale,
                v_scale, seed);
}

struct Fp8Batch {
    Batch b;
    std::vector<uint8_t> k, v;
    float k_scale = 1.0f, v_scale = 1.0f;
};

Fp8Batch quantize(Batch b, float k_scale = 1.0f, float v_scale = 1.0f) {
    Fp8Batch f{std::move(b), {}, {}, k_scale, v_scale};
    f.k.resize(f.b.k.size());
    f.v.resize(f.b.v.size());
    for (size_t i = 0; i < f.k.size(); ++i) {
        f.k[i] = to_fp8(from_bf16(f.b.k[i]) / k_scale);
        f.v[i] = to_fp8(from_bf16(f.b.v[i]) / v_scale);
        f.b.k[i] = to_bf16(from_fp8(f.k[i]) * k_scale);
        f.b.v[i] = to_bf16(from_fp8(f.v[i]) * v_scale);
    }
    return f;
}

std::vector<uint16_t> run_fp8(const Fp8Batch& f, uint32_t seed) {
    const Batch& b = f.b;
    const Layout& l = b.l;
    const int64_t n = b.rows() * l.q_width();
    const auto init = random_bf16(size_t(n + 2 * kGuard), seed);
    DeviceBuffer out_buf = upload(init);
    DeviceBuffer qkv = upload(b.qkv), k = upload(f.k), v = upload(f.v);
    DeviceBuffer tables = upload(b.tables), lens = upload(b.lens);
    const std::initializer_list<int64_t> cache_shape{b.num_blocks, l.kv_heads, l.block_size,
                                                     l.head_dim};

    CudaStream stream;
    paged_attention_decode_fp8(
        Tensor(static_cast<uint16_t*>(out_buf.data()) + kGuard, DType::BF16,
               {b.rows(), l.q_width()}),
        Tensor(qkv.data(), DType::BF16, {b.rows(), l.width()}),
        Tensor(k.data(), DType::F8E4M3, cache_shape), Tensor(v.data(), DType::F8E4M3, cache_shape),
        Tensor(tables.data(), DType::I32, {b.rows(), b.table_width}),
        Tensor(lens.data(), DType::I32, {b.rows()}), f.k_scale, f.v_scale, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    EXPECT_EQ(download<uint16_t>(qkv), b.qkv) << "qkv was modified";
    EXPECT_EQ(download<uint8_t>(k), f.k) << "k_cache was modified";
    EXPECT_EQ(download<uint8_t>(v), f.v) << "v_cache was modified";
    EXPECT_EQ(download<int32_t>(tables), b.tables) << "block_tables was modified";
    EXPECT_EQ(download<int32_t>(lens), b.lens) << "context_lens was modified";

    const auto after = download<uint16_t>(out_buf);
    for (int64_t i = 0; i < kGuard; ++i) {
        EXPECT_EQ(after[size_t(i)], init[size_t(i)]) << "write before out at " << i - kGuard;
        EXPECT_EQ(after[size_t(kGuard + n + i)], init[size_t(kGuard + n + i)])
            << "write after out at " << i;
    }
    for (int64_t r = 0; r < b.rows(); ++r) {
        if (b.lens[size_t(r)] != 0) continue;
        const auto row = size_t(kGuard + r * l.q_width());
        EXPECT_TRUE(
            std::equal(after.begin() + row, after.begin() + row + l.q_width(), init.begin() + row))
            << "padding row " << r << " was written";
    }
    return {after.begin() + kGuard, after.begin() + kGuard + n};
}

void check(const Fp8Batch& f, uint32_t seed) {
    expect_close(run_fp8(f, seed), reference(f.b));
}

void check_against_double(const Fp8Batch& f, uint32_t seed) {
    const Batch& b = f.b;
    const Layout& l = b.l;
    const int group = l.heads / l.kv_heads;
    const double scale = 1.0 / std::sqrt(double(l.head_dim));
    const auto actual = run_fp8(f, seed);
    size_t outside = 0;
    double worst = 0.0;
    for (int64_t r = 0; r < b.rows(); ++r) {
        const int64_t len = b.lens[size_t(r)];
        for (int h = 0; h < l.heads && len > 0; ++h) {
            std::vector<double> p(static_cast<size_t>(len));
            for (int64_t j = 0; j < len; ++j) {
                double dot = 0.0;
                for (int d = 0; d < l.head_dim; ++d)
                    dot += double(from_bf16(b.qkv[size_t(r * l.width() + h * l.head_dim + d)])) *
                           (double(from_fp8(f.k[size_t(b.cell(r, j, h / group, d))])) * f.k_scale);
                p[size_t(j)] = dot * scale;
            }
            const double m = *std::max_element(p.begin(), p.end());
            double sum = 0.0;
            for (double& s : p) sum += s = std::exp(s - m);
            for (int d = 0; d < l.head_dim; ++d) {
                double expected = 0.0, weighted_abs = 0.0;
                for (int64_t j = 0; j < len; ++j) {
                    const double value =
                        double(from_fp8(f.v[size_t(b.cell(r, j, h / group, d))])) * f.v_scale;
                    expected += p[size_t(j)] / sum * value;
                    weighted_abs += p[size_t(j)] / sum * std::abs(value);
                }
                const double slack = kTolerance * (weighted_abs + std::abs(expected));
                const double err = std::abs(
                    double(from_bf16(actual[size_t(r * l.q_width() + h * l.head_dim + d)])) -
                    expected);
                if (!(err <= slack)) ++outside;
                if (slack > 0.0) worst = std::max(worst, err / slack);
            }
        }
    }
    EXPECT_EQ(outside, 0u) << "worst " << worst << " times the allowed error";
}

class PagedLayoutFp8Test : public ::testing::TestWithParam<Layout> {};

}  // namespace

TEST(WriteKvFp8, SequenceThroughAShuffledBlockTable) {
    const Case c{32, 4, 64, 9, 16};
    check_write(c, sequence_slots({5, 2, 8}, 37, 16), 1.0f, 1.0f, 1);
}

TEST(WriteKvFp8, ScalesDivideBeforeRounding) {
    const Case c{32, 4, 64, 9, 16};
    const auto slots = sequence_slots({5, 2, 8}, 37, 16);
    check_write(c, slots, 0.5f, 0.25f, 2);
    check_write(c, slots, 4.0f, 0.015625f, 3);
    check_write(c, slots, 3.0f, 0.0437f, 4);
    check_write(c, slots, 0.01f, 7.3f, 5);
}

TEST(WriteKvFp8, ValuesAboveTheRangeSaturate) {
    const Case c{4, 2, 16, 3, 4};
    check_write(c, {0, 5, 11}, 1.0f, 1.0f, 6, 2000.0f);
    check_write(c, {0, 5, 11}, 0.001f, 0.002f, 7, 4.0f);

    std::vector<uint16_t> qkv(size_t(c.width()), to_bf16(1.0f));
    const int64_t kv = c.width() - 2 * c.kv_width();
    qkv[size_t(kv)] = to_bf16(448.0f);
    qkv[size_t(kv + 1)] = to_bf16(-449.0f);
    qkv[size_t(kv + 2)] = to_bf16(1.0e30f);
    qkv[size_t(kv + 3)] = to_bf16(-1.0e30f);
    DeviceBuffer q = upload(qkv);
    DeviceBuffer cache = upload(std::vector<uint8_t>(size_t(2 * c.cache_elems()), 0));
    DeviceBuffer slot = upload(std::vector<int32_t>{0});
    auto* base = static_cast<uint8_t*>(cache.data());
    CudaStream stream;
    write_kv_fp8(Tensor(base, DType::F8E4M3, {3, 2, 4, 16}),
                 Tensor(base + c.cache_elems(), DType::F8E4M3, {3, 2, 4, 16}),
                 Tensor(q.data(), DType::BF16, {1, c.width()}),
                 Tensor(slot.data(), DType::I32, {1}), 1.0f, 1.0f, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto bytes = download<uint8_t>(cache);
    EXPECT_EQ(bytes[0], 0x7E);
    EXPECT_EQ(bytes[1], 0xFE);
    EXPECT_EQ(bytes[2], 0x7E);
    EXPECT_EQ(bytes[3], 0xFE);
    EXPECT_EQ(bytes[4], 0x38) << "1.0 is 0x38 in E4M3";
}

TEST(WriteKvFp8, SpecialValuesAndTies) {
    const Case c{2, 1, 16, 2, 2};
    const float inf = std::numeric_limits<float>::infinity();
    const std::vector<float> values{
        0.0f,    -0.0f,    inf,      -inf,     std::nanf(""), 1.0625f, 1.1875f, -1.0625f,
        0x1p-9f, 0x1p-10f, 0x3p-11f, 0x1p-11f, 0x1p-20f,      447.0f,  463.0f,  464.0f,
    };
    std::vector<uint16_t> qkv(size_t(3 * c.width()), to_bf16(0.5f));
    for (int t = 0; t < 3; ++t)
        for (size_t i = 0; i < values.size(); ++i) {
            const auto row = size_t(t * c.width() + c.width() - 2 * c.kv_width());
            qkv[row + i] = to_bf16(values[i]);
            qkv[row + size_t(c.kv_width()) + i] = to_bf16(-values[values.size() - 1 - i]);
        }
    check_write(c, {3, 0, 2}, qkv, 1.0f, 1.0f, 8);
    check_write(c, {3, 0, 2}, qkv, 2.0f, 0.5f, 9);
    check_write(c, {3, 0, 2}, qkv, 3.0f, 1.7f, 10);
}

TEST(WriteKvFp8, EveryBf16ValueRoundsAsTheHostDoes) {
    const Case c{1, 1, 64, 64, 16};
    std::vector<uint16_t> qkv(size_t(1024 * c.width()), 0);
    std::vector<int32_t> slots(1024);
    std::iota(slots.begin(), slots.end(), 0);
    for (uint32_t u = 0; u < 65536; ++u) {
        const auto row = size_t(u / 64) * size_t(c.width()) + 64;
        qkv[row + u % 64] = uint16_t(u);
        qkv[row + 64 + u % 64] = uint16_t(65535 - u);
    }
    check_write(c, slots, qkv, 1.0f, 1.0f, 11);
    check_write(c, slots, qkv, 3.0f, 0.0437f, 12);
    check_write(c, slots, qkv, 0x1p-100f, 0x1p100f, 13);
}

TEST(WriteKvFp8, PaddingRowsWriteNothing) {
    const Case c{32, 4, 64, 9, 16};
    auto slots = sequence_slots({3, 7, 1}, 40, 16);
    for (size_t i = 0; i < slots.size(); i += 3) slots[i] = -1;
    slots.insert(slots.end(), {-1, -1, -1});
    check_write(c, slots, 1.0f, 2.0f, 14);
    check_write(c, std::vector<int32_t>(5, -1), 1.0f, 1.0f, 15);
}

TEST(WriteKvFp8, BoundarySlots) {
    const Case c{32, 4, 64, 6, 16};
    check_write(c, {0, 15, 16, 31, 95, 80, 48, 47}, 1.0f, 1.0f, 16);
}

TEST(WriteKvFp8, OddBlockSizesAndHeadLayouts) {
    check_write({6, 3, 6, 7, 1}, {6, 0, 3, 5}, 1.0f, 0.5f, 17);
    std::mt19937 rng(18);
    std::vector<int32_t> slots(35);
    std::iota(slots.begin(), slots.end(), 0);
    std::shuffle(slots.begin(), slots.end(), rng);
    slots.resize(20);
    check_write({6, 3, 6, 7, 5}, slots, 0.3f, 1.0f, 19);
    check_write({1, 1, 1, 3, 3}, {8, 4, 0}, 1.0f, 1.0f, 20);
}

TEST(WriteKvFp8, FullLengthPrefill) {
    std::mt19937 rng(21);
    check_write({32, 4, 64, 200, 16}, sequence_slots(random_blocks(128, 200, rng), 2048, 16), 1.0f,
                1.0f, 22);
}

TEST(WriteKvFp8, DecodeBatchOfSixteenSequences) {
    std::mt19937 rng(23);
    const Case c{32, 4, 64, 100, 16};
    const auto blocks = random_blocks(96, 100, rng);
    std::vector<int32_t> slots;
    for (int seq = 0; seq < 16; ++seq) {
        const std::vector<int> table(blocks.begin() + seq * 6, blocks.begin() + seq * 6 + 6);
        const int length = 1 + int(rng() % 96);
        slots.push_back(sequence_slots(table, length, 16).back());
    }
    check_write(c, slots, 0.5f, 0.5f, 24);
}

TEST(WriteKvFp8, NoTokensIsANoOp) {
    check_write({32, 4, 64, 4, 16}, {}, 1.0f, 1.0f, 25);
}

TEST(WriteKvFp8, RejectsBadArguments) {
    const Case c{4, 2, 8, 3, 4};
    DeviceBuffer memory(size_t(2 * c.cache_elems()) * 2);
    DeviceBuffer qkv_buf(size_t(5 * c.width()) * 2);
    DeviceBuffer slot_buf(5 * sizeof(int32_t));
    auto* base = static_cast<uint8_t*>(memory.data());
    const Tensor k(base, DType::F8E4M3, {3, 2, 4, 8});
    const Tensor v(base + c.cache_elems(), DType::F8E4M3, {3, 2, 4, 8});
    const Tensor qkv(qkv_buf.data(), DType::BF16, {5, c.width()});
    const Tensor slots(slot_buf.data(), DType::I32, {5});
    const engine::Stream s = nullptr;
    const auto call = [&](const Tensor& kc, const Tensor& vc, const Tensor& x, const Tensor& m,
                          float ks = 1.0f,
                          float vs = 1.0f) { write_kv_fp8(kc, vc, x, m, ks, vs, s); };

    EXPECT_THROW(call(Tensor(base, DType::BF16, {3, 2, 4, 8}), v, qkv, slots),
                 std::invalid_argument);
    EXPECT_THROW(call(k, Tensor(base, DType::BF16, {3, 2, 4, 8}), qkv, slots),
                 std::invalid_argument);
    EXPECT_THROW(call(Tensor(base, DType::F8E4M3, {3, 2, 32}), v, qkv, slots),
                 std::invalid_argument);
    EXPECT_THROW(call(k, Tensor(base, DType::F8E4M3, {3, 2, 8, 4}), qkv, slots),
                 std::invalid_argument);
    EXPECT_THROW(call(k, Tensor(base, DType::F8E4M3, {2, 2, 4, 8}), qkv, slots),
                 std::invalid_argument);
    EXPECT_THROW(call(k, v, Tensor(qkv_buf.data(), DType::F32, {5, c.width()}), slots),
                 std::invalid_argument);
    EXPECT_THROW(call(k, v, Tensor(qkv_buf.data(), DType::F8E4M3, {5, c.width()}), slots),
                 std::invalid_argument);
    EXPECT_THROW(call(k, v, Tensor(qkv_buf.data(), DType::BF16, {5 * c.width()}), slots),
                 std::invalid_argument);
    EXPECT_THROW(call(k, v, Tensor(qkv_buf.data(), DType::BF16, {5, 2 * c.kv_width()}), slots),
                 std::invalid_argument);
    EXPECT_THROW(call(k, v, qkv, Tensor(slot_buf.data(), DType::I64, {5})), std::invalid_argument);
    EXPECT_THROW(call(k, v, qkv, Tensor(slot_buf.data(), DType::I32, {5, 1})),
                 std::invalid_argument);
    EXPECT_THROW(call(k, v, qkv, Tensor(slot_buf.data(), DType::I32, {4})), std::invalid_argument);
    for (const float bad : {0.0f, -1.0f, std::numeric_limits<float>::infinity(), std::nanf("")}) {
        EXPECT_THROW(call(k, v, qkv, slots, bad, 1.0f), std::invalid_argument) << bad;
        EXPECT_THROW(call(k, v, qkv, slots, 1.0f, bad), std::invalid_argument) << bad;
    }
}

TEST(PagedAttentionDecodeFp8, OneSequenceThroughAShuffledBlockTable) {
    check(quantize(make_batch(kTinyLlama, {37}, 31)), 32);
}

TEST(PagedAttentionDecodeFp8, ContextLengthsAroundBlockBoundaries) {
    for (const int32_t len : {1, 2, 15, 16, 17, 31, 32, 33}) {
        SCOPED_TRACE("context " + std::to_string(len));
        check(quantize(make_batch(kTinyLlama, {len}, 33 + uint32_t(len))), 34);
    }
    check(quantize(make_batch(kTinyLlama, {1, 2, 15, 16, 17, 31, 32, 33}, 35)), 36);
}

TEST(PagedAttentionDecodeFp8, LongContexts) {
    check(quantize(make_batch(kTinyLlama, {kMaxContext}, 37, kMaxContext / 16)), 38);
    check(quantize(make_batch(kTinyLlama, {kMaxContext - 1, 1, kMaxContext}, 39, kMaxContext / 16)),
          40);
    check(quantize(make_batch(kTinyLlama, {8192, 1, 4097}, 41, 8192 / 16)), 42);
}

TEST(PagedAttentionDecodeFp8, DecodeBatchOfSixteenSequences) {
    std::mt19937 rng(43);
    std::vector<int32_t> lens{1, 15, 16, 17, kMaxContext};
    while (lens.size() < 16) lens.push_back(1 + int32_t(rng() % kMaxContext));
    std::shuffle(lens.begin(), lens.end(), rng);
    check(quantize(make_batch(kTinyLlama, lens, 44, kMaxContext / 16)), 45);
}

TEST(PagedAttentionDecodeFp8, PaddingRowsAreNotWritten) {
    check(quantize(make_batch(kTinyLlama, {0, 20, 0, 0, 1, 16, 0}, 46)), 47);
    check(quantize(make_batch(kTinyLlama, {0, 0, 0}, 48)), 49);
}

TEST(PagedAttentionDecodeFp8, PowerOfTwoScales) {
    const std::vector<int32_t> lens{1, 17, 100, 0, 33};
    check(quantize(make_batch(kTinyLlama, lens, 50), 0.5f, 2.0f), 51);
    check(quantize(make_batch(kTinyLlama, lens, 52), 0.125f, 0.03125f), 53);
    check(quantize(make_batch(kTinyLlama, lens, 54, 0, 0.25f), 0x1p-9f, 0x1p-10f), 55);
    check(quantize(make_batch(kTinyLlama, lens, 56), 4.0f, 8.0f), 57);
}

TEST(PagedAttentionDecodeFp8, ScalesThatAreNotPowersOfTwo) {
    const std::vector<int32_t> lens{1, 17, 100, 0, 33};
    check_against_double(quantize(make_batch(kTinyLlama, lens, 58, 0, 1.0f), 1.0f, 1.0f), 59);
    check_against_double(quantize(make_batch(kTinyLlama, lens, 60, 0, 1.0f), 0.3f, 0.7f), 61);
    check_against_double(quantize(make_batch(kTinyLlama, lens, 62, 0, 1.0f), 0.0123f, 3.3f), 63);
}

TEST(PagedAttentionDecodeFp8, TheScaleIsApplied) {
    Fp8Batch f = quantize(make_batch(kTinyLlama, {40, 3}, 64, 0, 1.0f));
    const auto base = run_fp8(f, 65);
    f.v_scale = 2.0f;
    const auto doubled = run_fp8(f, 65);
    for (size_t i = 0; i < base.size(); ++i)
        ASSERT_EQ(from_bf16(doubled[i]), 2.0f * from_bf16(base[i])) << "element " << i;
    f.v_scale = 1.0f;
    f.k_scale = 4.0f;
    for (auto& x : f.b.k) x = to_bf16(4.0f * from_bf16(x));
    expect_close(run_fp8(f, 65), reference(f.b));
}

TEST(PagedAttentionDecodeFp8, ASingleTokenAttendsOnlyToItsOwnValue) {
    for (const float v_scale : {1.0f, 0.5f}) {
        const Fp8Batch f = quantize(make_batch(kTinyLlama, {1, 1, 1}, 66), 1.0f, v_scale);
        const auto out = run_fp8(f, 67);
        const int group = f.b.l.heads / f.b.l.kv_heads;
        for (int64_t r = 0; r < f.b.rows(); ++r)
            for (int h = 0; h < f.b.l.heads; ++h)
                for (int d = 0; d < f.b.l.head_dim; ++d)
                    ASSERT_EQ(out[size_t(r * f.b.l.q_width() + h * f.b.l.head_dim + d)],
                              f.b.v[size_t(f.b.cell(r, 0, h / group, d))])
                        << "row " << r << ", head " << h << ", dim " << d;
    }
}

TEST(PagedAttentionDecodeFp8, SaturatedAndZeroValues) {
    Fp8Batch f = quantize(make_batch(kTinyLlama, {1, 17, 300}, 68));
    std::mt19937 rng(69);
    const uint8_t pick[] = {0x7E, 0xFE, 0x00, 0x80, 0x01, 0x81};
    for (size_t i = 0; i < f.v.size(); ++i) {
        f.v[i] = pick[rng() % 6];
        f.b.v[i] = to_bf16(from_fp8(f.v[i]));
    }
    check(f, 70);
}

TEST(PagedAttentionDecodeFp8, OutputDependsOnlyOnTheContext) {
    const Fp8Batch f = quantize(make_batch(kTinyLlama, {1, 15, 16, 17, 33, 0, 100}, 71));
    const Batch& b = f.b;
    const auto clean = run_fp8(f, 72);
    expect_close(clean, reference(b));
    const auto used = b.reached();

    Fp8Batch nan = f;
    for (size_t i = 0; i < used.size(); ++i)
        if (!used[i]) nan.k[i] = nan.v[i] = 0x7F;
    for (int64_t r = 0; r < b.rows(); ++r)
        std::fill(nan.b.qkv.begin() + r * b.l.width() + b.l.q_width(),
                  nan.b.qkv.begin() + (r + 1) * b.l.width(), kNan);
    EXPECT_EQ(run_fp8(nan, 72), clean) << "NaN outside the context reached the output";

    Fp8Batch huge = f;
    for (size_t i = 0; i < used.size(); ++i)
        if (!used[i]) {
            huge.k[i] = i % 2 ? 0x7E : 0xFE;
            huge.v[i] = 0x7E;
        }
    for (int64_t r = 0; r < b.rows(); ++r)
        for (int j = blocks_for(b.lens[size_t(r)], b.l.block_size); j < b.table_width; ++j)
            huge.b.tables[size_t(r * b.table_width + j)] = -1;
    EXPECT_EQ(run_fp8(huge, 72), clean) << "saturated bytes outside the context reached the output";
}

TEST(PagedAttentionDecodeFp8, ARowDoesNotDependOnTheRestOfTheBatch) {
    std::mt19937 rng(73);
    std::vector<int32_t> lens{0, 1, 16, 17, kMaxContext};
    while (lens.size() < 16) lens.push_back(1 + int32_t(rng() % kMaxContext));
    const Fp8Batch f = quantize(make_batch(kTinyLlama, lens, 74, kMaxContext / 16), 0.5f, 0.5f);
    const auto together = run_fp8(f, 75);
    EXPECT_EQ(run_fp8(f, 75), together) << "two identical calls differ";
    for (int64_t r = 0; r < f.b.rows(); ++r) {
        if (f.b.lens[size_t(r)] == 0) continue;
        Fp8Batch one = f;
        one.b = single_row(f.b, r);
        const auto alone = run_fp8(one, 76);
        EXPECT_TRUE(std::equal(alone.begin(), alone.end(), together.begin() + r * f.b.l.q_width()))
            << "row " << r << " alone differs from the same row in the batch";
    }
}

TEST(PagedAttentionDecodeFp8, RowsSharingABlockTableServeAPrefill) {
    check(quantize(make_prefill(kTinyLlama, 40, 77)), 78);
}

TEST_P(PagedLayoutFp8Test, MatchesNaiveAttention) {
    const Layout l = GetParam();
    std::vector<int32_t> lens{1, l.block_size, l.block_size + 1, 3 * l.block_size - 1, 0, 70};
    check(quantize(make_batch(l, lens, 79), 1.0f, 0.25f), 80);
    check(quantize(make_prefill(l, 23, 81)), 82);
}

INSTANTIATE_TEST_SUITE_P(LayoutsFp8, PagedLayoutFp8Test,
                         ::testing::Values(Layout{4, 2, 8, 5}, Layout{3, 3, 16, 1},
                                           Layout{6, 1, 4, 3}, Layout{6, 3, 6, 7},
                                           Layout{8, 2, 128, 16}, Layout{32, 4, 64, 32}),
                         [](const auto& info) {
                             const Layout& l = info.param;
                             return "heads" + std::to_string(l.heads) + "_kv" +
                                    std::to_string(l.kv_heads) + "_dim" +
                                    std::to_string(l.head_dim) + "_block" +
                                    std::to_string(l.block_size);
                         });

TEST(PagedAttentionDecodeFp8, NoRowsIsANoOp) {
    DeviceBuffer buf(2 * 4 * 16 * 64);
    CudaStream stream;
    const Tensor cache(buf.data(), DType::F8E4M3, {2, 4, 16, 64});
    paged_attention_decode_fp8(Tensor(nullptr, DType::BF16, {0, 2048}),
                               Tensor(nullptr, DType::BF16, {0, 2560}), cache, cache,
                               Tensor(nullptr, DType::I32, {0, 128}),
                               Tensor(nullptr, DType::I32, {0}), 1.0f, 1.0f, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TEST(PagedAttentionDecodeFp8, RejectsBadArguments) {
    DeviceBuffer buf(1 << 16);
    void* p = buf.data();
    const Tensor out(p, DType::BF16, {3, 32});
    const Tensor qkv(p, DType::BF16, {3, 64});
    const Tensor cache(p, DType::F8E4M3, {5, 2, 4, 8});
    const Tensor tables(p, DType::I32, {3, 4});
    const Tensor lens(p, DType::I32, {3});
    const engine::Stream s = nullptr;
    const auto call = [&](const Tensor& o, const Tensor& x, const Tensor& k, const Tensor& v,
                          const Tensor& t, const Tensor& c, float ks = 1.0f, float vs = 1.0f) {
        paged_attention_decode_fp8(o, x, k, v, t, c, ks, vs, s);
    };

    EXPECT_THROW(call(Tensor(p, DType::F32, {3, 32}), qkv, cache, cache, tables, lens),
                 std::invalid_argument);
    EXPECT_THROW(call(Tensor(p, DType::BF16, {96}), qkv, cache, cache, tables, lens),
                 std::invalid_argument);
    EXPECT_THROW(call(Tensor(p, DType::BF16, {2, 32}), qkv, cache, cache, tables, lens),
                 std::invalid_argument);
    EXPECT_THROW(call(Tensor(p, DType::BF16, {3, 24}), Tensor(p, DType::BF16, {3, 56}), cache,
                      cache, tables, lens),
                 std::invalid_argument);
    EXPECT_THROW(call(out, Tensor(p, DType::F32, {3, 64}), cache, cache, tables, lens),
                 std::invalid_argument);
    EXPECT_THROW(call(out, Tensor(p, DType::BF16, {192}), cache, cache, tables, lens),
                 std::invalid_argument);
    EXPECT_THROW(call(out, Tensor(p, DType::BF16, {3, 48}), cache, cache, tables, lens),
                 std::invalid_argument);
    EXPECT_THROW(call(out, Tensor(p, DType::BF16, {4, 64}), cache, cache, tables, lens),
                 std::invalid_argument);
    EXPECT_THROW(call(out, qkv, Tensor(p, DType::BF16, {5, 2, 4, 8}), cache, tables, lens),
                 std::invalid_argument);
    EXPECT_THROW(call(out, qkv, cache, Tensor(p, DType::BF16, {5, 2, 4, 8}), tables, lens),
                 std::invalid_argument);
    EXPECT_THROW(call(out, qkv, Tensor(p, DType::F8E4M3, {5, 8, 8}), cache, tables, lens),
                 std::invalid_argument);
    EXPECT_THROW(call(out, qkv, cache, Tensor(p, DType::F8E4M3, {5, 2, 8, 4}), tables, lens),
                 std::invalid_argument);
    EXPECT_THROW(call(out, qkv, cache, Tensor(p, DType::F8E4M3, {4, 2, 4, 8}), tables, lens),
                 std::invalid_argument);
    EXPECT_THROW(call(out, qkv, cache, cache, Tensor(p, DType::I64, {3, 4}), lens),
                 std::invalid_argument);
    EXPECT_THROW(call(out, qkv, cache, cache, Tensor(p, DType::I32, {12}), lens),
                 std::invalid_argument);
    EXPECT_THROW(call(out, qkv, cache, cache, Tensor(p, DType::I32, {2, 4}), lens),
                 std::invalid_argument);
    EXPECT_THROW(call(out, qkv, cache, cache, Tensor(p, DType::I32, {3, 0}), lens),
                 std::invalid_argument);
    EXPECT_THROW(call(out, qkv, cache, cache, tables, Tensor(p, DType::I64, {3})),
                 std::invalid_argument);
    EXPECT_THROW(call(out, qkv, cache, cache, tables, Tensor(p, DType::I32, {3, 1})),
                 std::invalid_argument);
    EXPECT_THROW(call(out, qkv, cache, cache, tables, Tensor(p, DType::I32, {4})),
                 std::invalid_argument);
    for (const float bad : {0.0f, -1.0f, std::numeric_limits<float>::infinity(), std::nanf("")}) {
        EXPECT_THROW(call(out, qkv, cache, cache, tables, lens, bad, 1.0f), std::invalid_argument)
            << bad;
        EXPECT_THROW(call(out, qkv, cache, cache, tables, lens, 1.0f, bad), std::invalid_argument)
            << bad;
    }
}
