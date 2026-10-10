#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine/kernels/attention.h"
#include "engine/kernels/paged_attention.h"
#include "engine/kernels/rope.h"
#include "engine/model/config.h"
#include "kernels/cuda_check.h"
#include "support/bf16.h"
#include "support/device.h"
#include "support/paths.h"
#include "support/reference.h"
#include "support/safetensors_file.h"

using engine::DeviceBuffer;
using engine::DType;
using engine::paged_attention_decode;
using engine::Tensor;
using engine::test::compare_bf16;
using engine::test::CudaStream;
using engine::test::download;
using engine::test::from_bf16;
using engine::test::random_bf16;
using engine::test::to_bf16;
using engine::test::upload;

namespace {

struct Layout {
    int heads, kv_heads, head_dim, block_size;
    int64_t q_width() const { return int64_t(heads) * head_dim; }
    int64_t kv_width() const { return int64_t(kv_heads) * head_dim; }
    int64_t width() const { return q_width() + 2 * kv_width(); }
    int64_t block_elems() const { return kv_width() * block_size; }
};

const Layout kTinyLlama{32, 4, 64, 16};
constexpr int kMaxContext = 2048;
constexpr int64_t kGuard = 256;
constexpr double kTolerance = 1.0 / 64;
constexpr float kBf16Epsilon = 1.0f / 128.0f;
constexpr uint16_t kNan = 0xFFFF;

int blocks_for(int len, int block_size) {
    return (len + block_size - 1) / block_size;
}

struct Batch {
    Layout l{};
    int num_blocks = 0;
    int table_width = 0;
    std::vector<int32_t> lens;
    std::vector<int32_t> tables;
    std::vector<uint16_t> qkv, k, v;

    int64_t rows() const { return int64_t(lens.size()); }

    int64_t cell(int64_t row, int64_t pos, int h, int d) const {
        const int64_t block = tables[size_t(row * table_width + pos / l.block_size)];
        return ((block * l.kv_heads + h) * l.block_size + pos % l.block_size) * l.head_dim + d;
    }

    std::vector<bool> reached() const {
        std::vector<bool> used(k.size(), false);
        for (int64_t r = 0; r < rows(); ++r)
            for (int64_t pos = 0; pos < lens[size_t(r)]; ++pos)
                for (int h = 0; h < l.kv_heads; ++h)
                    for (int d = 0; d < l.head_dim; ++d) used[size_t(cell(r, pos, h, d))] = true;
        return used;
    }
};

int default_width(const std::vector<int32_t>& lens, int block_size) {
    int widest = 1;
    for (const int32_t len : lens) widest = std::max(widest, blocks_for(len, block_size));
    return std::max(widest, std::min(widest + 2, kMaxContext / block_size));
}

void fill_cache(Batch& b, uint32_t seed, float range) {
    const auto cells = size_t(b.num_blocks * b.l.block_elems());
    b.qkv = random_bf16(size_t(b.rows() * b.l.width()), seed + 1, -range, range);
    b.k = random_bf16(cells, seed + 2, -range, range);
    b.v = random_bf16(cells, seed + 3, -range, range);
}

Batch make_batch(const Layout& l, std::vector<int32_t> lens, uint32_t seed, int table_width = 0,
                 float range = 2.0f) {
    std::mt19937 rng(seed);
    Batch b;
    b.l = l;
    b.lens = std::move(lens);
    b.table_width = table_width > 0 ? table_width : default_width(b.lens, l.block_size);
    int needed = 0;
    for (const int32_t len : b.lens) needed += blocks_for(len, l.block_size);
    b.num_blocks = needed + 3;
    std::vector<int32_t> ids(static_cast<size_t>(b.num_blocks));
    std::iota(ids.begin(), ids.end(), 0);
    std::shuffle(ids.begin(), ids.end(), rng);
    b.tables.resize(size_t(b.rows() * b.table_width));
    size_t next = 0;
    for (int64_t r = 0; r < b.rows(); ++r)
        for (int j = 0; j < b.table_width; ++j)
            b.tables[size_t(r * b.table_width + j)] =
                j < blocks_for(b.lens[size_t(r)], l.block_size)
                    ? ids[next++]
                    : int32_t(rng() % uint32_t(b.num_blocks));
    fill_cache(b, seed, range);
    return b;
}

Batch make_prefill(const Layout& l, int len, uint32_t seed) {
    std::mt19937 rng(seed);
    Batch b;
    b.l = l;
    b.table_width = default_width({len}, l.block_size);
    b.num_blocks = blocks_for(len, l.block_size) + 3;
    std::vector<int32_t> table(static_cast<size_t>(b.num_blocks));
    std::iota(table.begin(), table.end(), 0);
    std::shuffle(table.begin(), table.end(), rng);
    table.resize(size_t(b.table_width), 0);
    for (int t = 0; t < len; ++t) {
        b.lens.push_back(t + 1);
        b.tables.insert(b.tables.end(), table.begin(), table.end());
    }
    fill_cache(b, seed, 2.0f);
    return b;
}

Batch single_row(const Batch& b, int64_t row) {
    Batch one = b;
    one.lens = {b.lens[size_t(row)]};
    one.tables.assign(b.tables.begin() + row * b.table_width,
                      b.tables.begin() + (row + 1) * b.table_width);
    one.qkv.assign(b.qkv.begin() + row * b.l.width(), b.qkv.begin() + (row + 1) * b.l.width());
    return one;
}

std::vector<uint16_t> run_paged(const Batch& b, uint32_t seed) {
    const Layout& l = b.l;
    const int64_t n = b.rows() * l.q_width();
    const auto init = random_bf16(size_t(n + 2 * kGuard), seed);
    DeviceBuffer out_buf = upload(init);
    DeviceBuffer qkv = upload(b.qkv), k = upload(b.k), v = upload(b.v);
    DeviceBuffer tables = upload(b.tables), lens = upload(b.lens);
    const std::initializer_list<int64_t> cache_shape{b.num_blocks, l.kv_heads, l.block_size,
                                                     l.head_dim};

    CudaStream stream;
    paged_attention_decode(Tensor(static_cast<uint16_t*>(out_buf.data()) + kGuard, DType::BF16,
                                  {b.rows(), l.q_width()}),
                           Tensor(qkv.data(), DType::BF16, {b.rows(), l.width()}),
                           Tensor(k.data(), DType::BF16, cache_shape),
                           Tensor(v.data(), DType::BF16, cache_shape),
                           Tensor(tables.data(), DType::I32, {b.rows(), b.table_width}),
                           Tensor(lens.data(), DType::I32, {b.rows()}), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    EXPECT_EQ(download<uint16_t>(qkv), b.qkv) << "qkv was modified";
    EXPECT_EQ(download<uint16_t>(k), b.k) << "k_cache was modified";
    EXPECT_EQ(download<uint16_t>(v), b.v) << "v_cache was modified";
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

struct Expected {
    std::vector<uint16_t> out;
    std::vector<double> slack;
};

Expected reference(const Batch& b) {
    const Layout& l = b.l;
    const int group = l.heads / l.kv_heads;
    const double scale = 1.0 / std::sqrt(double(l.head_dim));
    Expected e{std::vector<uint16_t>(size_t(b.rows() * l.q_width())),
               std::vector<double>(size_t(b.rows() * l.q_width()), -1.0)};
    CudaStream stream;
    for (int64_t r = 0; r < b.rows(); ++r) {
        const int64_t len = b.lens[size_t(r)];
        if (len == 0) continue;
        std::vector<uint16_t> kc(size_t(len * l.kv_width())), vc(kc.size());
        for (int64_t j = 0; j < len; ++j)
            for (int h = 0; h < l.kv_heads; ++h)
                for (int d = 0; d < l.head_dim; ++d) {
                    const auto i = size_t(j * l.kv_width() + h * l.head_dim + d);
                    kc[i] = b.k[size_t(b.cell(r, j, h, d))];
                    vc[i] = b.v[size_t(b.cell(r, j, h, d))];
                }
        const std::vector<uint16_t> q(b.qkv.begin() + r * l.width(),
                                      b.qkv.begin() + (r + 1) * l.width());
        DeviceBuffer dq = upload(q), dk = upload(kc), dv = upload(vc);
        DeviceBuffer dout(size_t(l.q_width()) * sizeof(uint16_t));
        engine::naive::attention(Tensor(dout.data(), DType::BF16, {1, l.q_width()}),
                                 Tensor(dq.data(), DType::BF16, {1, l.width()}),
                                 Tensor(dk.data(), DType::BF16, {len, l.kv_width()}),
                                 Tensor(dv.data(), DType::BF16, {len, l.kv_width()}), len - 1,
                                 l.heads, l.kv_heads, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const auto naive = download<uint16_t>(dout);

        for (int h = 0; h < l.heads; ++h) {
            const int64_t kv = int64_t(h / group) * l.head_dim;
            std::vector<double> p(static_cast<size_t>(len));
            for (int64_t j = 0; j < len; ++j) {
                double dot = 0.0;
                for (int d = 0; d < l.head_dim; ++d)
                    dot += double(from_bf16(q[size_t(h * l.head_dim + d)])) *
                           from_bf16(kc[size_t(j * l.kv_width() + kv + d)]);
                p[size_t(j)] = dot * scale;
            }
            const double m = *std::max_element(p.begin(), p.end());
            double sum = 0.0;
            for (double& s : p) sum += s = std::exp(s - m);
            for (int d = 0; d < l.head_dim; ++d) {
                double weighted_abs = 0.0;
                for (int64_t j = 0; j < len; ++j)
                    weighted_abs += p[size_t(j)] / sum *
                                    std::abs(from_bf16(vc[size_t(j * l.kv_width() + kv + d)]));
                const auto local = size_t(h * l.head_dim + d);
                const auto i = size_t(r * l.q_width()) + local;
                e.out[i] = naive[local];
                e.slack[i] = kTolerance * (weighted_abs + std::abs(from_bf16(naive[local])));
            }
        }
    }
    return e;
}

void expect_close(const std::vector<uint16_t>& actual, const Expected& e) {
    size_t outside = 0, first = 0;
    double worst = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (e.slack[i] < 0.0) continue;
        const double err = std::abs(double(from_bf16(actual[i])) - from_bf16(e.out[i]));
        if (!(err <= e.slack[i]) && outside++ == 0) first = i;
        if (e.slack[i] > 0.0) worst = std::max(worst, err / e.slack[i]);
    }
    EXPECT_EQ(outside, 0u) << "first at element " << first << ", worst " << worst
                           << " times the allowed error, " << compare_bf16(actual, e.out);
}

void check(const Batch& b, uint32_t seed) {
    expect_close(run_paged(b, seed), reference(b));
}

size_t outside_value_scale(const std::vector<uint16_t>& actual,
                           const std::vector<uint16_t>& expected, const std::vector<uint16_t>& v,
                           int heads, int kv_heads, int head_dim) {
    const int64_t q_width = int64_t(heads) * head_dim, kv_width = int64_t(kv_heads) * head_dim;
    const int64_t tokens = int64_t(actual.size()) / q_width;
    const int group = heads / kv_heads;
    std::vector<float> v_max(size_t(kv_width), 0.0f);
    size_t n = 0;
    for (int64_t t = 0; t < tokens; ++t) {
        for (int64_t c = 0; c < kv_width; ++c)
            v_max[size_t(c)] =
                std::max(v_max[size_t(c)], std::abs(from_bf16(v[size_t(t * kv_width + c)])));
        for (int64_t c = 0; c < q_width; ++c) {
            const auto i = size_t(t * q_width + c);
            const int64_t kv_c = (c / head_dim) / group * head_dim + c % head_dim;
            if (std::abs(from_bf16(actual[i]) - from_bf16(expected[i])) >
                kBf16Epsilon * v_max[size_t(kv_c)])
                ++n;
        }
    }
    return n;
}

class PagedLayoutTest : public ::testing::TestWithParam<Layout> {};

}  // namespace

TEST(PagedAttentionDecode, OneSequenceThroughAShuffledBlockTable) {
    check(make_batch(kTinyLlama, {37}, 1), 2);
}

TEST(PagedAttentionDecode, ContextLengthsAroundBlockBoundaries) {
    for (const int32_t len : {1, 2, 15, 16, 17, 31, 32, 33}) {
        SCOPED_TRACE("context " + std::to_string(len));
        check(make_batch(kTinyLlama, {len}, 3 + uint32_t(len)), 4);
    }
    check(make_batch(kTinyLlama, {1, 2, 15, 16, 17, 31, 32, 33}, 5), 6);
}

TEST(PagedAttentionDecode, FullContextLength) {
    check(make_batch(kTinyLlama, {kMaxContext}, 7, kMaxContext / 16), 8);
    check(make_batch(kTinyLlama, {kMaxContext - 1, 1, kMaxContext}, 9, kMaxContext / 16), 10);
}

TEST(PagedAttentionDecode, DecodeBatchOfSixteenSequences) {
    std::mt19937 rng(11);
    std::vector<int32_t> lens{1, 15, 16, 17, kMaxContext};
    while (lens.size() < 16) lens.push_back(1 + int32_t(rng() % kMaxContext));
    std::shuffle(lens.begin(), lens.end(), rng);
    check(make_batch(kTinyLlama, lens, 12, kMaxContext / 16), 13);
}

TEST(PagedAttentionDecode, PaddingRowsAreNotWritten) {
    check(make_batch(kTinyLlama, {0, 20, 0, 0, 1, 16, 0}, 14), 15);
    check(make_batch(kTinyLlama, {0, 0, 0}, 16), 17);
}

TEST(PagedAttentionDecode, LargeScoresDoNotOverflow) {
    Batch b = make_batch(kTinyLlama, {1, 17, 300}, 18);
    std::mt19937 rng(19);
    const auto pick = [&] { return rng() % 10 ? to_bf16(4.0f) : uint16_t(0); };
    for (int64_t r = 0; r < b.rows(); ++r)
        for (int64_t c = 0; c < b.l.q_width(); ++c) b.qkv[size_t(r * b.l.width() + c)] = pick();
    for (auto& x : b.k) x = pick();
    check(b, 20);
}

TEST(PagedAttentionDecode, ASingleTokenAttendsOnlyToItsOwnValue) {
    const Batch b = make_batch(kTinyLlama, {1, 1, 1}, 21);
    const auto out = run_paged(b, 22);
    const int group = b.l.heads / b.l.kv_heads;
    for (int64_t r = 0; r < b.rows(); ++r)
        for (int h = 0; h < b.l.heads; ++h)
            for (int d = 0; d < b.l.head_dim; ++d)
                ASSERT_EQ(out[size_t(r * b.l.q_width() + h * b.l.head_dim + d)],
                          b.v[size_t(b.cell(r, 0, h / group, d))])
                    << "row " << r << ", head " << h << ", dim " << d;
}

TEST(PagedAttentionDecode, OutputDependsOnlyOnTheContext) {
    const Batch b = make_batch(kTinyLlama, {1, 15, 16, 17, 33, 0, 100}, 23);
    const auto clean = run_paged(b, 24);
    expect_close(clean, reference(b));
    const auto used = b.reached();

    Batch nan = b;
    for (size_t i = 0; i < used.size(); ++i)
        if (!used[i]) nan.k[i] = nan.v[i] = kNan;
    for (int64_t r = 0; r < b.rows(); ++r)
        std::fill(nan.qkv.begin() + r * b.l.width() + b.l.q_width(),
                  nan.qkv.begin() + (r + 1) * b.l.width(), kNan);
    EXPECT_EQ(run_paged(nan, 24), clean) << "NaN outside the context reached the output";

    Batch huge = b;
    for (size_t i = 0; i < used.size(); ++i)
        if (!used[i]) {
            huge.k[i] = to_bf16(i % 2 ? 30000.0f : -30000.0f);
            huge.v[i] = to_bf16(30000.0f);
        }
    for (int64_t r = 0; r < b.rows(); ++r)
        for (int j = blocks_for(b.lens[size_t(r)], b.l.block_size); j < b.table_width; ++j)
            huge.tables[size_t(r * b.table_width + j)] = -1;
    EXPECT_EQ(run_paged(huge, 24), clean) << "huge values outside the context reached the output";
}

TEST(PagedAttentionDecode, ARowDoesNotDependOnTheRestOfTheBatch) {
    std::mt19937 rng(27);
    std::vector<int32_t> lens{0, 1, 16, 17, kMaxContext};
    while (lens.size() < 16) lens.push_back(1 + int32_t(rng() % kMaxContext));
    const Batch b = make_batch(kTinyLlama, lens, 28, kMaxContext / 16);
    const auto together = run_paged(b, 29);
    EXPECT_EQ(run_paged(b, 29), together) << "two identical calls differ";
    for (int64_t r = 0; r < b.rows(); ++r) {
        if (b.lens[size_t(r)] == 0) continue;
        const auto alone = run_paged(single_row(b, r), 31);
        EXPECT_TRUE(std::equal(alone.begin(), alone.end(), together.begin() + r * b.l.q_width()))
            << "row " << r << " alone differs from the same row in the batch";
    }
}

TEST(PagedAttentionDecode, RowsSharingABlockTableServeAPrefill) {
    check(make_prefill(kTinyLlama, 40, 32), 33);
}

TEST_P(PagedLayoutTest, MatchesNaiveAttention) {
    const Layout l = GetParam();
    std::vector<int32_t> lens{1, l.block_size, l.block_size + 1, 3 * l.block_size - 1, 0, 70};
    check(make_batch(l, lens, 34), 35);
    check(make_prefill(l, 23, 36), 37);
}

INSTANTIATE_TEST_SUITE_P(Layouts, PagedLayoutTest,
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

TEST(PagedAttentionDecode, NoRowsIsANoOp) {
    DeviceBuffer buf(2 * 4 * 16 * 64 * sizeof(uint16_t));
    CudaStream stream;
    const Tensor cache(buf.data(), DType::BF16, {2, 4, 16, 64});
    paged_attention_decode(
        Tensor(nullptr, DType::BF16, {0, 2048}), Tensor(nullptr, DType::BF16, {0, 2560}), cache,
        cache, Tensor(nullptr, DType::I32, {0, 128}), Tensor(nullptr, DType::I32, {0}), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TEST(PagedAttentionDecode, MatchesHuggingFaceAttnOnEveryTracedLayer) {
    if (const auto why = engine::test::reference_skip_reason(); !why.empty()) GTEST_SKIP() << why;
    const auto config = engine::load_config(engine::test::model_dir() / "config.json");
    const engine::test::SafetensorsFile ref(engine::test::traced_reference_prompt().file);
    const Layout l{config.heads, config.kv_heads, config.head_dim, 16};
    const auto tokens = int(ref.entry("input_ids").shape[0]);
    std::vector<int32_t> positions(static_cast<size_t>(tokens));
    std::iota(positions.begin(), positions.end(), 0);
    CudaStream stream;
    DeviceBuffer dpos = upload(positions);
    for (int layer = 0; layer < config.layers; ++layer) {
        SCOPED_TRACE("layer " + std::to_string(layer));
        const std::string name = "layers." + std::to_string(layer) + ".";
        const auto q = ref.read<uint16_t>(name + "q");
        const auto k = ref.read<uint16_t>(name + "k");
        const auto v = ref.read<uint16_t>(name + "v");
        std::vector<uint16_t> qkv;
        for (int64_t t = 0; t < tokens; ++t) {
            qkv.insert(qkv.end(), q.begin() + t * l.q_width(), q.begin() + (t + 1) * l.q_width());
            qkv.insert(qkv.end(), k.begin() + t * l.kv_width(), k.begin() + (t + 1) * l.kv_width());
            qkv.insert(qkv.end(), v.begin() + t * l.kv_width(), v.begin() + (t + 1) * l.kv_width());
        }
        DeviceBuffer dqkv = upload(qkv);
        engine::naive::rope(Tensor(dqkv.data(), DType::BF16, {tokens, l.width()}),
                            Tensor(dpos.data(), DType::I32, {tokens}), l.heads, l.kv_heads,
                            float(config.rope_theta), stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        Batch b = make_prefill(l, tokens, 38 + uint32_t(layer));
        b.qkv = download<uint16_t>(dqkv);
        for (int64_t t = 0; t < tokens; ++t)
            for (int h = 0; h < l.kv_heads; ++h)
                for (int d = 0; d < l.head_dim; ++d) {
                    const auto col = size_t(t * l.width() + l.q_width() + h * l.head_dim + d);
                    b.k[size_t(b.cell(tokens - 1, t, h, d))] = b.qkv[col];
                    b.v[size_t(b.cell(tokens - 1, t, h, d))] = b.qkv[col + size_t(l.kv_width())];
                }
        const auto actual = run_paged(b, 60);
        const auto expected = ref.read<uint16_t>(name + "attn");
        EXPECT_EQ(outside_value_scale(actual, expected, v, l.heads, l.kv_heads, l.head_dim), 0u)
            << compare_bf16(actual, expected);
    }
}

TEST(PagedAttentionDecode, RejectsBadArguments) {
    DeviceBuffer buf(1 << 16);
    void* p = buf.data();
    const Tensor out(p, DType::BF16, {3, 32});
    const Tensor qkv(p, DType::BF16, {3, 64});
    const Tensor cache(p, DType::BF16, {5, 2, 4, 8});
    const Tensor tables(p, DType::I32, {3, 4});
    const Tensor lens(p, DType::I32, {3});
    const engine::Stream s = nullptr;
    const auto call = [&](const Tensor& o, const Tensor& x, const Tensor& k, const Tensor& v,
                          const Tensor& t,
                          const Tensor& c) { paged_attention_decode(o, x, k, v, t, c, s); };

    EXPECT_THROW(call(Tensor(p, DType::F32, {3, 32}), qkv, cache, cache, tables, lens),
                 std::invalid_argument);
    EXPECT_THROW(call(Tensor(p, DType::BF16, {96}), qkv, cache, cache, tables, lens),
                 std::invalid_argument);
    EXPECT_THROW(call(Tensor(p, DType::BF16, {2, 32}), qkv, cache, cache, tables, lens),
                 std::invalid_argument);
    EXPECT_THROW(call(Tensor(p, DType::BF16, {3, 24}), Tensor(p, DType::BF16, {3, 56}), cache,
                      cache, tables, lens),
                 std::invalid_argument);
    EXPECT_THROW(call(Tensor(p, DType::BF16, {3, 28}), Tensor(p, DType::BF16, {3, 60}), cache,
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
    EXPECT_THROW(call(out, qkv, Tensor(p, DType::F16, {5, 2, 4, 8}), cache, tables, lens),
                 std::invalid_argument);
    EXPECT_THROW(call(out, qkv, cache, Tensor(p, DType::F32, {5, 2, 4, 8}), tables, lens),
                 std::invalid_argument);
    EXPECT_THROW(call(out, qkv, Tensor(p, DType::BF16, {5, 8, 8}), cache, tables, lens),
                 std::invalid_argument);
    EXPECT_THROW(call(out, qkv, cache, Tensor(p, DType::BF16, {5, 2, 8, 4}), tables, lens),
                 std::invalid_argument);
    EXPECT_THROW(call(out, qkv, cache, Tensor(p, DType::BF16, {4, 2, 4, 8}), tables, lens),
                 std::invalid_argument);
    EXPECT_THROW(call(out, qkv, Tensor(p, DType::BF16, {5, 4, 4, 8}),
                      Tensor(p, DType::BF16, {5, 4, 4, 8}), tables, lens),
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
}
