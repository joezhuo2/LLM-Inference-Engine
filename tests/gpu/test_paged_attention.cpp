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
#include "support/paged_attention_harness.h"
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
using namespace engine::test::pa;

std::vector<uint16_t> run_paged(const Batch& b, uint32_t seed) {
    return run_kernel(&paged_attention_decode, b, seed);
}

void check(const Batch& b, uint32_t seed) {
    check_kernel(&paged_attention_decode, b, seed);
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
