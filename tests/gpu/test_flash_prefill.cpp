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
#include "engine/kernels/flash_prefill.h"
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
using engine::flash_attention_prefill;
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

struct Chunk {
    int32_t cached;
    int32_t count;
};

struct Step {
    Layout l{};
    int64_t rows = 0;
    int num_blocks = 0;
    int table_width = 0;
    std::vector<int32_t> starts, cached, tables;
    std::vector<uint16_t> qkv, k, v;

    int64_t chunks() const { return int64_t(cached.size()); }
    int32_t count(int64_t c) const { return starts[size_t(c) + 1] - starts[size_t(c)]; }

    int64_t cell(int64_t c, int64_t pos, int h, int d) const {
        const int64_t block = tables[size_t(c * table_width + pos / l.block_size)];
        return ((block * l.kv_heads + h) * l.block_size + pos % l.block_size) * l.head_dim + d;
    }

    bool in_chunk(int64_t row) const {
        return chunks() > 0 && row >= starts.front() && row < starts.back();
    }

    std::vector<bool> reached() const {
        std::vector<bool> used(k.size(), false);
        for (int64_t c = 0; c < chunks(); ++c)
            for (int64_t pos = 0; pos < cached[size_t(c)]; ++pos)
                for (int h = 0; h < l.kv_heads; ++h)
                    for (int d = 0; d < l.head_dim; ++d) used[size_t(cell(c, pos, h, d))] = true;
        return used;
    }
};

Step make_step(const Layout& l, int before, const std::vector<Chunk>& chunks, int after,
               uint32_t seed, float range = 2.0f) {
    std::mt19937 rng(seed);
    Step s;
    s.l = l;
    s.table_width = 1;
    int needed = 0;
    s.starts.push_back(before);
    for (const Chunk& c : chunks) {
        const int blocks = blocks_for(c.cached + c.count, l.block_size);
        needed += blocks;
        s.table_width = std::max(s.table_width, blocks + 2);
        s.cached.push_back(c.cached);
        s.starts.push_back(s.starts.back() + c.count);
    }
    s.rows = s.starts.back() + after;
    s.num_blocks = needed + 3;
    std::vector<int32_t> ids(static_cast<size_t>(s.num_blocks));
    std::iota(ids.begin(), ids.end(), 0);
    std::shuffle(ids.begin(), ids.end(), rng);
    size_t next = 0;
    for (const Chunk& c : chunks)
        for (int j = 0; j < s.table_width; ++j)
            s.tables.push_back(j < blocks_for(c.cached + c.count, l.block_size)
                                   ? ids[next++]
                                   : int32_t(rng() % uint32_t(s.num_blocks)));
    const auto cells = size_t(s.num_blocks * l.block_elems());
    s.qkv = random_bf16(size_t(s.rows * l.width()), seed + 1, -range, range);
    s.k = random_bf16(cells, seed + 2, -range, range);
    s.v = random_bf16(cells, seed + 3, -range, range);
    return s;
}

Step from_prompt(const Layout& l, const std::vector<uint16_t>& prompt, int32_t cached,
                 int32_t count, uint32_t seed) {
    Step s = make_step(l, 0, {{cached, count}}, 0, seed);
    s.qkv.assign(prompt.begin() + cached * l.width(),
                 prompt.begin() + (cached + count) * l.width());
    for (int64_t j = 0; j < cached; ++j)
        for (int h = 0; h < l.kv_heads; ++h)
            for (int d = 0; d < l.head_dim; ++d) {
                const auto col = size_t(j * l.width() + l.q_width() + h * l.head_dim + d);
                s.k[size_t(s.cell(0, j, h, d))] = prompt[col];
                s.v[size_t(s.cell(0, j, h, d))] = prompt[col + size_t(l.kv_width())];
            }
    return s;
}

Step single_chunk(const Step& s, int64_t c) {
    Step one = s;
    const int32_t first = s.starts[size_t(c)];
    one.rows = s.count(c);
    one.starts = {0, s.count(c)};
    one.cached = {s.cached[size_t(c)]};
    one.tables.assign(s.tables.begin() + c * s.table_width,
                      s.tables.begin() + (c + 1) * s.table_width);
    one.qkv.assign(s.qkv.begin() + first * s.l.width(),
                   s.qkv.begin() + (first + s.count(c)) * s.l.width());
    return one;
}

struct Device {
    DeviceBuffer out, qkv, k, v, tables, starts, cached;
};

Device upload_step(const Step& s, const std::vector<uint16_t>& init) {
    return {upload(init),     upload(s.qkv),    upload(s.k),     upload(s.v),
            upload(s.tables), upload(s.starts), upload(s.cached)};
}

void call(const Step& s, Device& d, engine::Stream stream) {
    const Layout& l = s.l;
    const std::initializer_list<int64_t> cache_shape{s.num_blocks, l.kv_heads, l.block_size,
                                                     l.head_dim};
    flash_attention_prefill(
        Tensor(static_cast<uint16_t*>(d.out.data()) + kGuard, DType::BF16, {s.rows, l.q_width()}),
        Tensor(d.qkv.data(), DType::BF16, {s.rows, l.width()}),
        Tensor(d.k.data(), DType::BF16, cache_shape), Tensor(d.v.data(), DType::BF16, cache_shape),
        Tensor(d.tables.data(), DType::I32, {s.chunks(), s.table_width}),
        Tensor(d.starts.data(), DType::I32, {s.chunks() + 1}),
        Tensor(d.cached.data(), DType::I32, {s.chunks()}), stream);
}

std::vector<uint16_t> run(const Step& s, uint32_t seed) {
    const int64_t n = s.rows * s.l.q_width();
    const auto init = random_bf16(size_t(n + 2 * kGuard), seed);
    Device d = upload_step(s, init);
    CudaStream stream;
    call(s, d, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    EXPECT_EQ(download<uint16_t>(d.qkv), s.qkv) << "qkv was modified";
    EXPECT_EQ(download<uint16_t>(d.k), s.k) << "k_cache was modified";
    EXPECT_EQ(download<uint16_t>(d.v), s.v) << "v_cache was modified";
    EXPECT_EQ(download<int32_t>(d.tables), s.tables) << "block_tables was modified";
    EXPECT_EQ(download<int32_t>(d.starts), s.starts) << "chunk_starts was modified";
    EXPECT_EQ(download<int32_t>(d.cached), s.cached) << "cached_lens was modified";

    const auto after = download<uint16_t>(d.out);
    for (int64_t i = 0; i < kGuard; ++i) {
        EXPECT_EQ(after[size_t(i)], init[size_t(i)]) << "write before out at " << i - kGuard;
        EXPECT_EQ(after[size_t(kGuard + n + i)], init[size_t(kGuard + n + i)])
            << "write after out at " << i;
    }
    for (int64_t r = 0; r < s.rows; ++r) {
        if (s.in_chunk(r)) continue;
        const auto row = size_t(kGuard + r * s.l.q_width());
        EXPECT_TRUE(std::equal(after.begin() + row, after.begin() + row + s.l.q_width(),
                               init.begin() + row))
            << "row " << r << " outside the chunks was written";
    }
    return {after.begin() + kGuard, after.begin() + kGuard + n};
}

Expected reference(const Step& s) {
    const Layout& l = s.l;
    const int group = l.heads / l.kv_heads;
    Expected e{std::vector<uint16_t>(size_t(s.rows * l.q_width())),
               std::vector<double>(size_t(s.rows * l.q_width()), -1.0)};
    CudaStream stream;
    for (int64_t c = 0; c < s.chunks(); ++c) {
        const int64_t count = s.count(c), cached = s.cached[size_t(c)], first = s.starts[size_t(c)];
        if (count == 0) continue;
        const int64_t context = cached + count;
        std::vector<uint16_t> kc(size_t(context * l.kv_width())), vc(kc.size());
        for (int64_t j = 0; j < context; ++j)
            for (int h = 0; h < l.kv_heads; ++h)
                for (int d = 0; d < l.head_dim; ++d) {
                    const auto i = size_t(j * l.kv_width() + h * l.head_dim + d);
                    if (j < cached) {
                        kc[i] = s.k[size_t(s.cell(c, j, h, d))];
                        vc[i] = s.v[size_t(s.cell(c, j, h, d))];
                    } else {
                        const auto col = size_t((first + j - cached) * l.width() + l.q_width() +
                                                h * l.head_dim + d);
                        kc[i] = s.qkv[col];
                        vc[i] = s.qkv[col + size_t(l.kv_width())];
                    }
                }
        const std::vector<uint16_t> q(s.qkv.begin() + first * l.width(),
                                      s.qkv.begin() + (first + count) * l.width());
        DeviceBuffer dq = upload(q), dk = upload(kc), dv = upload(vc);
        DeviceBuffer dout(size_t(count * l.q_width()) * sizeof(uint16_t));
        engine::naive::attention(Tensor(dout.data(), DType::BF16, {count, l.q_width()}),
                                 Tensor(dq.data(), DType::BF16, {count, l.width()}),
                                 Tensor(dk.data(), DType::BF16, {context, l.kv_width()}),
                                 Tensor(dv.data(), DType::BF16, {context, l.kv_width()}), cached,
                                 l.heads, l.kv_heads, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const auto naive = download<uint16_t>(dout);

        std::vector<double> v_max(size_t(l.kv_width()), 0.0);
        for (int64_t j = 0; j < context; ++j) {
            for (int64_t col = 0; col < l.kv_width(); ++col)
                v_max[size_t(col)] =
                    std::max(v_max[size_t(col)],
                             double(std::abs(from_bf16(vc[size_t(j * l.kv_width() + col)]))));
            if (j < cached) continue;
            const int64_t i = j - cached;
            for (int64_t col = 0; col < l.q_width(); ++col) {
                const int64_t kv_col = (col / l.head_dim) / group * l.head_dim + col % l.head_dim;
                const auto local = size_t(i * l.q_width() + col);
                const auto at = size_t((first + i) * l.q_width() + col);
                e.out[at] = naive[local];
                e.slack[at] =
                    kTolerance * (v_max[size_t(kv_col)] + std::abs(from_bf16(naive[local])));
            }
        }
    }
    return e;
}

void check(const Step& s, uint32_t seed) {
    expect_close(run(s, seed), reference(s));
}

std::vector<uint16_t> rows_of(const std::vector<uint16_t>& out, const Step& s, int64_t c) {
    const auto w = size_t(s.l.q_width());
    return {out.begin() + s.starts[size_t(c)] * w, out.begin() + s.starts[size_t(c) + 1] * w};
}

class FlashLayoutTest : public ::testing::TestWithParam<Layout> {};

}  // namespace

TEST(FlashAttentionPrefill, OneChunkOfThirtySevenTokens) {
    check(make_step(kTinyLlama, 0, {{0, 37}}, 0, 1), 2);
}

TEST(FlashAttentionPrefill, ChunkLengthsAroundTileSizes) {
    for (const int32_t len : {1, 2, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129}) {
        SCOPED_TRACE("chunk of " + std::to_string(len));
        check(make_step(kTinyLlama, 0, {{0, len}}, 0, 3 + uint32_t(len)), 4);
    }
}

TEST(FlashAttentionPrefill, LongPrompts) {
    for (const int32_t len : {2048, 4096, 8192}) {
        SCOPED_TRACE("prompt of " + std::to_string(len));
        check(make_step(kTinyLlama, 0, {{0, len}}, 0, 5 + uint32_t(len)), 6);
    }
}

TEST(FlashAttentionPrefill, ChunksAfterACachedPrefix) {
    for (const int32_t cached : {1, 15, 16, 17, 100, 1000})
        for (const int32_t count : {1, 37, 200}) {
            SCOPED_TRACE(std::to_string(cached) + " cached, chunk of " + std::to_string(count));
            check(make_step(kTinyLlama, 0, {{cached, count}}, 0, 7 + uint32_t(cached + count)), 8);
        }
}

TEST(FlashAttentionPrefill, RowsOutsideTheChunksAreNotWritten) {
    check(make_step(kTinyLlama, 5, {{0, 20}, {33, 7}}, 0, 9), 10);
    check(make_step(kTinyLlama, 3, {{16, 16}}, 4, 11), 12);
}

TEST(FlashAttentionPrefill, SixteenChunksOfMixedLengths) {
    std::mt19937 rng(13);
    std::vector<Chunk> chunks{{0, 1}, {0, 300}, {2000, 1}, {16, 16}};
    while (chunks.size() < 16) chunks.push_back({int32_t(rng() % 2001), 1 + int32_t(rng() % 300)});
    std::shuffle(chunks.begin(), chunks.end(), rng);
    check(make_step(kTinyLlama, 6, chunks, 0, 14), 15);
}

TEST(FlashAttentionPrefill, EmptyChunksWriteNothing) {
    check(make_step(kTinyLlama, 2, {{5, 0}, {0, 10}, {3, 0}, {40, 0}}, 1, 16), 17);
    check(make_step(kTinyLlama, 4, {{0, 0}}, 0, 18), 19);
}

TEST(FlashAttentionPrefill, LargeScoresDoNotOverflow) {
    Step s = make_step(kTinyLlama, 0, {{0, 1}, {0, 17}, {150, 150}}, 0, 20);
    std::mt19937 rng(21);
    const auto pick = [&] { return rng() % 10 ? to_bf16(4.0f) : uint16_t(0); };
    for (int64_t r = 0; r < s.rows; ++r)
        for (int64_t c = 0; c < s.l.q_width() + s.l.kv_width(); ++c)
            s.qkv[size_t(r * s.l.width() + c)] = pick();
    for (auto& x : s.k) x = pick();
    check(s, 22);
}

TEST(FlashAttentionPrefill, TheRunningMaximumMovesOnEveryTokenOrNever) {
    constexpr int kCached = 300, kCount = 300;
    for (const int shape : {0, 1, 2}) {
        SCOPED_TRACE(shape == 0   ? "ascending scores"
                     : shape == 1 ? "descending scores"
                                  : "one spike in the middle");
        Step s = make_step(kTinyLlama, 0, {{kCached, kCount}}, 0, 23 + uint32_t(shape));
        for (int64_t r = 0; r < s.rows; ++r)
            for (int c = 0; c < s.l.q_width(); ++c)
                s.qkv[size_t(r * s.l.width() + c)] = to_bf16(1.0f);
        for (int j = 0; j < kCached + kCount; ++j) {
            const int total = kCached + kCount;
            const float level = shape == 0   ? 0.01f * float(j)
                                : shape == 1 ? 0.01f * float(total - 1 - j)
                                             : (j == total / 2 ? 0.5f : 0.0f);
            for (int h = 0; h < s.l.kv_heads; ++h)
                for (int d = 0; d < s.l.head_dim; ++d) {
                    if (j < kCached)
                        s.k[size_t(s.cell(0, j, h, d))] = to_bf16(level);
                    else
                        s.qkv[size_t((j - kCached) * s.l.width() + s.l.q_width() +
                                     h * s.l.head_dim + d)] = to_bf16(level);
                }
        }
        check(s, 26);
    }
}

TEST(FlashAttentionPrefill, TheFirstPositionIsItsOwnValue) {
    const Step s = make_step(kTinyLlama, 3, {{0, 1}, {0, 5}, {0, 64}}, 0, 27);
    const auto out = run(s, 28);
    const Layout& l = s.l;
    const int group = l.heads / l.kv_heads;
    for (int64_t c = 0; c < s.chunks(); ++c) {
        const int64_t row = s.starts[size_t(c)];
        for (int h = 0; h < l.heads; ++h)
            for (int d = 0; d < l.head_dim; ++d)
                ASSERT_EQ(out[size_t(row * l.q_width() + h * l.head_dim + d)],
                          s.qkv[size_t(row * l.width() + l.q_width() + l.kv_width() +
                                       (h / group) * l.head_dim + d)])
                    << "chunk " << c << ", head " << h << ", dim " << d;
    }
}

TEST(FlashAttentionPrefill, OutputDependsOnlyOnTheContext) {
    const Step s = make_step(kTinyLlama, 2, {{0, 40}, {17, 30}, {100, 5}, {16, 1}}, 1, 29);
    const auto clean = run(s, 30);
    expect_close(clean, reference(s));
    const auto used = s.reached();

    Step nan = s;
    for (size_t i = 0; i < used.size(); ++i)
        if (!used[i]) nan.k[i] = nan.v[i] = kNan;
    for (int64_t r = 0; r < s.rows; ++r)
        if (!s.in_chunk(r))
            std::fill(nan.qkv.begin() + r * s.l.width(), nan.qkv.begin() + (r + 1) * s.l.width(),
                      kNan);
    EXPECT_EQ(run(nan, 30), clean) << "NaN outside the context reached the output";

    Step huge = s;
    for (size_t i = 0; i < used.size(); ++i)
        if (!used[i]) {
            huge.k[i] = to_bf16(i % 2 ? 30000.0f : -30000.0f);
            huge.v[i] = to_bf16(30000.0f);
        }
    for (int64_t c = 0; c < s.chunks(); ++c)
        for (int j = blocks_for(s.cached[size_t(c)], s.l.block_size); j < s.table_width; ++j)
            huge.tables[size_t(c * s.table_width + j)] = -1;
    EXPECT_EQ(run(huge, 30), clean) << "huge values outside the context reached the output";
}

TEST(FlashAttentionPrefill, AChunkDoesNotDependOnTheRestOfTheCall) {
    std::mt19937 rng(31);
    std::vector<Chunk> chunks{{0, 1}, {0, 129}, {1000, 64}, {5, 0}, {16, 17}};
    while (chunks.size() < 12) chunks.push_back({int32_t(rng() % 1500), 1 + int32_t(rng() % 200)});
    const Step s = make_step(kTinyLlama, 7, chunks, 0, 32);
    const auto together = run(s, 33);
    EXPECT_EQ(run(s, 33), together) << "two identical calls differ";
    for (int64_t c = 0; c < s.chunks(); ++c) {
        if (s.count(c) == 0) continue;
        EXPECT_EQ(run(single_chunk(s, c), 35), rows_of(together, s, c))
            << "chunk " << c << " alone differs from the same chunk in the call";
    }
}

TEST(FlashAttentionPrefill, ARowDoesNotDependOnTheRowsAfterIt) {
    for (const int32_t cached : {0, 50}) {
        SCOPED_TRACE(std::to_string(cached) + " cached");
        const Step full = make_step(kTinyLlama, 0, {{cached, 100}}, 0, 36 + uint32_t(cached));
        Step part = full;
        part.rows = 37;
        part.starts = {0, 37};
        part.qkv.resize(size_t(37 * full.l.width()));
        const auto all = run(full, 37);
        EXPECT_EQ(run(part, 38),
                  std::vector<uint16_t>(all.begin(), all.begin() + 37 * full.l.q_width()))
            << "the first 37 rows changed when the chunk ended after them";
    }
}

TEST(FlashAttentionPrefill, ChunkingsOfOnePromptAgree) {
    const Layout& l = kTinyLlama;
    constexpr int32_t kTokens = 300;
    const auto prompt = random_bf16(size_t(kTokens * l.width()), 39, -2.0f, 2.0f);
    const Step whole = from_prompt(l, prompt, 0, kTokens, 40);
    const Expected expected = reference(whole);
    expect_close(run(whole, 41), expected);
    for (const std::vector<int32_t>& cuts :
         {std::vector<int32_t>{0, 100, 200, 300}, std::vector<int32_t>{0, 1, 17, 300},
          std::vector<int32_t>{0, 256, 299, 300}}) {
        SCOPED_TRACE("cut at " + std::to_string(cuts[1]) + " and " + std::to_string(cuts[2]));
        std::vector<uint16_t> pieces;
        for (size_t i = 0; i + 1 < cuts.size(); ++i) {
            const Step s = from_prompt(l, prompt, cuts[i], cuts[i + 1] - cuts[i], 42 + uint32_t(i));
            const auto out = run(s, 45);
            expect_close(out, reference(s));
            pieces.insert(pieces.end(), out.begin(), out.end());
        }
        expect_close(pieces, expected);
    }
}

TEST(FlashAttentionPrefill, ReplaysFromACapturedGraph) {
    const Step s = make_step(kTinyLlama, 4, {{0, 70}, {33, 19}}, 0, 46);
    const auto eager = run(s, 47);
    const auto init = random_bf16(size_t(s.rows * s.l.q_width() + 2 * kGuard), 47);
    Device d = upload_step(s, init);
    CudaStream stream;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    try {
        call(s, d, stream);
    } catch (...) {
        cudaStreamEndCapture(stream, &graph);
        if (graph != nullptr) cudaGraphDestroy(graph);
        throw;
    }
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
    CUDA_CHECK(cudaGraphLaunch(exec, stream));
    CUDA_CHECK(cudaGraphLaunch(exec, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGraphExecDestroy(exec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    const auto after = download<uint16_t>(d.out);
    EXPECT_EQ(std::vector<uint16_t>(after.begin() + kGuard, after.end() - kGuard), eager)
        << "the graph replay differs from the eager call";
}

TEST_P(FlashLayoutTest, MatchesNaiveAttention) {
    const Layout l = GetParam();
    check(make_step(l, 1, {{0, 23}, {l.block_size + 1, 3 * l.block_size - 1}, {70, 1}, {0, 0}}, 0,
                    48),
          49);
    check(make_step(l, 0, {{0, 1}, {1, 1}, {2 * l.block_size, 70}}, 2, 50), 51);
}

INSTANTIATE_TEST_SUITE_P(FlashLayouts, FlashLayoutTest,
                         ::testing::Values(Layout{4, 2, 16, 5}, Layout{3, 3, 16, 1},
                                           Layout{6, 1, 32, 3}, Layout{6, 3, 48, 7},
                                           Layout{8, 2, 128, 16}, Layout{32, 4, 64, 32}),
                         [](const auto& info) {
                             const Layout& l = info.param;
                             return "heads" + std::to_string(l.heads) + "_kv" +
                                    std::to_string(l.kv_heads) + "_dim" +
                                    std::to_string(l.head_dim) + "_block" +
                                    std::to_string(l.block_size);
                         });

TEST(FlashAttentionPrefill, MatchesHuggingFaceAttnOnEveryTracedLayer) {
    if (const auto why = engine::test::reference_skip_reason(); !why.empty()) GTEST_SKIP() << why;
    const auto config = engine::load_config(engine::test::model_dir() / "config.json");
    const engine::test::SafetensorsFile ref(engine::test::traced_reference_prompt().file);
    const Layout l{config.heads, config.kv_heads, config.head_dim, 16};
    const auto tokens = int32_t(ref.entry("input_ids").shape[0]);
    const std::vector<int32_t> cuts{0, tokens / 3, 2 * tokens / 3 + 1, tokens};
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
        const auto roped = download<uint16_t>(dqkv);
        const auto expected = ref.read<uint16_t>(name + "attn");

        const auto whole = run(from_prompt(l, roped, 0, tokens, 52 + uint32_t(layer)), 53);
        EXPECT_EQ(outside_value_scale(whole, expected, v, l.heads, l.kv_heads, l.head_dim), 0u)
            << "one chunk, " << compare_bf16(whole, expected);

        std::vector<uint16_t> pieces;
        for (size_t i = 0; i + 1 < cuts.size(); ++i) {
            const auto out = run(
                from_prompt(l, roped, cuts[i], cuts[i + 1] - cuts[i], 80 + uint32_t(layer)), 54);
            pieces.insert(pieces.end(), out.begin(), out.end());
        }
        EXPECT_EQ(outside_value_scale(pieces, expected, v, l.heads, l.kv_heads, l.head_dim), 0u)
            << "three chunks, " << compare_bf16(pieces, expected);
    }
}

TEST(FlashAttentionPrefill, NoChunksOrNoRowsIsANoOp) {
    DeviceBuffer buf(2 * 4 * 16 * 64 * sizeof(uint16_t));
    DeviceBuffer zero = upload(std::vector<int32_t>{0});
    CudaStream stream;
    const Tensor cache(buf.data(), DType::BF16, {2, 4, 16, 64});
    const Tensor starts(zero.data(), DType::I32, {1});
    const Tensor none_cached(nullptr, DType::I32, {0});
    const Tensor none_tables(nullptr, DType::I32, {0, 128});
    flash_attention_prefill(Tensor(nullptr, DType::BF16, {0, 2048}),
                            Tensor(nullptr, DType::BF16, {0, 2560}), cache, cache, none_tables,
                            starts, none_cached, stream);
    DeviceBuffer rows(3 * 2560 * sizeof(uint16_t));
    flash_attention_prefill(Tensor(rows.data(), DType::BF16, {3, 2048}),
                            Tensor(rows.data(), DType::BF16, {3, 2560}), cache, cache, none_tables,
                            starts, none_cached, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TEST(FlashAttentionPrefill, RejectsBadArguments) {
    DeviceBuffer buf(1 << 16);
    void* p = buf.data();
    const Tensor out(p, DType::BF16, {3, 64});
    const Tensor qkv(p, DType::BF16, {3, 128});
    const Tensor cache(p, DType::BF16, {5, 2, 4, 16});
    const Tensor tables(p, DType::I32, {2, 4});
    const Tensor starts(p, DType::I32, {3});
    const Tensor cached(p, DType::I32, {2});
    const engine::Stream s = nullptr;
    const auto bad = [&](const Tensor& o, const Tensor& x, const Tensor& k, const Tensor& v,
                         const Tensor& t, const Tensor& st, const Tensor& c) {
        EXPECT_THROW(flash_attention_prefill(o, x, k, v, t, st, c, s), std::invalid_argument);
    };

    bad(Tensor(p, DType::F32, {3, 64}), qkv, cache, cache, tables, starts, cached);
    bad(Tensor(p, DType::BF16, {192}), qkv, cache, cache, tables, starts, cached);
    bad(Tensor(p, DType::BF16, {2, 64}), qkv, cache, cache, tables, starts, cached);
    bad(Tensor(p, DType::BF16, {3, 56}), Tensor(p, DType::BF16, {3, 120}), cache, cache, tables,
        starts, cached);
    bad(Tensor(p, DType::BF16, {3, 48}), Tensor(p, DType::BF16, {3, 112}), cache, cache, tables,
        starts, cached);
    bad(out, Tensor(p, DType::F32, {3, 128}), cache, cache, tables, starts, cached);
    bad(out, Tensor(p, DType::BF16, {384}), cache, cache, tables, starts, cached);
    bad(out, Tensor(p, DType::BF16, {3, 112}), cache, cache, tables, starts, cached);
    bad(out, Tensor(p, DType::BF16, {4, 128}), cache, cache, tables, starts, cached);
    bad(out, qkv, Tensor(p, DType::F16, {5, 2, 4, 16}), cache, tables, starts, cached);
    bad(out, qkv, cache, Tensor(p, DType::F32, {5, 2, 4, 16}), tables, starts, cached);
    bad(out, qkv, Tensor(p, DType::BF16, {5, 8, 16}), cache, tables, starts, cached);
    bad(out, qkv, cache, Tensor(p, DType::BF16, {5, 2, 8, 16}), tables, starts, cached);
    bad(out, qkv, cache, Tensor(p, DType::BF16, {4, 2, 4, 16}), tables, starts, cached);
    bad(out, qkv, Tensor(p, DType::BF16, {5, 4, 4, 16}), Tensor(p, DType::BF16, {5, 4, 4, 16}),
        tables, starts, cached);
    bad(Tensor(p, DType::BF16, {3, 32}), Tensor(p, DType::BF16, {3, 64}),
        Tensor(p, DType::BF16, {5, 2, 4, 8}), Tensor(p, DType::BF16, {5, 2, 4, 8}), tables, starts,
        cached);
    bad(Tensor(p, DType::BF16, {3, 144}), Tensor(p, DType::BF16, {3, 432}),
        Tensor(p, DType::BF16, {5, 1, 4, 144}), Tensor(p, DType::BF16, {5, 1, 4, 144}), tables,
        starts, cached);
    bad(out, qkv, cache, cache, Tensor(p, DType::I64, {2, 4}), starts, cached);
    bad(out, qkv, cache, cache, Tensor(p, DType::I32, {8}), starts, cached);
    bad(out, qkv, cache, cache, Tensor(p, DType::I32, {3, 4}), starts, cached);
    bad(out, qkv, cache, cache, Tensor(p, DType::I32, {2, 0}), starts, cached);
    bad(out, qkv, cache, cache, tables, Tensor(p, DType::I64, {3}), cached);
    bad(out, qkv, cache, cache, tables, Tensor(p, DType::I32, {3, 1}), cached);
    bad(out, qkv, cache, cache, tables, Tensor(p, DType::I32, {0}), cached);
    bad(out, qkv, cache, cache, tables, starts, Tensor(p, DType::I64, {2}));
    bad(out, qkv, cache, cache, tables, starts, Tensor(p, DType::I32, {2, 1}));
    bad(out, qkv, cache, cache, tables, starts, Tensor(p, DType::I32, {3}));
}
