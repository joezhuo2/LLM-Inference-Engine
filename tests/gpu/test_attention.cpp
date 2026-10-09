#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine/kernels/attention.h"
#include "support/bf16.h"
#include "support/device.h"

using engine::DeviceBuffer;
using engine::DType;
using engine::Tensor;
using engine::naive::attention;
using engine::naive::store_kv;
using engine::test::compare_bf16;
using engine::test::CudaStream;
using engine::test::download;
using engine::test::from_bf16;
using engine::test::round_bf16;
using engine::test::to_bf16;
using engine::test::upload;

namespace {

struct Layout {
    int heads, kv_heads, head_dim;
    int64_t q_width() const { return int64_t(heads) * head_dim; }
    int64_t kv_width() const { return int64_t(kv_heads) * head_dim; }
    int64_t width() const { return q_width() + 2 * kv_width(); }
};

constexpr float kBf16Epsilon = 1.0f / 128.0f;

float attention_scale(int head_dim) {
    return float(1.0 / std::sqrt(double(head_dim)));
}

struct Expected {
    std::vector<uint16_t> out;
    std::vector<float> slack;
};

Expected cpu_attention(const std::vector<uint16_t>& qkv, const Layout& l) {
    const int64_t tokens = int64_t(qkv.size()) / l.width();
    const auto k_at = [&](int64_t j, int64_t i) {
        return from_bf16(qkv[j * l.width() + l.q_width() + i]);
    };
    const auto v_at = [&](int64_t j, int64_t i) {
        return from_bf16(qkv[j * l.width() + l.q_width() + l.kv_width() + i]);
    };
    const int group = l.heads / l.kv_heads;
    const float scale = attention_scale(l.head_dim);
    Expected e{std::vector<uint16_t>(size_t(tokens * l.q_width())),
               std::vector<float>(size_t(tokens * l.q_width()))};
    for (int64_t t = 0; t < tokens; ++t) {
        const int64_t context = t + 1;
        for (int h = 0; h < l.heads; ++h) {
            const uint16_t* q = qkv.data() + t * l.width() + h * l.head_dim;
            const int64_t kv = int64_t(h / group) * l.head_dim;
            std::vector<float> p(static_cast<size_t>(context));
            for (int64_t j = 0; j < context; ++j) {
                float dot = 0.0f;
                for (int d = 0; d < l.head_dim; ++d)
                    dot = std::fma(from_bf16(q[d]), k_at(j, kv + d), dot);
                p[j] = round_bf16(round_bf16(dot) * scale);
            }
            float m = p[0];
            for (const float s : p) m = std::max(m, s);
            float sum = 0.0f;
            for (float& s : p) sum += s = std::exp(s - m);
            for (float& s : p) s = round_bf16(s / sum);
            for (int d = 0; d < l.head_dim; ++d) {
                float acc = 0.0f, weighted_abs = 0.0f;
                for (int64_t j = 0; j < context; ++j) {
                    acc = std::fma(p[j], v_at(j, kv + d), acc);
                    weighted_abs += p[j] * std::abs(v_at(j, kv + d));
                }
                const size_t i = size_t(t * l.q_width() + h * l.head_dim + d);
                e.out[i] = to_bf16(acc);
                e.slack[i] = kBf16Epsilon * (weighted_abs + std::abs(acc));
            }
        }
    }
    return e;
}

size_t outside_softmax_rounding(const std::vector<uint16_t>& actual, const Expected& e) {
    size_t n = 0;
    for (size_t i = 0; i < actual.size(); ++i)
        if (std::abs(from_bf16(actual[i]) - from_bf16(e.out[i])) > e.slack[i]) ++n;
    return n;
}

class Cache {
public:
    Cache(int64_t rows, const Layout& l)
        : l_(l),
          rows_(rows),
          k_(size_t(rows * l.kv_width()) * sizeof(uint16_t)),
          v_(size_t(rows * l.kv_width()) * sizeof(uint16_t)) {}

    Tensor k() { return Tensor(k_.data(), DType::BF16, {rows_, l_.kv_width()}); }
    Tensor v() { return Tensor(v_.data(), DType::BF16, {rows_, l_.kv_width()}); }

    std::vector<uint16_t> step(const std::vector<uint16_t>& qkv, int64_t start,
                               cudaStream_t stream) {
        const int64_t tokens = int64_t(qkv.size()) / l_.width();
        DeviceBuffer dqkv = upload(qkv);
        DeviceBuffer out(size_t(tokens * l_.q_width()) * sizeof(uint16_t));
        const Tensor x(dqkv.data(), DType::BF16, {tokens, l_.width()});
        store_kv(k(), v(), x, start, stream);
        attention(Tensor(out.data(), DType::BF16, {tokens, l_.q_width()}), x, k(), v(), start,
                  l_.heads, l_.kv_heads, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        return download<uint16_t>(out);
    }

private:
    Layout l_;
    int64_t rows_;
    DeviceBuffer k_, v_;
};

std::vector<uint16_t> rows(const std::vector<uint16_t>& v, int64_t width, int64_t begin,
                           int64_t end) {
    return {v.begin() + begin * width, v.begin() + end * width};
}

class AttentionLayoutTest : public ::testing::TestWithParam<Layout> {};

}  // namespace

TEST(StoreKv, CopiesKAndVRowsToTheirCachePositions) {
    constexpr int64_t tokens = 3, q_width = 8, kv_width = 4, width = q_width + 2 * kv_width;
    constexpr int64_t rows = 6, start = 2;
    const auto qkv = engine::test::random_bf16(size_t(tokens * width), 21);
    const auto k_init = engine::test::random_bf16(size_t(rows * kv_width), 22);
    const auto v_init = engine::test::random_bf16(size_t(rows * kv_width), 23);

    CudaStream stream;
    DeviceBuffer dqkv = upload(qkv), dk = upload(k_init), dv = upload(v_init);
    store_kv(Tensor(dk.data(), DType::BF16, {rows, kv_width}),
             Tensor(dv.data(), DType::BF16, {rows, kv_width}),
             Tensor(dqkv.data(), DType::BF16, {tokens, width}), start, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto k = download<uint16_t>(dk), v = download<uint16_t>(dv);

    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t i = 0; i < kv_width; ++i) {
            const size_t c = size_t(r * kv_width + i);
            if (r < start || r >= start + tokens) {
                EXPECT_EQ(k[c], k_init[c]) << "row " << r;
                EXPECT_EQ(v[c], v_init[c]) << "row " << r;
            } else {
                const size_t row = size_t((r - start) * width);
                EXPECT_EQ(k[c], qkv[row + q_width + i]) << "row " << r;
                EXPECT_EQ(v[c], qkv[row + q_width + kv_width + i]) << "row " << r;
            }
        }
    }
}

TEST(StoreKv, ZeroTokensIsANoOp) {
    DeviceBuffer buf(64);
    CudaStream stream;
    const Tensor cache(buf.data(), DType::BF16, {4, 4});
    store_kv(cache, cache, Tensor(nullptr, DType::BF16, {0, 16}), 4, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TEST(StoreKv, RejectsBadShapesAndRanges) {
    DeviceBuffer buf(4096);
    void* p = buf.data();
    CudaStream stream;
    const Tensor cache(p, DType::BF16, {4, 4});
    const Tensor qkv(p, DType::BF16, {2, 16});
    EXPECT_THROW(store_kv(cache, cache, qkv, 3, stream), std::invalid_argument);
    EXPECT_THROW(store_kv(cache, cache, qkv, -1, stream), std::invalid_argument);
    EXPECT_THROW(store_kv(cache, cache, Tensor(p, DType::BF16, {2, 8}), 0, stream),
                 std::invalid_argument);
    EXPECT_THROW(store_kv(cache, Tensor(p, DType::BF16, {4, 2}), qkv, 0, stream),
                 std::invalid_argument);
    EXPECT_THROW(store_kv(Tensor(p, DType::F32, {4, 4}), cache, qkv, 0, stream),
                 std::invalid_argument);
    EXPECT_THROW(store_kv(cache, cache, Tensor(p, DType::F32, {2, 16}), 0, stream),
                 std::invalid_argument);
}

TEST_P(AttentionLayoutTest, MatchesCpuReference) {
    const Layout l = GetParam();
    for (const int64_t tokens : {1, 2, 9, 300}) {
        SCOPED_TRACE("T = " + std::to_string(tokens));
        const auto qkv = engine::test::random_bf16(size_t(tokens * l.width()), 31, -2.0f, 2.0f);
        CudaStream stream;
        Cache cache(tokens, l);
        const auto actual = cache.step(qkv, 0, stream);
        const auto expected = cpu_attention(qkv, l);
        const auto diff = compare_bf16(actual, expected.out);
        EXPECT_LE(diff.mismatches * 1000, actual.size()) << diff;
        EXPECT_EQ(outside_softmax_rounding(actual, expected), 0u) << diff;
    }
}

TEST_P(AttentionLayoutTest, DecodingAfterPrefillMatchesOneFullPrefill) {
    const Layout l = GetParam();
    constexpr int64_t tokens = 40, prompt = 13;
    const auto qkv = engine::test::random_bf16(size_t(tokens * l.width()), 32, -2.0f, 2.0f);
    CudaStream stream;
    Cache full(tokens, l), incremental(tokens + 7, l);
    const auto expected = full.step(qkv, 0, stream);
    auto actual = incremental.step(rows(qkv, l.width(), 0, prompt), 0, stream);
    for (int64_t t = prompt; t < tokens; ++t) {
        const auto row = incremental.step(rows(qkv, l.width(), t, t + 1), t, stream);
        actual.insert(actual.end(), row.begin(), row.end());
    }
    EXPECT_EQ(actual, expected);
}

TEST_P(AttentionLayoutTest, ASingleTokenAttendsOnlyToItsOwnValue) {
    const Layout l = GetParam();
    const auto qkv = engine::test::random_bf16(size_t(l.width()), 33, -2.0f, 2.0f);
    CudaStream stream;
    Cache cache(4, l);
    const auto out = cache.step(qkv, 0, stream);
    const int group = l.heads / l.kv_heads;
    for (int h = 0; h < l.heads; ++h)
        for (int d = 0; d < l.head_dim; ++d)
            ASSERT_EQ(out[size_t(h * l.head_dim + d)],
                      qkv[size_t(l.q_width() + l.kv_width() + (h / group) * l.head_dim + d)])
                << "head " << h;
}

INSTANTIATE_TEST_SUITE_P(Layouts, AttentionLayoutTest,
                         ::testing::Values(Layout{4, 2, 8}, Layout{3, 3, 16}, Layout{6, 1, 4},
                                           Layout{32, 4, 64}),
                         [](const auto& info) {
                             const Layout& l = info.param;
                             return "heads" + std::to_string(l.heads) + "_kv" +
                                    std::to_string(l.kv_heads) + "_dim" +
                                    std::to_string(l.head_dim);
                         });

TEST(Attention, IgnoresCacheRowsBeyondTheQueryPosition) {
    const Layout l{4, 2, 8};
    constexpr int64_t tokens = 5;
    const auto qkv = engine::test::random_bf16(size_t(tokens * l.width()), 34, -2.0f, 2.0f);
    CudaStream stream;
    Cache clean(tokens, l), dirty(tokens + 3, l);
    const auto garbage =
        engine::test::random_bf16(size_t((tokens + 3) * l.width()), 35, -9.0f, 9.0f);
    dirty.step(garbage, 0, stream);
    EXPECT_EQ(dirty.step(qkv, 0, stream), clean.step(qkv, 0, stream));
}

TEST(Attention, ZeroTokensIsANoOp) {
    DeviceBuffer buf(256);
    CudaStream stream;
    const Tensor cache(buf.data(), DType::BF16, {4, 16});
    attention(Tensor(nullptr, DType::BF16, {0, 64}), Tensor(nullptr, DType::BF16, {0, 96}), cache,
              cache, 4, 4, 1, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TEST(Attention, RejectsBadShapesAndRanges) {
    DeviceBuffer buf(4096);
    void* p = buf.data();
    CudaStream stream;
    const Tensor out(p, DType::BF16, {2, 32});
    const Tensor qkv(p, DType::BF16, {2, 64});
    const Tensor cache(p, DType::BF16, {8, 16});
    EXPECT_NO_THROW(attention(out, qkv, cache, cache, 6, 4, 2, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    EXPECT_THROW(attention(out, qkv, cache, cache, 7, 4, 2, stream), std::invalid_argument);
    EXPECT_THROW(attention(out, qkv, cache, cache, -1, 4, 2, stream), std::invalid_argument);
    EXPECT_THROW(attention(out, qkv, cache, cache, 0, 3, 2, stream), std::invalid_argument);
    EXPECT_THROW(attention(out, qkv, cache, cache, 0, 4, 0, stream), std::invalid_argument);
    EXPECT_THROW(attention(out, Tensor(p, DType::BF16, {2, 60}), cache, cache, 0, 4, 2, stream),
                 std::invalid_argument);
    EXPECT_THROW(attention(Tensor(p, DType::BF16, {3, 32}), qkv, cache, cache, 0, 4, 2, stream),
                 std::invalid_argument);
    EXPECT_THROW(attention(Tensor(p, DType::BF16, {2, 16}), qkv, cache, cache, 0, 4, 2, stream),
                 std::invalid_argument);
    EXPECT_THROW(attention(out, qkv, Tensor(p, DType::BF16, {8, 8}), cache, 0, 4, 2, stream),
                 std::invalid_argument);
    EXPECT_THROW(attention(out, qkv, cache, Tensor(p, DType::BF16, {7, 16}), 0, 4, 2, stream),
                 std::invalid_argument);
    EXPECT_THROW(attention(out, Tensor(p, DType::F32, {2, 64}), cache, cache, 0, 4, 2, stream),
                 std::invalid_argument);
    EXPECT_THROW(attention(out, qkv, Tensor(p, DType::BF16, {20000, 16}),
                           Tensor(p, DType::BF16, {20000, 16}), 12280, 4, 2, stream),
                 std::invalid_argument);
}
