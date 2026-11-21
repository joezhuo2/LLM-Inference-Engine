#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine/loader/tensor_parallel.h"
#include "engine/runtime/generate.h"
#include "engine/runtime/sharded_runner.h"
#include "engine/tokenizer/tokenizer.h"
#include "support/bf16.h"
#include "support/device.h"
#include "support/logits.h"
#include "support/model.h"

using engine::DeviceWeights;
using engine::ModelRunner;
using engine::ModelWeights;
using engine::ShardedRunner;
using engine::Tensor;
using engine::test::argmax_rows;
using engine::test::CudaStream;
using engine::test::from_bf16;

namespace {

constexpr double kMaxLogitDiff = 1.0;
constexpr int kDecodeSteps = 8;
constexpr int kGreedyTokens = 32;

struct Shards {
    std::vector<DeviceWeights> owned;
    std::vector<ModelWeights> weights;
};

Shards load_shards(int num_ranks) {
    Shards s;
    s.owned.reserve(size_t(num_ranks));
    for (int r = 0; r < num_ranks; ++r) {
        s.owned.push_back(
            engine::load_shard_weights(engine::test::model_dir(), r, num_ranks, nullptr));
        s.weights.push_back(s.owned.back().weights);
    }
    return s;
}

std::vector<uint16_t> to_host(const Tensor& t) {
    std::vector<uint16_t> out(size_t(t.numel()));
    CUDA_CHECK(cudaMemcpy(out.data(), t.data, t.bytes(), cudaMemcpyDeviceToHost));
    return out;
}

double max_abs_diff(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
    double worst = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        worst = std::max(worst, std::abs(double(from_bf16(a[i])) - from_bf16(b[i])));
    return worst;
}

std::vector<int32_t> prompt_ids() {
    const engine::Tokenizer tok(engine::test::model_dir() / "tokenizer.model");
    return tok.encode(
        "The old lighthouse keeper climbed the spiral stairs every evening to light the lamp. "
        "One night a storm rolled in from the sea, and he saw a small boat struggling against "
        "the waves near the rocks below.",
        true);
}

class ShardedRunnerTest : public ::testing::TestWithParam<int> {
protected:
    void SetUp() override {
        const auto file = engine::test::model_dir() / "model.safetensors";
        if (!std::filesystem::exists(file)) GTEST_SKIP() << file << " not found";
    }

    const DeviceWeights& w() const { return engine::test::tinyllama(); }

    CudaStream stream;
};

}  // namespace

TEST_P(ShardedRunnerTest, PrefillAndDecodeLogitsMatchTheWholeModel) {
    const int n = GetParam();
    const Shards shards = load_shards(n);
    const auto ids = prompt_ids();
    const int64_t vocab = w().config.vocab;
    ModelRunner whole(w().config, w().weights, 128, 256, stream);
    ShardedRunner sharded(w().config, shards.weights, 128, 256, stream);
    ASSERT_EQ(sharded.num_ranks(), n);

    const auto expected = to_host(whole.forward(ids, true));
    const auto actual = to_host(sharded.forward(ids, true));
    ASSERT_EQ(actual.size(), ids.size() * size_t(vocab));
    double worst = max_abs_diff(actual, expected);
    std::printf("ranks %d: prefill of %zu tokens, largest logit difference %.4f\n", n, ids.size(),
                worst);
    if (n == 1) {
        EXPECT_EQ(actual, expected);
    }
    EXPECT_LE(worst, kMaxLogitDiff);
    EXPECT_EQ(argmax_rows(actual, vocab).back(), argmax_rows(expected, vocab).back());

    int32_t token = argmax_rows(expected, vocab).back();
    for (int step = 0; step < kDecodeSteps; ++step) {
        SCOPED_TRACE("decode step " + std::to_string(step));
        const auto e = to_host(whole.forward(std::span(&token, 1)));
        const auto a = to_host(sharded.forward(std::span(&token, 1)));
        const double diff = max_abs_diff(a, e);
        worst = std::max(worst, diff);
        if (n == 1) {
            EXPECT_EQ(a, e);
        }
        EXPECT_LE(diff, kMaxLogitDiff);
        EXPECT_EQ(argmax_rows(a, vocab), argmax_rows(e, vocab));
        token = argmax_rows(e, vocab)[0];
    }
    std::printf("ranks %d: largest logit difference over prefill and %d decode steps %.4f\n", n,
                kDecodeSteps, worst);
    EXPECT_EQ(sharded.length(), whole.length());
}

TEST_P(ShardedRunnerTest, GreedyTokensMatchTheWholeModel) {
    const Shards shards = load_shards(GetParam());
    const auto ids = prompt_ids();
    ModelRunner whole(w().config, w().weights, 128, 256, stream);
    ShardedRunner sharded(w().config, shards.weights, 128, 256, stream);
    const auto expected = generate(whole, ids, kGreedyTokens, -1);
    const auto actual = generate(sharded, ids, kGreedyTokens, -1);
    ASSERT_EQ(expected.size(), size_t(kGreedyTokens));
    EXPECT_EQ(actual, expected);
}

INSTANTIATE_TEST_SUITE_P(Ranks, ShardedRunnerTest, ::testing::Values(1, 2, 4),
                         [](const auto& info) { return "ranks" + std::to_string(info.param); });

TEST_F(ShardedRunnerTest, RejectsBadArguments) {
    const auto& c = w().config;
    EXPECT_THROW(ShardedRunner(c, {}, 16, 64, stream), std::invalid_argument);
    EXPECT_THROW(ShardedRunner(c, {w().weights, w().weights}, 16, 64, stream),
                 std::invalid_argument);
    EXPECT_THROW(ShardedRunner(c, {w().weights, w().weights, w().weights}, 16, 64, stream),
                 std::invalid_argument);

    ModelWeights missing_layer = w().weights;
    missing_layer.layers.pop_back();
    EXPECT_THROW(ShardedRunner(c, {missing_layer}, 16, 64, stream), std::invalid_argument);

    const Shards two = load_shards(2);
    EXPECT_THROW(ShardedRunner(c, {two.weights[0]}, 16, 64, stream), std::invalid_argument);
    EXPECT_THROW(ShardedRunner(c, {two.weights[0], w().weights}, 16, 64, stream),
                 std::invalid_argument);
    EXPECT_THROW(ShardedRunner(c, two.weights, 0, 64, stream), std::invalid_argument);
    EXPECT_THROW(ShardedRunner(c, two.weights, 65, 64, stream), std::invalid_argument);
    EXPECT_THROW(ShardedRunner(c, two.weights, 16, c.max_positions + 1, stream),
                 std::invalid_argument);

    ShardedRunner runner(c, two.weights, 16, 32, stream);
    const std::vector<int32_t> ids(17, 1);
    EXPECT_THROW(runner.forward({}), std::invalid_argument);
    EXPECT_THROW(runner.forward(ids), std::invalid_argument);
    runner.forward(std::span(ids).first(16));
    runner.forward(std::span(ids).first(15));
    EXPECT_THROW(runner.forward(std::span(ids).first(2)), std::invalid_argument);
    EXPECT_EQ(runner.length(), 31);
}
