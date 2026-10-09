#include <gtest/gtest.h>

#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine/runtime/model_runner.h"
#include "support/device.h"
#include "support/logits.h"
#include "support/model.h"
#include "support/reference.h"
#include "support/safetensors_file.h"

using engine::ModelRunner;
using engine::Tensor;
using engine::test::argmax_rows;
using engine::test::count_equal;
using engine::test::CudaStream;
using engine::test::mean_abs_diff;
using engine::test::relative_l2;

namespace {

constexpr double kLayer0RelativeL2 = 0.01;
constexpr double kLayerRelativeL2 = 0.05;
constexpr double kMeanAbsLogitError = 0.05;

std::vector<uint16_t> to_host(const Tensor& t) {
    std::vector<uint16_t> out(size_t(t.numel()));
    CUDA_CHECK(cudaMemcpy(out.data(), t.data, t.bytes(), cudaMemcpyDeviceToHost));
    return out;
}

class ModelRunnerTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (const auto why = engine::test::reference_skip_reason(); !why.empty())
            GTEST_SKIP() << why;
    }

    const engine::DeviceWeights& w() const { return engine::test::tinyllama(); }
    int64_t vocab() const { return w().config.vocab; }

    CudaStream stream;
};

}  // namespace

TEST_F(ModelRunnerTest, MatchesHuggingFaceOnTheTracedPrompt) {
    const engine::test::SafetensorsFile ref(engine::test::traced_reference_prompt().file);
    const auto ids = ref.read<int32_t>("input_ids");
    ModelRunner runner(w().config, w().weights, 512, 2048, stream);
    const auto logits = to_host(runner.forward(ids, true, [&](int layer, const Tensor& hidden) {
        SCOPED_TRACE("layer " + std::to_string(layer));
        const double rel = relative_l2(
            to_host(hidden), ref.read<uint16_t>("layers." + std::to_string(layer) + ".out"));
        EXPECT_LE(rel, layer == 0 ? kLayer0RelativeL2 : kLayerRelativeL2);
    }));
    const auto expected = ref.read<uint16_t>("logits");
    EXPECT_LE(mean_abs_diff(logits, expected), kMeanAbsLogitError);
    EXPECT_GE(count_equal(argmax_rows(logits, vocab()), argmax_rows(expected, vocab())) * 100,
              ids.size() * 99);
    EXPECT_EQ(runner.length(), int64_t(ids.size()));
}

TEST_F(ModelRunnerTest, DecodingAfterPrefillMatchesOnePrefill) {
    const auto prompt = engine::test::traced_reference_prompt();
    const auto ids = engine::test::SafetensorsFile(prompt.file).read<int32_t>("input_ids");
    const auto n = size_t(prompt.prompt_len);
    ModelRunner runner(w().config, w().weights, 512, 2048, stream);
    const auto full = to_host(runner.forward(ids, true));
    const std::vector<uint16_t> generated(full.begin() + std::ptrdiff_t(n * size_t(vocab())),
                                          full.end());

    runner.reset();
    runner.forward(std::span(ids).first(n));
    std::vector<uint16_t> decoded;
    for (size_t t = n; t < ids.size(); ++t) {
        const auto row = to_host(runner.forward(std::span(ids).subspan(t, 1)));
        decoded.insert(decoded.end(), row.begin(), row.end());
    }
    EXPECT_LE(mean_abs_diff(decoded, generated), kMeanAbsLogitError);
    EXPECT_GE(count_equal(argmax_rows(decoded, vocab()), argmax_rows(generated, vocab())) + 1,
              ids.size() - n);
}

TEST_F(ModelRunnerTest, ReturnsOnlyTheLastRowUnlessAllLogitsAreRequested) {
    const std::vector<int32_t> ids = {1, 15043, 29892, 920, 526, 366};
    ModelRunner runner(w().config, w().weights, 16, 64, stream);
    const Tensor last = runner.forward(ids);
    EXPECT_EQ(last.ndim, 2);
    EXPECT_EQ(last.shape[0], 1);
    EXPECT_EQ(last.shape[1], vocab());
    const auto last_ids = argmax_rows(to_host(last), vocab());
    runner.reset();
    const auto all = argmax_rows(to_host(runner.forward(ids, true)), vocab());
    ASSERT_EQ(all.size(), ids.size());
    EXPECT_EQ(last_ids[0], all.back());
}

TEST_F(ModelRunnerTest, ResetStartsAgainAtPositionZero) {
    const std::vector<int32_t> ids = {1, 15043, 29892, 920, 526, 366};
    ModelRunner runner(w().config, w().weights, 16, 64, stream);
    const auto first = to_host(runner.forward(ids, true));
    runner.forward(ids);
    EXPECT_EQ(runner.length(), 12);
    runner.reset();
    EXPECT_EQ(runner.length(), 0);
    EXPECT_EQ(to_host(runner.forward(ids, true)), first);
}

TEST_F(ModelRunnerTest, RejectsBadSizes) {
    const auto& c = w().config;
    EXPECT_THROW(ModelRunner(c, w().weights, 0, 64, stream), std::invalid_argument);
    EXPECT_THROW(ModelRunner(c, w().weights, 65, 64, stream), std::invalid_argument);
    EXPECT_THROW(ModelRunner(c, w().weights, 16, c.max_positions + 1, stream),
                 std::invalid_argument);

    ModelRunner runner(c, w().weights, 16, 32, stream);
    const std::vector<int32_t> ids(17, 1);
    EXPECT_THROW(runner.forward({}), std::invalid_argument);
    EXPECT_THROW(runner.forward(ids), std::invalid_argument);
    runner.forward(std::span(ids).first(16));
    runner.forward(std::span(ids).first(15));
    EXPECT_THROW(runner.forward(std::span(ids).first(2)), std::invalid_argument);
    EXPECT_EQ(runner.length(), 31);
    EXPECT_NO_THROW(runner.forward(std::span(ids).first(1)));
    EXPECT_EQ(runner.length(), 32);
}
