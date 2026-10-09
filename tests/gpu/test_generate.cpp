#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <span>
#include <stdexcept>
#include <vector>

#include "engine/runtime/generate.h"
#include "support/bf16.h"
#include "support/device.h"
#include "support/model.h"
#include "support/reference.h"
#include "support/safetensors_file.h"

using engine::generate_greedy;
using engine::ModelRunner;

namespace {

constexpr int kMaxNewTokens = 64;
constexpr int kNearTieUlps = 2;

class GenerateTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (const auto why = engine::test::reference_skip_reason(); !why.empty())
            GTEST_SKIP() << why;
    }

    const engine::DeviceWeights& w() const { return engine::test::tinyllama(); }

    engine::test::CudaStream stream;
};

const std::vector<int32_t> kPrompt = {1, 15043, 29892, 920, 526, 366};

}  // namespace

TEST_F(GenerateTest, FreeRunningGreedyMatchesHuggingFaceUpToNearTies) {
    ModelRunner runner(w().config, w().weights, 512, 2048, stream);
    const int64_t vocab = w().config.vocab;
    int identical = 0;
    for (const auto& prompt : engine::test::load_reference_manifest()) {
        SCOPED_TRACE(prompt.name);
        const engine::test::SafetensorsFile ref(prompt.file);
        const auto ids = ref.read<int32_t>("input_ids");
        const auto out = generate_greedy(runner, std::span(ids).first(size_t(prompt.prompt_len)),
                                         kMaxNewTokens, w().config.eos_id);
        if (out == prompt.generated_ids) {
            ++identical;
            std::printf("%-16s identical (%zu tokens)\n", prompt.name.c_str(), out.size());
            continue;
        }
        size_t step = 0;
        while (step < out.size() && step < prompt.generated_ids.size() &&
               out[step] == prompt.generated_ids[step])
            ++step;
        ASSERT_LT(step, out.size());
        ASSERT_LT(step, prompt.generated_ids.size());
        const auto logits = ref.read<uint16_t>("logits");
        const uint16_t* row =
            logits.data() + (size_t(prompt.prompt_len) + step - 1) * size_t(vocab);
        const int gap = engine::test::bf16_ulps(row[out[step]], row[prompt.generated_ids[step]]);
        std::printf(
            "%-16s differs at step %zu: ours %d, HF %d, %d BF16 ulps apart in HF's logits\n",
            prompt.name.c_str(), step, out[step], prompt.generated_ids[step], gap);
        EXPECT_LE(gap, kNearTieUlps) << "first difference at step " << step;
    }
    std::printf("free-running greedy identical to HF on %d of 20 prompts\n", identical);
}

TEST_F(GenerateTest, StopsAtMaxNewTokensOrAfterEos) {
    ModelRunner runner(w().config, w().weights, 16, 64, stream);
    const auto five = generate_greedy(runner, kPrompt, 5, -1);
    ASSERT_EQ(five.size(), 5u);
    EXPECT_EQ(runner.length(), int64_t(kPrompt.size()) + 4);
    EXPECT_EQ(generate_greedy(runner, kPrompt, 5, five[2]),
              std::vector<int32_t>(five.begin(), five.begin() + 3));
    EXPECT_EQ(generate_greedy(runner, kPrompt, 1, -1), std::vector<int32_t>{five[0]});
}

TEST_F(GenerateTest, ReportsEveryTokenInOrder) {
    ModelRunner runner(w().config, w().weights, 16, 64, stream);
    std::vector<int32_t> seen;
    const auto out =
        generate_greedy(runner, kPrompt, 8, -1, [&](int32_t token) { seen.push_back(token); });
    EXPECT_EQ(seen, out);
}

TEST_F(GenerateTest, RejectsBadLimits) {
    ModelRunner runner(w().config, w().weights, 16, 16, stream);
    EXPECT_THROW(generate_greedy(runner, kPrompt, 0, -1), std::invalid_argument);
    EXPECT_THROW(generate_greedy(runner, kPrompt, 12, -1), std::invalid_argument);
    EXPECT_EQ(generate_greedy(runner, kPrompt, 11, -1).size(), 11u);
    EXPECT_EQ(runner.length(), 16);
}
