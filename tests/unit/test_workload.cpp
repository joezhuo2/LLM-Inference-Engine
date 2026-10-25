#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#include "engine/bench/workload.h"

using engine::LengthRange;
using engine::make_workload;
using engine::parse_length_range;
using engine::WorkloadConfig;

namespace {

WorkloadConfig small(int n) {
    return {.num_requests = n, .prompt_len = {5, 9}, .output_len = {2, 4}, .vocab = 50};
}

}  // namespace

TEST(LengthRange, ParsesOneLengthOrARange) {
    EXPECT_EQ(parse_length_range("128").min, 128);
    EXPECT_EQ(parse_length_range("128").max, 128);
    EXPECT_EQ(parse_length_range("64:256").min, 64);
    EXPECT_EQ(parse_length_range("64:256").max, 256);
    EXPECT_EQ(parse_length_range("7:7").max, 7);
}

TEST(LengthRange, RejectsMalformedRanges) {
    for (const char* text :
         {"", "0", "-3", "12x", " 12", "1:", ":5", "1:2:3", "9:4", "0:4", "a:b", "99999999999"})
        EXPECT_THROW(parse_length_range(text), std::invalid_argument) << text;
}

TEST(Workload, FixedLengthsAreExact) {
    const auto w =
        make_workload({.num_requests = 8, .prompt_len = {128, 128}, .output_len = {256, 256}});
    ASSERT_EQ(w.requests.size(), 8u);
    for (const auto& r : w.requests) {
        EXPECT_EQ(r.prompt.size(), 128u);
        EXPECT_EQ(r.max_new_tokens, 256);
    }
}

TEST(Workload, LengthsCoverTheWholeRange) {
    const auto w = make_workload(small(2000));
    std::set<size_t> prompts;
    std::set<int> outputs;
    for (const auto& r : w.requests) {
        prompts.insert(r.prompt.size());
        outputs.insert(r.max_new_tokens);
    }
    EXPECT_EQ(prompts, (std::set<size_t>{5, 6, 7, 8, 9}));
    EXPECT_EQ(outputs, (std::set<int>{2, 3, 4}));
}

TEST(Workload, PromptsStartWithBosAndAvoidSpecialTokensAfter) {
    for (const auto& [bos, eos] :
         {std::pair{1, 2}, std::pair{2, 1}, std::pair{0, 49}, std::pair{48, 49}}) {
        auto config = small(300);
        config.bos_id = bos;
        config.eos_id = eos;
        std::set<int32_t> seen;
        for (const auto& r : make_workload(config).requests) {
            ASSERT_EQ(r.prompt[0], bos);
            for (size_t j = 1; j < r.prompt.size(); ++j) seen.insert(r.prompt[j]);
        }
        EXPECT_EQ(seen.size(), 48u) << bos << " " << eos;
        EXPECT_FALSE(seen.contains(bos));
        EXPECT_FALSE(seen.contains(eos));
        EXPECT_GE(*seen.begin(), 0);
        EXPECT_LT(*seen.rbegin(), 50);
    }
}

TEST(Workload, RequestsAreWaitingWithSequentialIdsAndSeeds) {
    auto config = small(5);
    config.seed = 3;
    config.params = {.temperature = 0.7f, .seed = 100, .top_k = 40, .top_p = 0.9f};
    const auto w = make_workload(config);
    for (int i = 0; i < 5; ++i) {
        const auto& r = w.requests[size_t(i)];
        EXPECT_EQ(r.id, i);
        EXPECT_EQ(r.params.seed, 100u + uint64_t(i));
        EXPECT_EQ(r.params.temperature, 0.7f);
        EXPECT_EQ(r.params.top_k, 40);
        EXPECT_EQ(r.params.top_p, 0.9f);
        EXPECT_TRUE(r.output.empty());
        EXPECT_EQ(r.state, engine::RequestState::Waiting);
        EXPECT_NO_THROW(validate(r, 2048));
    }
}

TEST(Workload, TheSameSeedGivesTheSameWorkload) {
    auto config = small(50);
    config.request_rate = 4.0;
    config.seed = 11;
    const auto a = make_workload(config);
    const auto b = make_workload(config);
    config.seed = 12;
    const auto c = make_workload(config);
    bool differs = false;
    for (size_t i = 0; i < 50; ++i) {
        EXPECT_EQ(a.requests[i].prompt, b.requests[i].prompt);
        EXPECT_EQ(a.requests[i].max_new_tokens, b.requests[i].max_new_tokens);
        EXPECT_EQ(a.arrivals[i], b.arrivals[i]);
        differs |= a.requests[i].prompt != c.requests[i].prompt;
    }
    EXPECT_TRUE(differs);
}

TEST(Workload, PinnedTokensKeepWorkloadsComparableAcrossMachines) {
    const auto w = make_workload({.num_requests = 2, .prompt_len = {4, 4}, .output_len = {1, 9}});
    EXPECT_EQ(w.requests[0].prompt, (std::vector<int32_t>{1, 7689, 23406, 81}));
    EXPECT_EQ(w.requests[1].prompt, (std::vector<int32_t>{1, 18454, 30427, 8986}));
    EXPECT_EQ(w.requests[0].max_new_tokens, 8);
    EXPECT_EQ(w.requests[1].max_new_tokens, 4);
}

TEST(Workload, EveryRequestArrivesAtOnceWithoutARate) {
    const auto w = make_workload(small(20));
    ASSERT_EQ(w.arrivals.size(), 20u);
    for (const double t : w.arrivals) EXPECT_EQ(t, 0.0);
}

TEST(Workload, ArrivalsFollowAPoissonProcessAtTheRate) {
    auto config = small(20001);
    config.request_rate = 8.0;
    const auto w = make_workload(config);
    EXPECT_EQ(w.arrivals.front(), 0.0);
    EXPECT_TRUE(std::is_sorted(w.arrivals.begin(), w.arrivals.end()));
    const double mean_gap = w.arrivals.back() / 20000.0;
    EXPECT_NEAR(mean_gap, 1.0 / 8.0, 0.03 / 8.0);
    int short_gaps = 0;
    for (size_t i = 1; i < w.arrivals.size(); ++i)
        short_gaps += w.arrivals[i] - w.arrivals[i - 1] < 1.0 / 8.0;
    EXPECT_NEAR(short_gaps / 20000.0, 1.0 - std::exp(-1.0), 0.015);
}

TEST(Workload, RejectsBadConfigs) {
    const auto bad = [](auto edit) {
        auto config = small(4);
        edit(config);
        return config;
    };
    for (const auto& config : {
             bad([](WorkloadConfig& c) { c.num_requests = 0; }),
             bad([](WorkloadConfig& c) { c.prompt_len = {0, 4}; }),
             bad([](WorkloadConfig& c) { c.prompt_len = {5, 4}; }),
             bad([](WorkloadConfig& c) { c.output_len = {0, 0}; }),
             bad([](WorkloadConfig& c) { c.output_len = {3, 2}; }),
             bad([](WorkloadConfig& c) { c.vocab = 2; }),
             bad([](WorkloadConfig& c) { c.bos_id = 50; }),
             bad([](WorkloadConfig& c) { c.eos_id = -1; }),
             bad([](WorkloadConfig& c) { c.eos_id = c.bos_id; }),
             bad([](WorkloadConfig& c) { c.request_rate = -1.0; }),
             bad([](WorkloadConfig& c) {
                 c.request_rate = std::numeric_limits<double>::infinity();
             }),
             bad([](WorkloadConfig& c) { c.request_rate = std::nan(""); }),
             bad([](WorkloadConfig& c) { c.params.temperature = -1.0f; }),
             bad([](WorkloadConfig& c) { c.params.top_p = 0.0f; }),
         })
        EXPECT_THROW(make_workload(config), std::invalid_argument);
}

TEST(Workload, FixedPromptsAreUsedInTurn) {
    WorkloadConfig c = small(5);
    c.prompts = {{1, 10, 11}, {1, 20}};
    const auto w = make_workload(c);
    ASSERT_EQ(w.requests.size(), 5u);
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_EQ(w.requests[i].prompt, c.prompts[i % 2]) << "request " << i;
        EXPECT_GE(w.requests[i].max_new_tokens, c.output_len.min);
        EXPECT_LE(w.requests[i].max_new_tokens, c.output_len.max);
    }
}

TEST(Workload, RejectsAnEmptyFixedPrompt) {
    WorkloadConfig c = small(2);
    c.prompts = {{1, 10}, {}};
    EXPECT_THROW(make_workload(c), std::invalid_argument);
}
