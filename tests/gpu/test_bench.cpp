#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <vector>

#include "engine/bench/workload.h"
#include "engine/runtime/bench.h"
#include "engine/runtime/generate.h"
#include "support/device.h"
#include "support/model.h"

using engine::Clock;
using engine::make_workload;
using engine::ModelRunner;
using engine::RequestState;
using engine::run_one_at_a_time;
using engine::WorkloadConfig;

namespace {

class BenchTest : public ::testing::Test {
protected:
    void SetUp() override {
        const auto model = engine::test::model_dir() / "model.safetensors";
        if (!std::filesystem::exists(model)) GTEST_SKIP() << model << " not found";
    }

    const engine::DeviceWeights& w() const { return engine::test::tinyllama(); }

    WorkloadConfig config(int n) const {
        return {.num_requests = n,
                .prompt_len = {3, 12},
                .output_len = {1, 6},
                .vocab = w().config.vocab,
                .bos_id = w().config.bos_id,
                .eos_id = w().config.eos_id,
                .seed = 5};
    }

    engine::test::CudaStream stream;
};

}  // namespace

TEST_F(BenchTest, EachRequestMatchesGenerateAndRunsToItsLength) {
    ModelRunner runner(w().config, w().weights, 12, 18, stream);
    auto workload = make_workload(config(6));
    run_one_at_a_time(runner, workload, -1, Clock::now());
    for (const auto& r : workload.requests) {
        SCOPED_TRACE(r.id);
        EXPECT_EQ(r.state, RequestState::Finished);
        EXPECT_EQ(int(r.output.size()), r.max_new_tokens);
        EXPECT_EQ(r.output, engine::generate(runner, r.prompt, r.max_new_tokens, -1, r.params));
    }
}

TEST_F(BenchTest, SampledRequestsMatchGenerateWithTheirSeeds) {
    ModelRunner runner(w().config, w().weights, 12, 18, stream);
    auto c = config(4);
    c.params = {.temperature = 1.0f, .seed = 9, .top_k = 50, .top_p = 0.9f};
    auto workload = make_workload(c);
    run_one_at_a_time(runner, workload, -1, Clock::now());
    for (const auto& r : workload.requests)
        EXPECT_EQ(r.output, engine::generate(runner, r.prompt, r.max_new_tokens, -1, r.params));
}

TEST_F(BenchTest, TimestampsAreOrderedAndRequestsDoNotOverlap) {
    ModelRunner runner(w().config, w().weights, 12, 18, stream);
    auto workload = make_workload(config(5));
    const auto start = Clock::now();
    run_one_at_a_time(runner, workload, -1, start);
    Clock::time_point previous = start;
    for (const auto& r : workload.requests) {
        EXPECT_EQ(r.arrived, start);
        EXPECT_GT(r.first_token, previous);
        EXPECT_GE(r.finished, r.first_token);
        if (r.output.size() == 1) {
            EXPECT_EQ(r.finished, r.first_token);
        }
        previous = r.finished;
    }
}

TEST_F(BenchTest, WaitsForEachArrival) {
    ModelRunner runner(w().config, w().weights, 12, 18, stream);
    auto workload = make_workload(config(3));
    workload.arrivals = {0.0, 0.2, 0.25};
    const auto start = Clock::now();
    run_one_at_a_time(runner, workload, -1, start);
    for (size_t i = 0; i < 3; ++i) {
        const auto& r = workload.requests[i];
        const auto offset = std::chrono::duration<double>(r.arrived - start).count();
        EXPECT_NEAR(offset, workload.arrivals[i], 1e-6);
        EXPECT_GE(r.first_token, r.arrived);
    }
    EXPECT_GE(workload.requests[1].first_token - start, std::chrono::milliseconds(200));
}

TEST_F(BenchTest, StopsAtEosOnlyWhenAsked) {
    ModelRunner runner(w().config, w().weights, 12, 18, stream);
    auto workload = make_workload(config(1));
    auto& r = workload.requests[0];
    const auto free_run = engine::generate(runner, r.prompt, r.max_new_tokens, -1, r.params);
    run_one_at_a_time(runner, workload, free_run[0], Clock::now());
    EXPECT_EQ(r.output, std::vector<int32_t>{free_run[0]});
}

TEST_F(BenchTest, RejectsRequestsBeforeRunningAny) {
    ModelRunner runner(w().config, w().weights, 12, 16, stream);
    auto workload = make_workload(config(6));
    workload.requests.back().prompt.assign(12, 5);
    workload.requests.back().max_new_tokens = 6;
    EXPECT_THROW(run_one_at_a_time(runner, workload, -1, Clock::now()), std::invalid_argument);
    for (const auto& r : workload.requests) EXPECT_TRUE(r.output.empty());

    auto mismatched = make_workload(config(2));
    mismatched.arrivals.pop_back();
    EXPECT_THROW(run_one_at_a_time(runner, mismatched, -1, Clock::now()), std::invalid_argument);
}
