#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <vector>

#include "engine/bench/workload.h"
#include "engine/runtime/bench.h"
#include "engine/runtime/kv_cache.h"
#include "support/device.h"
#include "support/model.h"

using engine::Clock;
using engine::KvCache;
using engine::make_workload;
using engine::PagedRunner;
using engine::RequestState;
using engine::run_batched;
using engine::Sampler;
using engine::Scheduler;
using engine::SchedulerConfig;
using engine::Workload;
using engine::WorkloadConfig;

namespace {

constexpr int kBlockSize = 16;
constexpr int kNumBlocks = 256;
constexpr int kContext = 64;

class BatchedBenchTest : public ::testing::Test {
protected:
    void SetUp() override {
        const auto model = engine::test::model_dir() / "model.safetensors";
        if (!std::filesystem::exists(model)) GTEST_SKIP() << model << " not found";
    }

    const engine::DeviceWeights& w() const { return engine::test::tinyllama(); }

    WorkloadConfig config(int n) const {
        return {.num_requests = n,
                .prompt_len = {3, 20},
                .output_len = {2, 12},
                .vocab = w().config.vocab,
                .bos_id = w().config.bos_id,
                .eos_id = w().config.eos_id,
                .seed = 5};
    }

    engine::ServeStats run(Workload& workload, int max_batch, int num_blocks = kNumBlocks,
                           Clock::time_point start = Clock::now()) {
        KvCache cache(w().config, num_blocks, kBlockSize);
        PagedRunner runner(w().config, w().weights, cache, kContext, max_batch, kContext, stream);
        Scheduler scheduler(SchedulerConfig{.num_blocks = num_blocks,
                                            .block_size = kBlockSize,
                                            .max_batch = max_batch,
                                            .max_prefill_tokens = kContext,
                                            .max_context = kContext});
        Sampler sampler(max_batch);
        return run_batched(scheduler, runner, sampler, workload, start);
    }

    engine::test::CudaStream stream;
};

}  // namespace

TEST_F(BatchedBenchTest, EveryRequestFinishesWithItsFullOutputLength) {
    auto workload = make_workload(config(10));
    const auto stats = run(workload, 4);
    for (const auto& r : workload.requests) {
        SCOPED_TRACE(r.id);
        EXPECT_EQ(r.state, RequestState::Finished);
        EXPECT_EQ(int(r.output.size()), r.max_new_tokens);
        EXPECT_GE(r.first_token, r.arrived);
        EXPECT_GE(r.finished, r.first_token);
    }
    EXPECT_GT(stats.steps, 0);
    EXPECT_GT(stats.prefill_steps, 0);
}

TEST_F(BatchedBenchTest, OutputsMatchServeOnTheSameRequestsAndLimits) {
    auto bench = make_workload(config(8));
    run(bench, 8);

    const auto reference = make_workload(config(8));
    KvCache cache(w().config, kNumBlocks, kBlockSize);
    PagedRunner runner(w().config, w().weights, cache, kContext, 8, kContext, stream);
    Scheduler scheduler(SchedulerConfig{.num_blocks = kNumBlocks,
                                        .block_size = kBlockSize,
                                        .max_batch = 8,
                                        .max_prefill_tokens = kContext,
                                        .max_context = kContext});
    Sampler sampler(8);
    for (const auto& r : reference.requests) scheduler.submit(r);
    const auto served = engine::serve(scheduler, runner, sampler);
    ASSERT_EQ(served.size(), bench.requests.size());
    for (const auto& r : served) {
        SCOPED_TRACE(r.id);
        EXPECT_EQ(bench.requests[size_t(r.id)].output, r.output);
    }
}

TEST_F(BatchedBenchTest, ABatchLimitOfOneNeverRunsTwoRequestsTogether) {
    auto workload = make_workload(config(5));
    int64_t tokens = 0;
    for (const auto& r : workload.requests) tokens += r.max_new_tokens;
    const auto stats = run(workload, 1);
    EXPECT_EQ(stats.steps, tokens);
    EXPECT_EQ(stats.prefill_steps, 5);
}

TEST_F(BatchedBenchTest, FewestFreeBlocksCoversTheLargestSequenceAndLeavesTheRestFree) {
    auto workload = make_workload(config(6));
    int needed = 0;
    for (const auto& r : workload.requests) {
        const int tokens = int(r.prompt.size()) + r.max_new_tokens - 1;
        needed = std::max(needed, (tokens + kBlockSize - 1) / kBlockSize);
    }
    const auto stats = run(workload, 6);
    const int peak = kNumBlocks - stats.min_free_blocks;
    EXPECT_GE(peak, needed);
    EXPECT_LT(peak, kNumBlocks);
}

TEST_F(BatchedBenchTest, ArrivalsAreHonouredAndRecordedAtTheirScheduledTime) {
    auto workload = make_workload(config(3));
    workload.arrivals = {0.0, 0.2, 0.25};
    const auto start = Clock::now();
    run(workload, 4, kNumBlocks, start);
    for (size_t i = 0; i < 3; ++i) {
        const auto& r = workload.requests[i];
        const auto offset = std::chrono::duration<double>(r.arrived - start).count();
        EXPECT_NEAR(offset, workload.arrivals[i], 1e-6);
        EXPECT_GE(r.first_token, r.arrived);
    }
    EXPECT_GE(workload.requests[1].first_token - start, std::chrono::milliseconds(200));
}

TEST_F(BatchedBenchTest, RejectsMismatchedArrivalsAndDuplicateIds) {
    auto mismatched = make_workload(config(2));
    mismatched.arrivals.pop_back();
    EXPECT_THROW(run(mismatched, 2), std::invalid_argument);

    auto duplicate = make_workload(config(2));
    duplicate.requests[1].id = duplicate.requests[0].id;
    EXPECT_THROW(run(duplicate, 2), std::invalid_argument);
}
