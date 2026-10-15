#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

#include "engine/runtime/kv_cache.h"
#include "engine/runtime/model_runner.h"
#include "engine/runtime/paged_runner.h"
#include "engine/sched/paged_batch.h"
#include "support/device.h"
#include "support/logits.h"
#include "support/model.h"
#include "support/reference.h"
#include "support/safetensors_file.h"

using engine::KvCache;
using engine::ModelRunner;
using engine::PagedRunner;
using engine::Request;
using engine::Scheduler;
using engine::SchedulerConfig;
using engine::Tensor;
using engine::test::argmax_rows;
using engine::test::CudaStream;
using engine::test::relative_l2;

namespace {

constexpr double kLogitRelativeL2 = 0.05;
constexpr double kMeanAbsLogitError = 0.05;
constexpr double kArgmaxMatch = 0.99;
constexpr int kBlockSize = 16;
constexpr int kMaxContext = 128;

std::vector<uint16_t> to_host(const Tensor& t) {
    std::vector<uint16_t> out(size_t(t.numel()));
    CUDA_CHECK(cudaMemcpy(out.data(), t.data, t.bytes(), cudaMemcpyDeviceToHost));
    return out;
}

std::vector<int32_t> prompt_of(int64_t id, int length, int vocab) {
    std::vector<int32_t> p;
    for (int i = 0; i < length; ++i)
        p.push_back(int32_t(1 + (id * 7919 + i * 104729) % (vocab - 1)));
    return p;
}

class PagedRunnerTest : public ::testing::Test {
protected:
    const engine::DeviceWeights& w() const { return engine::test::tinyllama(); }
    int64_t vocab() const { return w().config.vocab; }

    CudaStream stream;
};

}  // namespace

TEST_F(PagedRunnerTest, MatchesTheContiguousRunnerOnAMixedBatchOverSeveralSteps) {
    const std::vector<int> lengths = {6, 20, 1, 33};
    constexpr int kSteps = 8;
    KvCache cache(w().config, 64, kBlockSize);
    PagedRunner runner(w().config, w().weights, cache, 128, 4, kMaxContext, stream);
    Scheduler scheduler(SchedulerConfig{.num_blocks = 64,
                                        .block_size = kBlockSize,
                                        .max_batch = 4,
                                        .max_prefill_tokens = kMaxContext,
                                        .max_context = kMaxContext});
    std::vector<std::unique_ptr<ModelRunner>> solo;
    for (size_t i = 0; i < lengths.size(); ++i) {
        solo.push_back(
            std::make_unique<ModelRunner>(w().config, w().weights, 64, kMaxContext, stream));
        Request r{.id = int64_t(i), .max_new_tokens = kSteps};
        r.prompt = prompt_of(int64_t(i), lengths[i], int(vocab()));
        scheduler.submit(r);
    }

    double worst = 0.0;
    for (int step_no = 0; step_no < kSteps; ++step_no) {
        const auto step = scheduler.schedule();
        ASSERT_EQ(step.prefill, step_no == 0);
        ASSERT_EQ(step.seqs.size(), lengths.size());
        const auto batch = engine::build_paged_batch(step, scheduler, int(runner.max_blocks()));
        const auto paged = to_host(runner.forward(batch));
        std::vector<int32_t> tokens;
        for (size_t i = 0; i < step.seqs.size(); ++i) {
            SCOPED_TRACE("step " + std::to_string(step_no) + " seq " + std::to_string(i));
            const Request& r = scheduler.request(step.seqs[i]);
            ModelRunner& m = *solo[size_t(step.seqs[i])];
            const std::vector<int32_t> feed =
                step.prefill ? r.prompt : std::vector<int32_t>{r.output.back()};
            const auto expected = to_host(m.forward(feed));
            const std::vector<uint16_t> row(paged.begin() + int64_t(i) * vocab(),
                                            paged.begin() + int64_t(i + 1) * vocab());
            const double rel = relative_l2(row, expected);
            worst = std::max(worst, rel);
            EXPECT_LE(rel, kLogitRelativeL2);
            tokens.push_back(argmax_rows(expected, vocab())[0]);
        }
        scheduler.update(tokens);
    }
    std::printf("worst relative L2 of paged logits against the contiguous runner: %g\n", worst);
}

TEST_F(PagedRunnerTest, ASingleTokenPromptIsBitIdenticalToTheContiguousRunner) {
    KvCache cache(w().config, 16, kBlockSize);
    PagedRunner runner(w().config, w().weights, cache, 16, 1, kMaxContext, stream);
    Scheduler scheduler(SchedulerConfig{.num_blocks = 16,
                                        .block_size = kBlockSize,
                                        .max_batch = 1,
                                        .max_prefill_tokens = kMaxContext,
                                        .max_context = kMaxContext});
    Request r{.id = 1, .max_new_tokens = 2};
    r.prompt = {1};
    scheduler.submit(r);
    const auto paged = to_host(runner.forward(
        engine::build_paged_batch(scheduler.schedule(), scheduler, int(runner.max_blocks()))));
    ModelRunner solo(w().config, w().weights, 16, kMaxContext, stream);
    EXPECT_EQ(paged, to_host(solo.forward(r.prompt)));
}

TEST_F(PagedRunnerTest, AStepDoesNotDisturbTheOtherSequencesBlocks) {
    KvCache cache(w().config, 32, kBlockSize);
    PagedRunner runner(w().config, w().weights, cache, 64, 2, kMaxContext, stream);
    Scheduler scheduler(SchedulerConfig{.num_blocks = 32,
                                        .block_size = kBlockSize,
                                        .max_batch = 2,
                                        .max_prefill_tokens = kMaxContext,
                                        .max_context = kMaxContext});
    Request a{.id = 1, .max_new_tokens = 2};
    a.prompt = prompt_of(1, 20, int(vocab()));
    scheduler.submit(a);
    const auto step = scheduler.schedule();
    const auto first = to_host(
        runner.forward(engine::build_paged_batch(step, scheduler, int(runner.max_blocks()))));
    scheduler.update(std::vector<int32_t>{argmax_rows(first, vocab())[0]});

    Request b{.id = 2, .max_new_tokens = 2};
    b.prompt = prompt_of(2, 25, int(vocab()));
    scheduler.submit(b);
    const auto admit = scheduler.schedule();
    ASSERT_TRUE(admit.prefill);
    ASSERT_EQ(admit.seqs, std::vector<int64_t>{2});
    runner.forward(engine::build_paged_batch(admit, scheduler, int(runner.max_blocks())));
    scheduler.update(std::vector<int32_t>{5});

    ModelRunner solo(w().config, w().weights, 64, kMaxContext, stream);
    solo.forward(a.prompt);
    const auto expected =
        to_host(solo.forward(std::vector<int32_t>{argmax_rows(first, vocab())[0]}));
    const auto decode = scheduler.schedule();
    ASSERT_FALSE(decode.prefill);
    ASSERT_EQ(decode.seqs.size(), 2u);
    const auto paged = to_host(
        runner.forward(engine::build_paged_batch(decode, scheduler, int(runner.max_blocks()))));
    const std::vector<uint16_t> row(paged.begin(), paged.begin() + vocab());
    EXPECT_LE(relative_l2(row, expected), kLogitRelativeL2);
}

TEST_F(PagedRunnerTest, RejectsBadArguments) {
    KvCache cache(w().config, 32, kBlockSize);
    EXPECT_THROW(PagedRunner(w().config, w().weights, cache, 16, 17, kMaxContext, stream),
                 std::invalid_argument);
    EXPECT_THROW(PagedRunner(w().config, w().weights, cache, 16, 0, kMaxContext, stream),
                 std::invalid_argument);
    EXPECT_THROW(PagedRunner(w().config, w().weights, cache, 16, 4, 0, stream),
                 std::invalid_argument);

    PagedRunner runner(w().config, w().weights, cache, 16, 4, kMaxContext, stream);
    engine::PagedBatch batch;
    EXPECT_THROW(runner.forward(batch), std::invalid_argument);
    batch.num_seqs = 1;
    batch.max_blocks = runner.max_blocks();
    batch.token_ids = {1, 2};
    batch.positions = {0, 1};
    batch.slot_mapping = {16, 17};
    batch.context_lens = {1, 2};
    batch.logit_rows = {1};
    batch.block_tables.assign(size_t(2 * runner.max_blocks()), 0);
    batch.block_tables[0] = batch.block_tables[size_t(runner.max_blocks())] = 1;
    batch.logit_rows = {2};
    EXPECT_THROW(runner.forward(batch), std::invalid_argument);
    batch.logit_rows = {1};
    batch.max_blocks = runner.max_blocks() + 1;
    EXPECT_THROW(runner.forward(batch), std::invalid_argument);
    batch.max_blocks = runner.max_blocks();
    batch.positions.pop_back();
    EXPECT_THROW(runner.forward(batch), std::invalid_argument);
}

TEST_F(PagedRunnerTest, ConcurrentSequencesMatchHuggingFaceTeacherForced) {
    if (const auto why = engine::test::reference_skip_reason(); !why.empty()) GTEST_SKIP() << why;
    struct Item {
        std::vector<int32_t> ids;
        std::vector<uint16_t> logits;
        int prompt_len;
    };
    std::vector<Item> items;
    for (const auto& prompt : engine::test::load_reference_manifest()) {
        const engine::test::SafetensorsFile ref(prompt.file);
        items.push_back(
            {ref.read<int32_t>("input_ids"), ref.read<uint16_t>("logits"), prompt.prompt_len});
    }
    ASSERT_FALSE(items.empty());

    constexpr int kMaxBatch = 16;
    constexpr int kContext = 2048;
    const int num_blocks = 1024;
    KvCache cache(w().config, num_blocks, kBlockSize);
    PagedRunner runner(w().config, w().weights, cache, kContext, kMaxBatch, kContext, stream);
    Scheduler scheduler(SchedulerConfig{.num_blocks = num_blocks,
                                        .block_size = kBlockSize,
                                        .max_batch = kMaxBatch,
                                        .max_prefill_tokens = kContext,
                                        .max_context = kContext});
    for (size_t i = 0; i < items.size(); ++i) {
        const Item& item = items[i];
        Request r{.id = int64_t(i), .max_new_tokens = int(item.ids.size()) - item.prompt_len + 1};
        r.prompt.assign(item.ids.begin(), item.ids.begin() + item.prompt_len);
        scheduler.submit(r);
    }

    size_t positions = 0, matches = 0;
    double abs_error = 0.0;
    int max_running = 0;
    while (scheduler.num_waiting() + scheduler.num_running() > 0) {
        const auto step = scheduler.schedule();
        ASSERT_FALSE(step.seqs.empty());
        max_running = std::max(max_running, scheduler.num_running());
        const auto logits = to_host(
            runner.forward(engine::build_paged_batch(step, scheduler, int(runner.max_blocks()))));
        std::vector<int32_t> tokens;
        for (size_t i = 0; i < step.seqs.size(); ++i) {
            const Item& item = items[size_t(step.seqs[i])];
            const size_t generated = scheduler.request(step.seqs[i]).output.size();
            const size_t row = size_t(item.prompt_len) - 1 + generated;
            ASSERT_LT(row, item.ids.size());
            const std::vector<uint16_t> ours(logits.begin() + int64_t(i) * vocab(),
                                             logits.begin() + int64_t(i + 1) * vocab());
            const std::vector<uint16_t> theirs(item.logits.begin() + int64_t(row) * vocab(),
                                               item.logits.begin() + int64_t(row + 1) * vocab());
            matches += argmax_rows(ours, vocab())[0] == argmax_rows(theirs, vocab())[0];
            abs_error += engine::test::mean_abs_diff(ours, theirs);
            ++positions;
            tokens.push_back(row + 1 < item.ids.size() ? item.ids[row + 1] : 0);
        }
        scheduler.update(tokens);
    }
    const double match_rate = double(matches) / double(positions);
    const double mean_abs = abs_error / double(positions);
    std::printf(
        "paged teacher-forced argmax %zu/%zu (%.2f%%), mean |dlogit| %.4f, up to %d running\n",
        matches, positions, 100.0 * match_rate, mean_abs, max_running);
    EXPECT_GE(match_rate, kArgmaxMatch);
    EXPECT_LT(mean_abs, kMeanAbsLogitError);
}

TEST_F(PagedRunnerTest, StagingAndRunningADecodeBatchIsBitIdenticalToForward) {
    constexpr int kSeqs = 3;
    KvCache cache(w().config, 64, kBlockSize);
    PagedRunner runner(w().config, w().weights, cache, 64, 8, kMaxContext, stream);
    Scheduler scheduler(SchedulerConfig{.num_blocks = 64,
                                        .block_size = kBlockSize,
                                        .max_batch = 8,
                                        .max_prefill_tokens = kMaxContext,
                                        .max_context = kMaxContext});
    for (int i = 0; i < kSeqs; ++i) {
        Request r{.id = i, .max_new_tokens = 4};
        r.prompt = prompt_of(i, 5 + 3 * i, int(vocab()));
        scheduler.submit(r);
    }
    const auto prefill = scheduler.schedule();
    const auto logits = to_host(
        runner.forward(engine::build_paged_batch(prefill, scheduler, int(runner.max_blocks()))));
    scheduler.update(argmax_rows(logits, vocab()));

    const auto step = scheduler.schedule();
    ASSERT_FALSE(step.prefill);
    const auto batch = engine::build_paged_batch(step, scheduler, int(runner.max_blocks()));
    const auto eager = to_host(runner.forward(batch));
    runner.stage(batch);
    const auto staged = to_host(runner.run_staged(kSeqs, kSeqs));
    EXPECT_EQ(staged, eager);
}

TEST_F(PagedRunnerTest, RunStagedRejectsInconsistentCounts) {
    KvCache cache(w().config, 16, kBlockSize);
    PagedRunner runner(w().config, w().weights, cache, 16, 4, kMaxContext, stream);
    EXPECT_THROW(runner.run_staged(0, 0), std::invalid_argument);
    EXPECT_THROW(runner.run_staged(17, 4), std::invalid_argument);
    EXPECT_THROW(runner.run_staged(4, 5), std::invalid_argument);
    EXPECT_THROW(runner.run_staged(4, 2), std::invalid_argument);
    const std::vector<int32_t> rows = {0, 3};
    EXPECT_THROW(runner.run_staged(4, 3, rows), std::invalid_argument);
    const std::vector<int32_t> outside = {0, 4};
    EXPECT_THROW(runner.run_staged(4, 2, outside), std::invalid_argument);
}
