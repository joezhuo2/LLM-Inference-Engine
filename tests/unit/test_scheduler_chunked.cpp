#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <random>
#include <set>
#include <span>
#include <stdexcept>
#include <vector>

#include "engine/sched/scheduler.h"

using engine::ChunkedStep;
using engine::Clock;
using engine::PrefillChunk;
using engine::Request;
using engine::RequestState;
using engine::Scheduler;
using engine::SchedulerConfig;
using engine::SeqId;

namespace {

constexpr int32_t kEos = 2;
constexpr int kBlock = 16;

Request make(SeqId id, int prompt_len, int max_new_tokens, uint64_t seed = 0) {
    Request r{.id = id, .max_new_tokens = max_new_tokens, .params = {0.7f, seed}};
    for (int i = 0; i < prompt_len; ++i) r.prompt.push_back(int32_t(3 + (id * 131 + i * 17) % 997));
    return r;
}

SchedulerConfig config(int num_blocks, int max_batch, int max_context = 2048, int32_t eos_id = -1) {
    return {.num_blocks = num_blocks,
            .block_size = kBlock,
            .max_batch = max_batch,
            .max_prefill_tokens = max_context,
            .max_context = max_context,
            .eos_id = eos_id};
}

int32_t model_next(std::span<const int32_t> context, uint64_t seed, uint64_t step) {
    uint64_t h = 1469598103934665603ull ^ seed;
    const auto mix = [&h](uint64_t v) {
        h ^= v;
        h *= 1099511628211ull;
    };
    for (const int32_t t : context) mix(uint64_t(t));
    mix(step + 0x9e37);
    h ^= h >> 29;
    return h % 53 == 0 ? kEos : int32_t(3 + h % 997);
}

std::vector<int32_t> solo(const Request& r) {
    std::vector<int32_t> context = r.prompt;
    std::vector<int32_t> out;
    while (true) {
        const int32_t t = model_next(context, r.params.seed, out.size());
        out.push_back(t);
        if (t == kEos || int(out.size()) == r.max_new_tokens) return out;
        context.push_back(t);
    }
}

void expect_chunks(const ChunkedStep& step, const std::vector<PrefillChunk>& want) {
    ASSERT_EQ(step.prefill.size(), want.size());
    for (size_t i = 0; i < want.size(); ++i)
        EXPECT_TRUE(step.prefill[i].id == want[i].id && step.prefill[i].first == want[i].first &&
                    step.prefill[i].count == want[i].count)
            << "chunk " << i << " is {" << step.prefill[i].id << ", " << step.prefill[i].first
            << ", " << step.prefill[i].count << "}";
}

void give(Scheduler& s, const ChunkedStep& step, int32_t token = 7) {
    const std::vector<int32_t> tokens(step.sampled.size(), token);
    s.update(tokens);
}

ChunkedStep run(Scheduler& s, int budget, int32_t token = 7) {
    const ChunkedStep step = s.schedule_chunked(budget);
    if (!step.empty()) give(s, step, token);
    s.take_finished();
    return step;
}

class Driver {
public:
    Driver(Scheduler& scheduler, const SchedulerConfig& cfg)
        : s_(scheduler), cache_(size_t(cfg.num_blocks) * size_t(cfg.block_size), -1) {}

    void track(SeqId id) { held_.insert(id); }

    void step(int budget, const SchedulerConfig& cfg) {
        std::vector<SeqId> ready;
        std::map<SeqId, int> preempted_before;
        for (const SeqId id : held_) {
            const Request& r = s_.request(id);
            preempted_before[id] = r.preemptions;
            if (r.state == RequestState::Running && s_.blocks().length(id) < r.length())
                ready.push_back(id);
        }

        const ChunkedStep step = s_.schedule_chunked(budget);
        ASSERT_LE(step.num_tokens(), budget);

        for (const SeqId id : held_) {
            const Request& r = s_.request(id);
            if (r.preemptions != preempted_before[id]) progress_.erase(id);
        }

        for (const SeqId id : ready) {
            if (s_.request(id).preemptions == preempted_before[id]) {
                ASSERT_NE(std::ranges::find(step.decode, id), step.decode.end())
                    << "seq " << id << " finished prefill but got no decode row";
            }
        }

        const auto& blocks = s_.blocks();
        std::vector<SeqId> sampled = step.decode;
        for (const SeqId id : step.decode) {
            const Request& r = s_.request(id);
            ASSERT_EQ(r.state, RequestState::Running);
            ASSERT_EQ(blocks.length(id), r.length());
            ASSERT_EQ(progress_.at(id) + 1, r.length()) << "seq " << id << " decoded early";
            cache_[size_t(blocks.slot(id, int(r.length()) - 1))] = r.token(r.length() - 1);
            progress_[id] = int(r.length());
        }
        bool unfinished = false;
        for (const PrefillChunk& chunk : step.prefill) {
            const Request& r = s_.request(chunk.id);
            ASSERT_EQ(r.state, RequestState::Running);
            ASSERT_EQ(blocks.length(chunk.id), r.length());
            ASSERT_GT(chunk.count, 0);
            const auto known = progress_.find(chunk.id);
            ASSERT_EQ(chunk.first, known == progress_.end() ? 0 : known->second)
                << "seq " << chunk.id << " chunks are not contiguous from 0";
            ASSERT_LE(chunk.first + chunk.count, r.length());
            for (int pos = chunk.first; pos < chunk.first + chunk.count; ++pos)
                cache_[size_t(blocks.slot(chunk.id, pos))] = r.token(pos);
            progress_[chunk.id] = chunk.first + chunk.count;
            if (progress_[chunk.id] == r.length())
                sampled.push_back(chunk.id);
            else
                unfinished = true;
        }
        ASSERT_EQ(step.sampled, sampled);
        if (unfinished) {
            ASSERT_EQ(step.num_tokens(), budget) << "budget left while in prefill";
        }

        expect_blocks(cfg);

        std::vector<int32_t> tokens;
        for (const SeqId id : step.sampled) {
            const Request& r = s_.request(id);
            const auto& table = blocks.table(id);
            std::vector<int32_t> context;
            for (int pos = 0; pos < blocks.length(id); ++pos)
                context.push_back(
                    cache_[size_t(table[size_t(pos / kBlock)]) * kBlock + pos % kBlock]);
            tokens.push_back(model_next(context, r.params.seed, r.output.size()));
        }
        if (!step.empty()) s_.update(tokens);
        for (Request& r : s_.take_finished()) {
            held_.erase(r.id);
            progress_.erase(r.id);
            done_[r.id] = r.output;
        }
    }

    const std::map<SeqId, std::vector<int32_t>>& done() const { return done_; }

private:
    void expect_blocks(const SchedulerConfig& cfg) {
        const auto& blocks = s_.blocks();
        int owned = 0;
        int running = 0;
        for (const SeqId id : held_) {
            if (s_.request(id).state != RequestState::Running) continue;
            ++running;
            owned += int(blocks.table(id).size());
            ASSERT_EQ(int(blocks.table(id).size()), (blocks.length(id) + kBlock - 1) / kBlock);
        }
        ASSERT_LE(running, cfg.max_batch);
        ASSERT_EQ(s_.num_running(), running);
        ASSERT_EQ(blocks.num_free() + owned, cfg.num_blocks - 1);
    }

    Scheduler& s_;
    std::vector<int32_t> cache_;
    std::set<SeqId> held_;
    std::map<SeqId, int> progress_;
    std::map<SeqId, std::vector<int32_t>> done_;
};

}  // namespace

TEST(SchedulerChunked, RejectsBudgetBelowMaxBatch) {
    Scheduler s(config(200, 4));
    s.submit(make(1, 8, 2));
    for (const int budget : {-1, 0, 3})
        EXPECT_THROW(s.schedule_chunked(budget), std::invalid_argument);
    EXPECT_EQ(s.num_waiting(), 1);
    EXPECT_EQ(s.num_running(), 0);
    EXPECT_FALSE(s.schedule_chunked(4).empty());
}

TEST(SchedulerChunked, IdleStepIsEmptyAndNotPending) {
    Scheduler s(config(200, 4));
    EXPECT_TRUE(s.schedule_chunked(32).empty());
    EXPECT_TRUE(s.schedule_chunked(32).empty());
    EXPECT_THROW(s.update({}), std::invalid_argument);
}

TEST(SchedulerChunked, RejectsSchedulingWhileAStepIsPending) {
    Scheduler s(config(200, 4));
    s.submit(make(1, 8, 2));
    const ChunkedStep step = s.schedule_chunked(32);
    EXPECT_THROW(s.schedule_chunked(32), std::invalid_argument);
    give(s, step);
    EXPECT_FALSE(s.schedule_chunked(32).empty());
}

TEST(SchedulerChunked, ChunkOnlyStepIsPendingUntilAnEmptyUpdate) {
    Scheduler s(config(200, 4));
    s.submit(make(1, 100, 2));
    const ChunkedStep step = s.schedule_chunked(32);
    EXPECT_TRUE(step.sampled.empty());
    EXPECT_THROW(s.schedule_chunked(32), std::invalid_argument);
    EXPECT_THROW(s.cancel(1), std::invalid_argument);
    const std::vector<int32_t> wrong{7};
    EXPECT_THROW(s.update(wrong), std::invalid_argument);
    s.update({});
    EXPECT_THROW(s.update({}), std::invalid_argument);
    expect_chunks(s.schedule_chunked(32), {{1, 32, 32}});
}

TEST(SchedulerChunked, SpreadsALongPromptOverSteps) {
    Scheduler s(config(200, 4));
    s.submit(make(1, 100, 5));
    for (const int first : {0, 32, 64}) {
        const ChunkedStep step = run(s, 32);
        EXPECT_TRUE(step.decode.empty());
        EXPECT_TRUE(step.sampled.empty());
        expect_chunks(step, {{1, first, 32}});
        EXPECT_EQ(s.request(1).output.size(), 0u);
    }
    const ChunkedStep last = run(s, 32);
    expect_chunks(last, {{1, 96, 4}});
    EXPECT_EQ(last.sampled, std::vector<SeqId>{1});
    EXPECT_EQ(s.request(1).output.size(), 1u);
    const ChunkedStep next = run(s, 32);
    EXPECT_EQ(next.decode, std::vector<SeqId>{1});
    EXPECT_TRUE(next.prefill.empty());
    EXPECT_EQ(next.num_tokens(), 1);
}

TEST(SchedulerChunked, AllocatesTheWholeLengthAtAdmission) {
    Scheduler s(config(200, 4));
    s.submit(make(1, 100, 5));
    const int free_before = s.blocks().num_free();
    const ChunkedStep step = s.schedule_chunked(10);
    EXPECT_EQ(s.request(1).state, RequestState::Running);
    EXPECT_EQ(s.blocks().length(1), 100);
    EXPECT_EQ(s.blocks().num_free(), free_before - 7);
    give(s, step);
    EXPECT_EQ(s.blocks().num_free(), free_before - 7);
    EXPECT_EQ(s.num_running(), 1);
    EXPECT_EQ(s.num_waiting(), 0);
}

TEST(SchedulerChunked, PromptEqualToTheBudgetFinishesInOneStep) {
    Scheduler s(config(200, 4));
    s.submit(make(1, 32, 5));
    const ChunkedStep first = run(s, 32);
    expect_chunks(first, {{1, 0, 32}});
    EXPECT_EQ(first.sampled, std::vector<SeqId>{1});
    EXPECT_EQ(run(s, 32).decode, std::vector<SeqId>{1});
}

TEST(SchedulerChunked, DecodesComeFirstAndPrefillTakesTheRest) {
    Scheduler s(config(200, 4));
    s.submit(make(1, 4, 50));
    s.submit(make(2, 4, 50));
    run(s, 32);
    s.submit(make(3, 100, 5));
    const ChunkedStep step = run(s, 32);
    EXPECT_EQ(step.decode, (std::vector<SeqId>{1, 2}));
    expect_chunks(step, {{3, 0, 30}});
    EXPECT_EQ(step.sampled, (std::vector<SeqId>{1, 2}));
    EXPECT_EQ(step.num_tokens(), 32);
    const ChunkedStep next = run(s, 32);
    EXPECT_EQ(next.decode, (std::vector<SeqId>{1, 2}));
    expect_chunks(next, {{3, 30, 30}});
}

TEST(SchedulerChunked, OneStepCanFinishOneSequenceAndStartTheNext) {
    Scheduler s(config(200, 4));
    s.submit(make(1, 10, 5));
    s.submit(make(2, 50, 5));
    const ChunkedStep step = run(s, 32);
    expect_chunks(step, {{1, 0, 10}, {2, 0, 22}});
    EXPECT_EQ(step.sampled, std::vector<SeqId>{1});
    EXPECT_EQ(step.num_tokens(), 32);
}

TEST(SchedulerChunked, ContinuesOlderPrefillBeforeAdmitting) {
    Scheduler s(config(200, 4));
    s.submit(make(1, 50, 5));
    run(s, 32);
    s.submit(make(2, 50, 5));
    const ChunkedStep second = run(s, 32);
    expect_chunks(second, {{1, 32, 18}, {2, 0, 14}});
    EXPECT_EQ(second.decode, std::vector<SeqId>{});
    EXPECT_EQ(second.sampled, std::vector<SeqId>{1});
}

TEST(SchedulerChunked, SampledListsDecodesThenFinishedChunks) {
    Scheduler s(config(200, 4));
    s.submit(make(1, 4, 50));
    run(s, 16);
    s.submit(make(2, 3, 50));
    s.submit(make(3, 5, 50));
    s.submit(make(4, 40, 50));
    const ChunkedStep step = run(s, 16);
    EXPECT_EQ(step.decode, std::vector<SeqId>{1});
    expect_chunks(step, {{2, 0, 3}, {3, 0, 5}, {4, 0, 7}});
    EXPECT_EQ(step.sampled, (std::vector<SeqId>{1, 2, 3}));
    EXPECT_EQ(s.request(4).output.size(), 0u);
}

TEST(SchedulerChunked, OnePromptTokenLeftIsStillAChunk) {
    Scheduler s(config(200, 4));
    s.submit(make(1, 33, 5));
    run(s, 32);
    const int length = s.blocks().length(1);
    const ChunkedStep step = run(s, 32);
    EXPECT_TRUE(step.decode.empty());
    expect_chunks(step, {{1, 32, 1}});
    EXPECT_EQ(step.sampled, std::vector<SeqId>{1});
    EXPECT_EQ(s.blocks().length(1), length);
}

TEST(SchedulerChunked, StopsAdmittingAtMaxBatch) {
    Scheduler s(config(200, 2));
    s.submit(make(1, 3, 50));
    s.submit(make(2, 3, 50));
    s.submit(make(3, 3, 50));
    const ChunkedStep step = run(s, 32);
    expect_chunks(step, {{1, 0, 3}, {2, 0, 3}});
    EXPECT_EQ(s.num_running(), 2);
    EXPECT_EQ(s.num_waiting(), 1);
    EXPECT_EQ(s.request(3).state, RequestState::Waiting);
}

TEST(SchedulerChunked, BudgetOfMaxBatchStillAdvancesPrefill) {
    Scheduler s(config(200, 4));
    for (const SeqId id : {1, 2, 3}) s.submit(make(id, 1, 50));
    run(s, 4);
    s.submit(make(4, 3, 50));
    for (const int first : {0, 1, 2}) {
        const ChunkedStep step = run(s, 4);
        EXPECT_EQ(step.decode, (std::vector<SeqId>{1, 2, 3}));
        expect_chunks(step, {{4, first, 1}});
    }
    EXPECT_EQ(s.request(4).output.size(), 1u);
}

TEST(SchedulerChunked, NeverSkipsTheHeadOfTheQueue) {
    Scheduler s(config(21, 4, 256));
    s.submit(make(1, 200, 50));
    s.submit(make(2, 200, 50));
    s.submit(make(3, 2, 50));
    const ChunkedStep step = run(s, 32);
    expect_chunks(step, {{1, 0, 32}});
    EXPECT_EQ(s.num_waiting(), 2);
    EXPECT_EQ(s.request(2).state, RequestState::Waiting);
    EXPECT_EQ(s.request(3).state, RequestState::Waiting);
    expect_chunks(run(s, 512), {{1, 32, 168}});
    EXPECT_EQ(s.num_waiting(), 2);
}

TEST(SchedulerChunked, IgnoresMaxPrefillTokens) {
    SchedulerConfig c = config(200, 4, 256);
    c.max_prefill_tokens = 256;
    Scheduler s(c);
    for (const SeqId id : {1, 2, 3}) s.submit(make(id, 200, 5));
    const ChunkedStep step = run(s, 512);
    expect_chunks(step, {{1, 0, 200}, {2, 0, 200}, {3, 0, 112}});
}

TEST(SchedulerChunked, StampsFirstTokenOnlyWhenItArrives) {
    Scheduler s(config(200, 4));
    s.submit(make(1, 100, 5));
    run(s, 32);
    EXPECT_EQ(s.request(1).first_token, Clock::time_point{});
    run(s, 32);
    run(s, 32);
    EXPECT_EQ(s.request(1).first_token, Clock::time_point{});
    run(s, 32);
    EXPECT_NE(s.request(1).first_token, Clock::time_point{});
}

TEST(SchedulerChunked, OneNewTokenFinishesOnTheLastChunk) {
    Scheduler s(config(200, 4));
    const int all_free = s.blocks().num_free();
    s.submit(make(1, 40, 1));
    ChunkedStep step = s.schedule_chunked(32);
    give(s, step);
    EXPECT_TRUE(s.take_finished().empty());
    EXPECT_LT(s.blocks().num_free(), all_free);
    step = s.schedule_chunked(32);
    give(s, step);
    const std::vector<Request> finished = s.take_finished();
    ASSERT_EQ(finished.size(), 1u);
    EXPECT_EQ(finished[0].output, std::vector<int32_t>{7});
    EXPECT_EQ(s.blocks().num_free(), all_free);
    EXPECT_EQ(s.num_running(), 0);
    EXPECT_TRUE(s.schedule_chunked(32).empty());
}

TEST(SchedulerChunked, PreemptsPartlyPrefilledSequenceFirstAndRestartsIt) {
    Scheduler s(config(17, 4, 256));
    s.submit(make(1, kBlock, 100));
    run(s, 32);
    s.submit(make(2, 14 * kBlock, 5));
    const ChunkedStep chunked = run(s, 4);
    EXPECT_EQ(chunked.decode, std::vector<SeqId>{1});
    expect_chunks(chunked, {{2, 0, 3}});
    EXPECT_EQ(s.blocks().num_free(), 0);
    int guard = 0;
    ChunkedStep step;
    do {
        step = run(s, 4);
        ASSERT_LT(++guard, 100);
    } while (s.request(2).preemptions == 0);
    EXPECT_EQ(s.request(2).state, RequestState::Waiting);
    EXPECT_EQ(s.num_waiting(), 1);
    EXPECT_EQ(s.request(2).output.size(), 0u);
    EXPECT_EQ(step.decode, std::vector<SeqId>{1});
    EXPECT_TRUE(step.prefill.empty());
    EXPECT_EQ(s.request(1).preemptions, 0);
    s.cancel(1);
    expect_chunks(run(s, 4), {{2, 0, 4}});
}

TEST(SchedulerChunked, PreemptedDecoderReplaysPromptAndOutput) {
    Scheduler s(config(5, 4, 64));
    s.submit(make(1, 8, 50));
    s.submit(make(2, 16, 40));
    run(s, 32);
    int guard = 0;
    while (s.request(2).preemptions == 0) {
        run(s, 32);
        ASSERT_LT(++guard, 100);
    }
    const int64_t length = s.request(2).length();
    EXPECT_GT(s.request(2).output.size(), 0u);
    s.cancel(1);
    const ChunkedStep step = s.schedule_chunked(4);
    expect_chunks(step, {{2, 0, 4}});
    EXPECT_TRUE(step.sampled.empty());
    EXPECT_EQ(s.blocks().length(2), length);
    s.update({});
    int first = 4;
    while (first + 4 < length) {
        const ChunkedStep middle = s.schedule_chunked(4);
        expect_chunks(middle, {{2, first, 4}});
        s.update({});
        first += 4;
    }
    const ChunkedStep last = s.schedule_chunked(4);
    expect_chunks(last, {{2, first, int(length) - first}});
    EXPECT_EQ(last.sampled, std::vector<SeqId>{2});
}

TEST(SchedulerChunked, CancelInPrefillFreesBlocksAndForgetsProgress) {
    Scheduler s(config(200, 4));
    const int all_free = s.blocks().num_free();
    s.submit(make(1, 100, 5));
    run(s, 32);
    run(s, 32);
    s.cancel(1);
    EXPECT_EQ(s.blocks().num_free(), all_free);
    EXPECT_TRUE(s.schedule_chunked(32).empty());
    s.submit(make(1, 100, 5));
    expect_chunks(run(s, 32), {{1, 0, 32}});
}

TEST(SchedulerChunked, IdReusedAfterFinishingStartsAtZero) {
    Scheduler s(config(200, 4));
    s.submit(make(1, 40, 1));
    run(s, 32);
    run(s, 32);
    EXPECT_EQ(s.num_running(), 0);
    s.submit(make(1, 40, 1));
    expect_chunks(run(s, 32), {{1, 0, 32}});
}

TEST(SchedulerChunked, RunningListKeepsItsOrderAcrossChunks) {
    Scheduler s(config(200, 8));
    s.submit(make(1, 20, 50));
    s.submit(make(2, 20, 50));
    s.submit(make(3, 20, 50));
    EXPECT_EQ(run(s, 8).prefill.size(), 1u);
    EXPECT_EQ(run(s, 8).prefill.size(), 1u);
    const ChunkedStep third = run(s, 8);
    expect_chunks(third, {{1, 16, 4}, {2, 0, 4}});
    expect_chunks(run(s, 8), {{2, 4, 7}});
    const ChunkedStep later = run(s, 8);
    EXPECT_EQ(later.decode, std::vector<SeqId>{1});
    expect_chunks(later, {{2, 11, 7}});
    const ChunkedStep end = run(s, 8);
    EXPECT_EQ(end.decode, std::vector<SeqId>{1});
    expect_chunks(end, {{2, 18, 2}, {3, 0, 5}});
    EXPECT_EQ(run(s, 8).decode, (std::vector<SeqId>{1, 2}));
}

TEST(SchedulerChunked, AnswersMatchAloneForAnyBudgetOnALargePool) {
    const SchedulerConfig c = config(400, 8, 2048, kEos);
    std::vector<Request> requests;
    const int prompts[8] = {1, 17, 500, 33, 900, 5, 120, 16};
    const int fresh[8] = {40, 1, 56, 20, 64, 3, 30, 60};
    for (int i = 0; i < 8; ++i) requests.push_back(make(100 + i, prompts[i], fresh[i], i % 3));
    for (const int budget : {8, 9, 32, 256, 4096}) {
        Scheduler s(c);
        Driver d(s, c);
        for (const Request& r : requests) {
            s.submit(r);
            d.track(r.id);
        }
        for (int i = 0; i < 100000 && s.num_waiting() + s.num_running() > 0; ++i) {
            d.step(budget, c);
            if (HasFatalFailure()) return;
        }
        ASSERT_EQ(d.done().size(), requests.size()) << "budget " << budget;
        for (const Request& r : requests)
            EXPECT_EQ(d.done().at(r.id), solo(r)) << "budget " << budget;
        EXPECT_EQ(s.blocks().num_free(), c.num_blocks - 1);
    }
}

TEST(SchedulerChunked, RandomRunsWithPreemptionMatchAlone) {
    int preempted = 0;
    for (int seed = 0; seed < 40; ++seed) {
        std::mt19937 rng(uint32_t(seed) + 11);
        const auto pick = [&rng](int lo, int hi) {
            return std::uniform_int_distribution<int>(lo, hi)(rng);
        };
        const SchedulerConfig c = config(pick(18, 50), pick(1, 8), 256, kEos);
        const int budget = c.max_batch + pick(0, 64);
        std::vector<Request> requests;
        for (int i = 0; i < 16; ++i) {
            const int prompt = pick(1, 180);
            requests.push_back(make(i, prompt, pick(1, 256 - prompt), uint64_t(pick(0, 4))));
        }
        Scheduler s(c);
        Driver d(s, c);
        size_t next = 0;
        std::map<SeqId, int> most;
        for (int i = 0; i < 200000; ++i) {
            if (next < requests.size() && pick(0, 3) == 0) {
                d.track(requests[next].id);
                s.submit(requests[next++]);
            }
            if (next == requests.size() && s.num_waiting() + s.num_running() == 0) break;
            d.step(budget, c);
            if (HasFatalFailure()) return;
            for (const Request& r : requests)
                if (!d.done().contains(r.id) && r.id < SeqId(next))
                    most[r.id] = std::max(most[r.id], s.request(r.id).preemptions);
        }
        ASSERT_EQ(d.done().size(), requests.size()) << "seed " << seed;
        for (const Request& r : requests) EXPECT_EQ(d.done().at(r.id), solo(r)) << "seed " << seed;
        EXPECT_EQ(s.blocks().num_free(), c.num_blocks - 1) << "seed " << seed;
        for (const auto& [id, count] : most) preempted += count;
    }
    EXPECT_GT(preempted, 0);
}
