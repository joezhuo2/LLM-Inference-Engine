#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

#include "engine/sched/scheduler.h"

using engine::Clock;
using engine::Request;
using engine::RequestState;
using engine::Scheduler;
using engine::SchedulerConfig;
using engine::SeqId;
using engine::Step;

namespace {

constexpr int32_t kEos = 2;

Request make(SeqId id, int prompt_len, int max_new_tokens, uint64_t seed = 0) {
    Request r{.id = id, .max_new_tokens = max_new_tokens, .params = {0.7f, seed}};
    for (int i = 0; i < prompt_len; ++i) r.prompt.push_back(int32_t(3 + (id * 131 + i * 17) % 997));
    return r;
}

void expect_step(const Step& step, bool prefill, const std::vector<SeqId>& seqs) {
    EXPECT_EQ(step.prefill, prefill);
    EXPECT_EQ(step.seqs, seqs);
}

void give(Scheduler& s, std::vector<int32_t> tokens) {
    s.update(tokens);
}

int blocks_for(int64_t tokens, int block_size) {
    return int((tokens + block_size - 1) / block_size);
}

struct FakeModel {
    int32_t next(std::span<const int32_t> context, uint64_t seed, uint64_t step) const {
        uint64_t h = 1469598103934665603ull ^ seed;
        auto mix = [&h](uint64_t v) {
            h ^= v;
            h *= 1099511628211ull;
        };
        for (const int32_t t : context) mix(uint64_t(t));
        mix(step + 0x9e37);
        h ^= h >> 29;
        return h % 53 == 0 ? kEos : int32_t(3 + h % 997);
    }

    std::vector<int32_t> solo(const Request& r) const {
        std::vector<int32_t> context = r.prompt;
        std::vector<int32_t> out;
        while (true) {
            const int32_t t = next(context, r.params.seed, out.size());
            out.push_back(t);
            if (t == kEos || int(out.size()) == r.max_new_tokens) return out;
            context.push_back(t);
        }
    }
};

class FakeRunner {
public:
    explicit FakeRunner(const SchedulerConfig& config)
        : block_size_(config.block_size),
          cache_(size_t(config.num_blocks) * size_t(config.block_size), -1) {}

    std::vector<int32_t> run(const Scheduler& s, const Step& step) {
        const auto& blocks = s.blocks();
        for (const SeqId seq : step.seqs) {
            const Request& r = s.request(seq);
            EXPECT_EQ(r.state, RequestState::Running) << "seq " << seq;
            const int len = blocks.length(seq);
            EXPECT_EQ(len, r.length()) << "seq " << seq;
            for (int pos = step.prefill ? 0 : len - 1; pos < len; ++pos)
                cache_[size_t(blocks.slot(seq, pos))] = r.token(pos);
        }
        std::vector<int32_t> tokens;
        for (const SeqId seq : step.seqs) {
            const Request& r = s.request(seq);
            const auto& table = blocks.table(seq);
            std::vector<int32_t> context;
            for (int pos = 0; pos < blocks.length(seq); ++pos)
                context.push_back(
                    cache_[size_t(table[pos / block_size_]) * block_size_ + pos % block_size_]);
            tokens.push_back(model_.next(context, r.params.seed, r.output.size()));
        }
        return tokens;
    }

private:
    FakeModel model_;
    int block_size_;
    std::vector<int32_t> cache_;
};

void expect_invariants(const Scheduler& s, const SchedulerConfig& config,
                       const std::vector<SeqId>& held, const Step* pending) {
    const auto& blocks = s.blocks();
    int waiting = 0;
    int running = 0;
    int owned = 0;
    for (const SeqId id : held) {
        const Request& r = s.request(id);
        if (r.first_token != Clock::time_point{}) {
            ASSERT_LE(r.arrived, r.first_token);
        }
        if (r.state == RequestState::Finished) {
            ASSERT_LE(r.first_token, r.finished);
        }
        if (r.state != RequestState::Running) {
            waiting += r.state == RequestState::Waiting;
            ASSERT_THROW(blocks.length(id), std::invalid_argument) << "seq " << id;
            continue;
        }
        ++running;
        const bool fed = pending && std::ranges::find(pending->seqs, id) != pending->seqs.end();
        ASSERT_EQ(blocks.length(id), r.length() - (fed ? 0 : 1)) << "seq " << id;
        ASSERT_EQ(int(blocks.table(id).size()), blocks_for(blocks.length(id), config.block_size));
        owned += int(blocks.table(id).size());
    }
    ASSERT_EQ(s.num_waiting(), waiting);
    ASSERT_EQ(s.num_running(), running);
    ASSERT_LE(running, config.max_batch);
    ASSERT_EQ(blocks.num_free() + owned, config.num_blocks - 1);
}

std::vector<Request> sixteen_requests() {
    std::vector<Request> requests;
    const int prompts[16] = {1, 17, 200, 33, 64, 5, 120, 16, 90, 2, 48, 150, 31, 8, 77, 15};
    const int new_tokens[16] = {40, 1, 56, 20, 64, 3, 30, 60, 12, 50, 9, 70, 25, 45, 16, 33};
    for (int i = 0; i < 16; ++i)
        requests.push_back(make(100 + i, prompts[i], new_tokens[i], uint64_t(i % 5)));
    return requests;
}

struct RunResult {
    std::map<SeqId, Request> finished;
    int max_decode_batch = 0;
    int preemptions = 0;
};

RunResult run_to_completion(const SchedulerConfig& config, const std::vector<Request>& requests) {
    Scheduler s(config);
    FakeRunner runner(config);
    std::vector<SeqId> held;
    for (const Request& r : requests) {
        s.submit(r);
        held.push_back(r.id);
    }
    RunResult result;
    for (int steps = 0; s.num_waiting() + s.num_running() > 0; ++steps) {
        if (steps > 100000) {
            ADD_FAILURE() << "no progress";
            break;
        }
        const Step step = s.schedule();
        if (step.seqs.empty()) {
            ADD_FAILURE() << "empty step with work left";
            break;
        }
        if (!step.prefill)
            result.max_decode_batch = std::max(result.max_decode_batch, int(step.seqs.size()));
        give(s, runner.run(s, step));
        expect_invariants(s, config, held, nullptr);
        for (Request& r : s.take_finished()) {
            std::erase(held, r.id);
            result.preemptions += r.preemptions;
            result.finished[r.id] = std::move(r);
        }
    }
    return result;
}

}  // namespace

TEST(Scheduler, RejectsBadConfigs) {
    const SchedulerConfig ok{
        .num_blocks = 5, .block_size = 4, .max_prefill_tokens = 16, .max_context = 16};
    EXPECT_NO_THROW(Scheduler{ok});
    auto with = [&ok](auto change) {
        SchedulerConfig c = ok;
        change(c);
        return c;
    };
    EXPECT_THROW(Scheduler{SchedulerConfig{}}, std::invalid_argument);
    EXPECT_THROW(Scheduler{with([](auto& c) { c.num_blocks = 1; })}, std::invalid_argument);
    EXPECT_THROW(Scheduler{with([](auto& c) { c.block_size = 0; })}, std::invalid_argument);
    EXPECT_THROW(Scheduler{with([](auto& c) { c.max_batch = 0; })}, std::invalid_argument);
    EXPECT_THROW(Scheduler{with([](auto& c) { c.max_context = 0; })}, std::invalid_argument);
    EXPECT_THROW(Scheduler{with([](auto& c) { c.max_prefill_tokens = 15; })},
                 std::invalid_argument);
    EXPECT_THROW(Scheduler{with([](auto& c) { c.max_context = c.max_prefill_tokens = 17; })},
                 std::invalid_argument);
    EXPECT_NO_THROW(Scheduler{with([](auto& c) { c.max_prefill_tokens = 1000; })});
    EXPECT_NO_THROW(Scheduler{with([](auto& c) { c.max_batch = 1; })});
}

TEST(Scheduler, StartsIdleWithEveryBlockFree) {
    const SchedulerConfig config{
        .num_blocks = 9, .block_size = 4, .max_prefill_tokens = 32, .max_context = 32};
    Scheduler s(config);
    EXPECT_EQ(s.num_waiting(), 0);
    EXPECT_EQ(s.num_running(), 0);
    EXPECT_EQ(s.blocks().num_free(), 8);
    expect_step(s.schedule(), false, {});
    expect_step(s.schedule(), false, {});
    EXPECT_THROW(give(s, {}), std::invalid_argument);
    EXPECT_TRUE(s.take_finished().empty());
}

TEST(Scheduler, SubmitStampsAndResetsTheRequest) {
    Scheduler s({.num_blocks = 9, .block_size = 4, .max_prefill_tokens = 32, .max_context = 32});
    Request r = make(7, 5, 3);
    r.state = RequestState::Finished;
    r.preemptions = 4;
    const auto before = Clock::now();
    s.submit(r);
    const auto after = Clock::now();
    const Request& held = s.request(7);
    EXPECT_EQ(held.state, RequestState::Waiting);
    EXPECT_EQ(held.preemptions, 0);
    EXPECT_GE(held.arrived, before);
    EXPECT_LE(held.arrived, after);
    EXPECT_EQ(held.prompt, r.prompt);
    EXPECT_EQ(held.max_new_tokens, 3);
    EXPECT_EQ(held.params.seed, r.params.seed);
    EXPECT_EQ(s.num_waiting(), 1);
    EXPECT_EQ(s.num_running(), 0);
}

TEST(Scheduler, SubmitRejectsInvalidAndHeldIds) {
    Scheduler s({.num_blocks = 17, .block_size = 4, .max_prefill_tokens = 32, .max_context = 32});
    EXPECT_THROW(s.submit(make(1, 0, 3)), std::invalid_argument);
    EXPECT_THROW(s.submit(make(1, 30, 4)), std::invalid_argument);
    EXPECT_THROW(s.submit(make(1, 33, 1)), std::invalid_argument);
    Request bad = make(1, 4, 3);
    bad.output = {9};
    EXPECT_THROW(s.submit(bad), std::invalid_argument);
    EXPECT_EQ(s.num_waiting(), 0);
    EXPECT_THROW(s.request(1), std::invalid_argument);

    s.submit(make(1, 30, 3));
    s.submit(make(2, 32, 1));
    EXPECT_THROW(s.submit(make(1, 2, 2)), std::invalid_argument);
    EXPECT_EQ(s.num_waiting(), 2);
    EXPECT_EQ(s.request(1).prompt.size(), 30u);

    expect_step(s.schedule(), true, {1});
    EXPECT_THROW(s.submit(make(1, 2, 2)), std::invalid_argument);
    give(s, {5});
    expect_step(s.schedule(), true, {2});
    give(s, {6});
    EXPECT_EQ(s.request(2).state, RequestState::Finished);
    EXPECT_THROW(s.submit(make(2, 2, 2)), std::invalid_argument);
    EXPECT_EQ(s.num_waiting(), 0);
    EXPECT_EQ(s.num_running(), 1);

    const auto finished = s.take_finished();
    ASSERT_EQ(finished.size(), 1u);
    EXPECT_EQ(finished[0].id, 2);
    EXPECT_THROW(s.request(2), std::invalid_argument);
    EXPECT_NO_THROW(s.submit(make(2, 2, 2)));
    EXPECT_EQ(s.num_waiting(), 1);
}

TEST(Scheduler, AdmitsFirstComeFirstServedUpToMaxBatch) {
    Scheduler s({.num_blocks = 33,
                 .block_size = 4,
                 .max_batch = 3,
                 .max_prefill_tokens = 32,
                 .max_context = 32});
    for (SeqId id = 0; id < 5; ++id) s.submit(make(id, 2 + int(id), 10));
    expect_step(s.schedule(), true, {0, 1, 2});
    EXPECT_EQ(s.num_running(), 3);
    EXPECT_EQ(s.num_waiting(), 2);
    for (SeqId id = 0; id < 3; ++id) {
        EXPECT_EQ(s.request(id).state, RequestState::Running);
        EXPECT_EQ(s.blocks().length(id), 2 + id);
    }
    EXPECT_EQ(s.request(3).state, RequestState::Waiting);
    EXPECT_THROW(s.blocks().length(3), std::invalid_argument);
    give(s, {10, 11, 12});
    for (SeqId id = 0; id < 3; ++id) EXPECT_EQ(s.blocks().length(id), 2 + id);
    EXPECT_EQ(s.request(1).output, (std::vector<int32_t>{11}));

    expect_step(s.schedule(), false, {0, 1, 2});
    for (SeqId id = 0; id < 3; ++id) EXPECT_EQ(s.blocks().length(id), 3 + id);
    give(s, {20, 21, 22});
    EXPECT_EQ(s.request(2).output, (std::vector<int32_t>{12, 22}));
    EXPECT_EQ(s.num_waiting(), 2);
}

TEST(Scheduler, PrefillStaysWithinTheTokenBudget) {
    const SchedulerConfig config{
        .num_blocks = 65, .block_size = 4, .max_prefill_tokens = 32, .max_context = 32};
    {
        Scheduler s(config);
        s.submit(make(0, 20, 5));
        s.submit(make(1, 12, 5));
        s.submit(make(2, 1, 5));
        expect_step(s.schedule(), true, {0, 1});
        give(s, {5, 6});
        expect_step(s.schedule(), true, {2});
        give(s, {7});
        expect_step(s.schedule(), false, {0, 1, 2});
    }
    {
        Scheduler s(config);
        s.submit(make(0, 20, 5));
        s.submit(make(1, 13, 5));
        s.submit(make(2, 1, 5));
        expect_step(s.schedule(), true, {0});
        give(s, {5});
        expect_step(s.schedule(), true, {1, 2});
        give(s, {6, 7});
        expect_step(s.schedule(), false, {0, 1, 2});
    }
}

TEST(Scheduler, AdmissionStopsAtTheFirstRequestThatDoesNotFit) {
    const SchedulerConfig config{
        .num_blocks = 9, .block_size = 4, .max_prefill_tokens = 32, .max_context = 32};
    Scheduler s(config);
    s.submit(make(0, 16, 12));
    s.submit(make(1, 20, 5));
    s.submit(make(2, 1, 5));
    expect_step(s.schedule(), true, {0});
    EXPECT_EQ(s.blocks().num_free(), 4);
    give(s, {5});
    expect_step(s.schedule(), false, {0});
    EXPECT_EQ(s.blocks().num_free(), 3);
    EXPECT_EQ(s.num_waiting(), 2);
    give(s, {6});
    expect_step(s.schedule(), false, {0});
    give(s, {7});
    EXPECT_EQ(s.request(2).state, RequestState::Waiting);
}

TEST(Scheduler, AdmittingExactlyTheFreeBlocksFillsTheCache) {
    Scheduler s({.num_blocks = 5, .block_size = 4, .max_prefill_tokens = 16, .max_context = 16});
    s.submit(make(0, 7, 2));
    s.submit(make(1, 8, 2));
    s.submit(make(2, 1, 2));
    expect_step(s.schedule(), true, {0, 1});
    EXPECT_EQ(s.blocks().num_free(), 0);
    EXPECT_EQ(s.num_waiting(), 1);
}

TEST(Scheduler, PrefillGoesBeforeDecode) {
    Scheduler s({.num_blocks = 33, .block_size = 4, .max_prefill_tokens = 32, .max_context = 32});
    s.submit(make(0, 5, 10));
    expect_step(s.schedule(), true, {0});
    give(s, {5});
    expect_step(s.schedule(), false, {0});
    give(s, {6});
    s.submit(make(1, 3, 10));
    expect_step(s.schedule(), true, {1});
    EXPECT_EQ(s.blocks().length(0), 6);
    EXPECT_EQ(s.blocks().length(1), 3);
    give(s, {7});
    EXPECT_EQ(s.blocks().length(0), 6);
    expect_step(s.schedule(), false, {0, 1});
    EXPECT_EQ(s.blocks().length(0), 7);
    EXPECT_EQ(s.blocks().length(1), 4);
}

TEST(Scheduler, RetiresAtMaxNewTokensAndFreesTheBlocks) {
    Scheduler s({.num_blocks = 9,
                 .block_size = 4,
                 .max_prefill_tokens = 32,
                 .max_context = 32,
                 .eos_id = kEos});
    s.submit(make(0, 6, 3));
    expect_step(s.schedule(), true, {0});
    give(s, {5});
    expect_step(s.schedule(), false, {0});
    give(s, {6});
    expect_step(s.schedule(), false, {0});
    EXPECT_EQ(s.blocks().num_free(), 6);
    give(s, {7});
    EXPECT_EQ(s.blocks().num_free(), 8);
    EXPECT_THROW(s.blocks().length(0), std::invalid_argument);
    EXPECT_EQ(s.num_running(), 0);
    const Request& r = s.request(0);
    EXPECT_EQ(r.state, RequestState::Finished);
    EXPECT_EQ(r.output, (std::vector<int32_t>{5, 6, 7}));
    EXPECT_LE(r.arrived, r.first_token);
    EXPECT_LE(r.first_token, r.finished);
    expect_step(s.schedule(), false, {});
    EXPECT_THROW(give(s, {}), std::invalid_argument);
    const auto finished = s.take_finished();
    ASSERT_EQ(finished.size(), 1u);
    EXPECT_EQ(finished[0].output, (std::vector<int32_t>{5, 6, 7}));
    EXPECT_EQ(finished[0].state, RequestState::Finished);
    EXPECT_TRUE(s.take_finished().empty());
}

TEST(Scheduler, RetiresOnEosAndKeepsIt) {
    Scheduler s({.num_blocks = 9,
                 .block_size = 4,
                 .max_prefill_tokens = 32,
                 .max_context = 32,
                 .eos_id = kEos});
    s.submit(make(0, 4, 10));
    s.submit(make(1, 4, 10));
    s.submit(make(2, 4, 10));
    expect_step(s.schedule(), true, {0, 1, 2});
    give(s, {5, kEos, 6});
    EXPECT_EQ(s.request(1).state, RequestState::Finished);
    EXPECT_EQ(s.request(1).output, (std::vector<int32_t>{kEos}));
    expect_step(s.schedule(), false, {0, 2});
    give(s, {kEos, 7});
    expect_step(s.schedule(), false, {2});
    give(s, {kEos});
    EXPECT_EQ(s.blocks().num_free(), 8);
    const auto finished = s.take_finished();
    ASSERT_EQ(finished.size(), 3u);
    EXPECT_EQ(finished[0].id, 1);
    EXPECT_EQ(finished[1].id, 0);
    EXPECT_EQ(finished[2].id, 2);
    EXPECT_EQ(finished[1].output, (std::vector<int32_t>{5, kEos}));
    EXPECT_EQ(finished[2].output, (std::vector<int32_t>{6, 7, kEos}));
}

TEST(Scheduler, FinishedRequestsComeOutInFinishingOrder) {
    Scheduler s({.num_blocks = 9, .block_size = 4, .max_prefill_tokens = 32, .max_context = 32});
    s.submit(make(0, 2, 3));
    s.submit(make(1, 2, 1));
    s.submit(make(2, 2, 2));
    s.submit(make(3, 2, 1));
    expect_step(s.schedule(), true, {0, 1, 2, 3});
    give(s, {5, 5, 5, 5});
    expect_step(s.schedule(), false, {0, 2});
    give(s, {6, 6});
    std::vector<SeqId> order;
    for (const Request& r : s.take_finished()) order.push_back(r.id);
    EXPECT_EQ(order, (std::vector<SeqId>{1, 3, 2}));
}

TEST(Scheduler, MaxNewTokensOfOneNeverDecodes) {
    Scheduler s({.num_blocks = 9, .block_size = 4, .max_prefill_tokens = 32, .max_context = 32});
    s.submit(make(0, 32, 1));
    expect_step(s.schedule(), true, {0});
    EXPECT_EQ(s.blocks().num_free(), 0);
    give(s, {9});
    EXPECT_EQ(s.request(0).state, RequestState::Finished);
    EXPECT_EQ(s.blocks().num_free(), 8);
    expect_step(s.schedule(), false, {});
}

TEST(Scheduler, PreemptsTheMostRecentlyAdmittedAndRecomputesIt) {
    const SchedulerConfig config{
        .num_blocks = 5, .block_size = 4, .max_prefill_tokens = 16, .max_context = 16};
    Scheduler s(config);
    s.submit(make(10, 4, 8));
    s.submit(make(11, 4, 8));
    s.submit(make(12, 8, 4));
    std::vector<SeqId> held = {10, 11, 12};

    expect_step(s.schedule(), true, {10, 11, 12});
    EXPECT_EQ(s.blocks().num_free(), 0);
    give(s, {20, 21, 22});
    const auto first_token_11 = s.request(11).first_token;
    const auto first_token_12 = s.request(12).first_token;

    const Step second = s.schedule();
    expect_step(second, false, {10, 11});
    const Request& c = s.request(12);
    EXPECT_EQ(c.state, RequestState::Waiting);
    EXPECT_EQ(c.preemptions, 1);
    EXPECT_EQ(c.output, (std::vector<int32_t>{22}));
    EXPECT_EQ(s.num_waiting(), 1);
    EXPECT_THROW(s.blocks().length(12), std::invalid_argument);
    EXPECT_EQ(s.blocks().num_free(), 0);
    expect_invariants(s, config, held, &second);
    give(s, {30, 31});

    int32_t next = 40;
    std::vector<Step> steps;
    while (s.num_waiting() + s.num_running() > 0 && steps.size() < 40) {
        const Step step = s.schedule();
        steps.push_back(step);
        expect_invariants(s, config, held, &step);
        if (step.prefill && step.seqs == std::vector<SeqId>{11}) {
            EXPECT_EQ(s.blocks().length(11), 9);
            EXPECT_EQ(s.request(11).output.size(), 5u);
        }
        if (step.prefill && step.seqs == std::vector<SeqId>{12}) {
            EXPECT_EQ(s.blocks().length(12), 9);
            EXPECT_EQ(s.request(12).output, (std::vector<int32_t>{22}));
        }
        give(s, std::vector<int32_t>(step.seqs.size(), next++));
        expect_invariants(s, config, held, nullptr);
    }
    const std::vector<std::pair<bool, std::vector<SeqId>>> expected = {
        {false, {10, 11}}, {false, {10, 11}}, {false, {10, 11}}, {false, {10}},
        {false, {10}},     {false, {10}},     {true, {11}},      {false, {11}},
        {false, {11}},     {true, {12}},      {false, {12}},     {false, {12}}};
    ASSERT_EQ(steps.size(), expected.size());
    for (size_t i = 0; i < steps.size(); ++i) {
        SCOPED_TRACE(i);
        expect_step(steps[i], expected[i].first, expected[i].second);
    }

    const auto finished = s.take_finished();
    ASSERT_EQ(finished.size(), 3u);
    EXPECT_EQ(finished[0].id, 10);
    EXPECT_EQ(finished[1].id, 11);
    EXPECT_EQ(finished[2].id, 12);
    EXPECT_EQ(finished[0].preemptions, 0);
    EXPECT_EQ(finished[1].preemptions, 1);
    EXPECT_EQ(finished[2].preemptions, 1);
    EXPECT_EQ(finished[0].output.size(), 8u);
    EXPECT_EQ(finished[1].output.size(), 8u);
    EXPECT_EQ(finished[2].output, (std::vector<int32_t>{22, 49, 50, 51}));
    EXPECT_EQ(finished[1].first_token, first_token_11);
    EXPECT_EQ(finished[2].first_token, first_token_12);
    for (const Request& r : finished) EXPECT_LE(r.first_token, r.finished);
    EXPECT_EQ(s.blocks().num_free(), 4);
}

TEST(Scheduler, ASequenceThatCannotGrowPreemptsItself) {
    Scheduler s({.num_blocks = 4, .block_size = 4, .max_prefill_tokens = 12, .max_context = 12});
    s.submit(make(0, 3, 5));
    s.submit(make(1, 8, 5));
    expect_step(s.schedule(), true, {0, 1});
    give(s, {5, 6});
    expect_step(s.schedule(), false, {0});
    EXPECT_EQ(s.request(1).state, RequestState::Waiting);
    EXPECT_EQ(s.request(1).preemptions, 1);
    EXPECT_EQ(s.blocks().length(0), 4);
    EXPECT_EQ(s.blocks().num_free(), 2);
}

TEST(Scheduler, PreemptedSequencesGoToTheFrontInAdmissionOrder) {
    Scheduler s({.num_blocks = 4,
                 .block_size = 4,
                 .max_batch = 3,
                 .max_prefill_tokens = 12,
                 .max_context = 12});
    for (SeqId id = 0; id < 4; ++id) s.submit(make(id, 4, 6));
    expect_step(s.schedule(), true, {0, 1, 2});
    give(s, {5, 5, 5});
    expect_step(s.schedule(), false, {0});
    EXPECT_EQ(s.num_waiting(), 3);
    EXPECT_EQ(s.request(1).preemptions, 1);
    EXPECT_EQ(s.request(2).preemptions, 1);
    EXPECT_EQ(s.request(3).preemptions, 0);
    give(s, {6});

    std::vector<SeqId> admitted;
    for (int i = 0; i < 100 && s.num_waiting() + s.num_running() > 0; ++i) {
        const Step step = s.schedule();
        if (step.prefill)
            for (const SeqId seq : step.seqs)
                if (std::ranges::find(admitted, seq) == admitted.end()) admitted.push_back(seq);
        give(s, std::vector<int32_t>(step.seqs.size(), 7));
    }
    EXPECT_EQ(admitted, (std::vector<SeqId>{1, 2, 3}));
}

TEST(Scheduler, RejectedCallsChangeNothing) {
    const SchedulerConfig config{
        .num_blocks = 9, .block_size = 4, .max_prefill_tokens = 32, .max_context = 32};
    Scheduler s(config);
    s.submit(make(0, 5, 4));
    s.submit(make(1, 6, 4));
    const Step step = s.schedule();
    expect_step(step, true, {0, 1});
    s.submit(make(2, 3, 4));
    auto state = [&s] {
        std::vector<int64_t> v = {s.num_waiting(), s.num_running(), s.blocks().num_free()};
        for (const SeqId id : {0, 1, 2}) {
            const Request& r = s.request(id);
            v.push_back(int64_t(r.state));
            v.push_back(r.length());
            v.push_back(r.preemptions);
        }
        for (const SeqId id : {0, 1}) v.push_back(s.blocks().length(id));
        return v;
    };
    const auto before = state();
    EXPECT_THROW(s.schedule(), std::invalid_argument);
    EXPECT_EQ(state(), before);
    EXPECT_THROW(give(s, {5}), std::invalid_argument);
    EXPECT_EQ(state(), before);
    EXPECT_THROW(give(s, {5, 6, 7}), std::invalid_argument);
    EXPECT_EQ(state(), before);
    EXPECT_THROW(give(s, {}), std::invalid_argument);
    EXPECT_EQ(state(), before);
    EXPECT_THROW(s.submit(make(1, 3, 3)), std::invalid_argument);
    EXPECT_THROW(s.submit(make(3, 0, 3)), std::invalid_argument);
    EXPECT_THROW(s.request(3), std::invalid_argument);
    EXPECT_EQ(state(), before);

    give(s, {5, 6});
    EXPECT_THROW(give(s, {5, 6}), std::invalid_argument);
    EXPECT_EQ(s.request(0).output, (std::vector<int32_t>{5}));
    expect_step(s.schedule(), true, {2});
}

TEST(Scheduler, SameCallsGiveSameSteps) {
    const SchedulerConfig config{.num_blocks = 13,
                                 .block_size = 4,
                                 .max_batch = 5,
                                 .max_prefill_tokens = 40,
                                 .max_context = 40};
    Scheduler a(config);
    Scheduler b(config);
    FakeRunner runner(config);
    std::mt19937 rng(20261010);
    SeqId next_id = 0;
    for (int i = 0; i < 2000; ++i) {
        if (rng() % 3 == 0) {
            const int len = 1 + int(rng() % 30);
            const int max_new = 1 + int(rng() % (41 - len));
            const Request r = make(next_id++, len, max_new, rng() % 4);
            a.submit(r);
            b.submit(r);
        }
        const Step sa = a.schedule();
        const Step sb = b.schedule();
        ASSERT_EQ(sa.prefill, sb.prefill) << "step " << i;
        ASSERT_EQ(sa.seqs, sb.seqs) << "step " << i;
        for (const SeqId seq : sa.seqs) ASSERT_EQ(a.blocks().table(seq), b.blocks().table(seq));
        if (sa.seqs.empty()) continue;
        const auto tokens = runner.run(a, sa);
        a.update(tokens);
        b.update(tokens);
    }
}

TEST(Scheduler, SixteenConcurrentRequestsMatchSoloRuns) {
    const auto requests = sixteen_requests();
    const SchedulerConfig config{.num_blocks = 1 + 16 * 16,
                                 .block_size = 16,
                                 .max_batch = 16,
                                 .max_prefill_tokens = 512,
                                 .max_context = 256,
                                 .eos_id = kEos};
    const RunResult result = run_to_completion(config, requests);
    const FakeModel model;
    ASSERT_EQ(result.finished.size(), 16u);
    int eos_endings = 0;
    for (const Request& r : requests) {
        SCOPED_TRACE(r.id);
        const Request& done = result.finished.at(r.id);
        EXPECT_EQ(done.output, model.solo(r));
        EXPECT_EQ(done.state, RequestState::Finished);
        eos_endings += done.output.back() == kEos;
    }
    EXPECT_EQ(result.preemptions, 0);
    EXPECT_GE(result.max_decode_batch, 12);
    EXPECT_GE(eos_endings, 2);
    EXPECT_LE(eos_endings, 12);
}

TEST(Scheduler, ForcedPreemptionStillMatchesSoloRuns) {
    const auto requests = sixteen_requests();
    const FakeModel model;
    const SchedulerConfig configs[] = {
        {.num_blocks = 17, .block_size = 16, .max_prefill_tokens = 256, .max_context = 256},
        {.num_blocks = 20,
         .block_size = 16,
         .max_batch = 4,
         .max_prefill_tokens = 300,
         .max_context = 256},
        {.num_blocks = 70, .block_size = 4, .max_prefill_tokens = 2048, .max_context = 256},
        {.num_blocks = 300,
         .block_size = 1,
         .max_batch = 7,
         .max_prefill_tokens = 256,
         .max_context = 256},
        {.num_blocks = 21, .block_size = 13, .max_prefill_tokens = 256, .max_context = 256},
    };
    for (SchedulerConfig config : configs) {
        config.eos_id = kEos;
        SCOPED_TRACE(testing::Message() << config.num_blocks << " blocks of " << config.block_size);
        const RunResult result = run_to_completion(config, requests);
        ASSERT_EQ(result.finished.size(), 16u);
        for (const Request& r : requests) {
            SCOPED_TRACE(r.id);
            EXPECT_EQ(result.finished.at(r.id).output, model.solo(r));
        }
        EXPECT_GE(result.preemptions, 3);
    }
}

TEST(Scheduler, RandomStepsKeepEveryInvariant) {
    const SchedulerConfig config{.num_blocks = 17,
                                 .block_size = 4,
                                 .max_batch = 6,
                                 .max_prefill_tokens = 80,
                                 .max_context = 64,
                                 .eos_id = kEos};
    Scheduler s(config);
    FakeRunner runner(config);
    const FakeModel model;
    std::mt19937 rng(20261011);
    std::map<SeqId, Request> submitted;
    std::vector<SeqId> held;
    SeqId next_id = 1000;
    int finished = 0;
    int preemptions = 0;
    for (int i = 0; i < 10000; ++i) {
        const int arrivals = s.num_waiting() < 8 && rng() % 2 == 0 ? int(rng() % 3) : 0;
        for (int a = 0; a < arrivals; ++a) {
            const int len = 1 + int(rng() % 48);
            const int max_new = 1 + int(rng() % (65 - len));
            const Request r = make(next_id++, len, max_new, rng() % 8);
            s.submit(r);
            submitted[r.id] = r;
            held.push_back(r.id);
        }
        const Step step = s.schedule();
        ASSERT_EQ(step.seqs.empty(), s.num_waiting() + s.num_running() == 0) << "step " << i;
        expect_invariants(s, config, held, &step);
        if (step.prefill) {
            int tokens = 0;
            for (const SeqId seq : step.seqs) tokens += s.blocks().length(seq);
            ASSERT_LE(tokens, config.max_prefill_tokens);
        }
        if (!step.seqs.empty()) s.update(runner.run(s, step));
        expect_invariants(s, config, held, nullptr);
        if (HasFatalFailure()) return;
        if (rng() % 5 == 0) {
            for (const Request& r : s.take_finished()) {
                ASSERT_EQ(r.output, model.solo(submitted.at(r.id))) << "seq " << r.id;
                std::erase(held, r.id);
                ++finished;
                preemptions += r.preemptions;
            }
        }
    }
    EXPECT_GE(finished, 600);
    EXPECT_GE(preemptions, 100);
}
