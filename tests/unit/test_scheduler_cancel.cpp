#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>

#include "engine/sched/scheduler.h"

using engine::Request;
using engine::RequestState;
using engine::Scheduler;
using engine::SchedulerConfig;
using engine::SeqId;
using engine::Step;

namespace {

constexpr int32_t kEos = 2;
constexpr int32_t kWord = 7;

Request make(SeqId id, int prompt_len, int max_new_tokens) {
    Request r{.id = id, .max_new_tokens = max_new_tokens};
    for (int i = 0; i < prompt_len; ++i) r.prompt.push_back(int32_t(3 + (id * 31 + i) % 97));
    return r;
}

SchedulerConfig config(int num_blocks, int block_size, int max_batch, int max_context) {
    return {.num_blocks = num_blocks,
            .block_size = block_size,
            .max_batch = max_batch,
            .max_prefill_tokens = max_context,
            .max_context = max_context,
            .eos_id = kEos};
}

int32_t next_token(const Request& r) {
    uint64_t h = 1469598103934665603ull ^ uint64_t(r.id);
    for (int64_t i = 0; i < r.length(); ++i) h = (h ^ uint64_t(r.token(i))) * 1099511628211ull;
    h ^= h >> 29;
    return h % 41 == 0 ? kEos : int32_t(3 + h % 97);
}

std::vector<int32_t> solo(Request r) {
    std::vector<int32_t> out;
    while (true) {
        const int32_t t = next_token(r);
        out.push_back(t);
        r.output.push_back(t);
        if (t == kEos || int(out.size()) == r.max_new_tokens) return out;
    }
}

Step run_step(Scheduler& s) {
    const Step step = s.schedule();
    std::vector<int32_t> tokens;
    for (const SeqId seq : step.seqs) tokens.push_back(next_token(s.request(seq)));
    if (!step.seqs.empty()) s.update(tokens);
    return step;
}

void give_words(Scheduler& s) {
    const Step step = s.schedule();
    s.update(std::vector<int32_t>(step.seqs.size(), kWord));
}

}  // namespace

TEST(SchedulerCancel, AWaitingRequestLeavesTheQueue) {
    Scheduler s(config(64, 16, 4, 256));
    s.submit(make(0, 20, 8));
    s.submit(make(1, 20, 8));
    const int free = s.blocks().num_free();
    s.cancel(0);
    EXPECT_EQ(s.num_waiting(), 1);
    EXPECT_EQ(s.num_running(), 0);
    EXPECT_EQ(s.blocks().num_free(), free);
    EXPECT_THROW(s.request(0), std::invalid_argument);
    const Step step = s.schedule();
    EXPECT_TRUE(step.prefill);
    EXPECT_EQ(step.seqs, (std::vector<SeqId>{1}));
}

TEST(SchedulerCancel, ARunningRequestFreesAllItsBlocksAtOnce) {
    Scheduler s(config(64, 16, 4, 256));
    s.submit(make(0, 40, 50));
    for (int i = 0; i < 10; ++i) give_words(s);
    EXPECT_EQ(s.blocks().num_free(), 63 - 4);
    s.cancel(0);
    EXPECT_EQ(s.blocks().num_free(), 63);
    EXPECT_EQ(s.num_running(), 0);
    EXPECT_EQ(s.num_waiting(), 0);
    EXPECT_THROW(s.request(0), std::invalid_argument);
    EXPECT_TRUE(s.schedule().seqs.empty());
}

TEST(SchedulerCancel, OnlyTheOtherBlocksStayHeld) {
    Scheduler s(config(64, 16, 4, 256));
    s.submit(make(0, 17, 50));
    s.submit(make(1, 33, 50));
    give_words(s);
    EXPECT_EQ(s.blocks().num_free(), 63 - 2 - 3);
    s.cancel(1);
    EXPECT_EQ(s.blocks().num_free(), 63 - 2);
    EXPECT_EQ(s.blocks().length(0), 17);
    const Step step = s.schedule();
    EXPECT_FALSE(step.prefill);
    EXPECT_EQ(step.seqs, (std::vector<SeqId>{0}));
}

TEST(SchedulerCancel, TheOtherRequestsKeepTheirBatchOrder) {
    Scheduler s(config(64, 16, 8, 256));
    for (SeqId id = 0; id < 4; ++id) s.submit(make(id, 10, 20));
    for (SeqId id = 4; id < 6; ++id) s.submit(make(id, 10, 20));
    give_words(s);
    s.cancel(1);
    s.cancel(4);
    const Step step = s.schedule();
    EXPECT_FALSE(step.prefill);
    EXPECT_EQ(step.seqs, (std::vector<SeqId>{0, 2, 3, 5}));
}

TEST(SchedulerCancel, ACancelledRequestIsNeverReturnedAsFinished) {
    Scheduler s(config(64, 16, 4, 256));
    s.submit(make(0, 10, 3));
    s.submit(make(1, 10, 3));
    give_words(s);
    s.cancel(0);
    std::vector<SeqId> finished;
    while (s.num_running() + s.num_waiting() > 0) {
        give_words(s);
        for (const Request& r : s.take_finished()) finished.push_back(r.id);
    }
    EXPECT_EQ(finished, (std::vector<SeqId>{1}));
}

TEST(SchedulerCancel, FreeingTheFrontOfTheQueueAdmitsTheNextRequest) {
    Scheduler s(config(9, 4, 4, 32));
    s.submit(make(0, 20, 4));
    s.submit(make(1, 20, 4));
    s.submit(make(2, 12, 4));
    EXPECT_EQ(s.schedule().seqs, (std::vector<SeqId>{0}));
    s.update(std::vector<int32_t>{kWord});
    s.cancel(1);
    const Step step = s.schedule();
    EXPECT_TRUE(step.prefill);
    EXPECT_EQ(step.seqs, (std::vector<SeqId>{2}));
}

TEST(SchedulerCancel, APreemptedRequestCanBeCancelledWhileItWaits) {
    Scheduler s(config(6, 4, 2, 16));
    s.submit(make(0, 4, 12));
    s.submit(make(1, 4, 12));
    SeqId victim = -1;
    for (int i = 0; i < 40 && victim < 0; ++i) {
        give_words(s);
        for (const SeqId id : {SeqId(0), SeqId(1)})
            if (s.request(id).state == RequestState::Waiting && s.request(id).preemptions > 0)
                victim = id;
    }
    ASSERT_GE(victim, 0) << "the setup never preempted a request";
    const int waiting = s.num_waiting();
    s.cancel(victim);
    EXPECT_EQ(s.num_waiting(), waiting - 1);
    std::vector<SeqId> finished;
    while (s.num_running() + s.num_waiting() > 0) {
        const Step step = s.schedule();
        EXPECT_EQ(std::count(step.seqs.begin(), step.seqs.end(), victim), 0);
        s.update(std::vector<int32_t>(step.seqs.size(), kWord));
        for (const Request& r : s.take_finished()) finished.push_back(r.id);
    }
    EXPECT_EQ(finished, (std::vector<SeqId>{1 - victim}));
    EXPECT_EQ(s.blocks().num_free(), 5);
}

TEST(SchedulerCancel, RejectsIdsItDoesNotHold) {
    Scheduler s(config(64, 16, 4, 256));
    EXPECT_THROW(s.cancel(5), std::invalid_argument);
    s.submit(make(0, 10, 1));
    s.cancel(0);
    EXPECT_THROW(s.cancel(0), std::invalid_argument);
    s.submit(make(1, 10, 1));
    give_words(s);
    EXPECT_THROW(s.cancel(1), std::invalid_argument);
    EXPECT_EQ(s.take_finished().size(), 1u);
    EXPECT_THROW(s.cancel(1), std::invalid_argument);
    EXPECT_EQ(s.blocks().num_free(), 63);
}

TEST(SchedulerCancel, RejectsACancelWhileAStepIsPending) {
    Scheduler s(config(64, 16, 4, 256));
    s.submit(make(0, 10, 5));
    const Step step = s.schedule();
    const int free = s.blocks().num_free();
    EXPECT_THROW(s.cancel(0), std::invalid_argument);
    EXPECT_EQ(s.blocks().num_free(), free);
    EXPECT_EQ(s.num_running(), 1);
    s.update(std::vector<int32_t>(step.seqs.size(), kWord));
    EXPECT_EQ(s.request(0).output.size(), 1u);
    s.cancel(0);
    EXPECT_EQ(s.num_running(), 0);
}

TEST(SchedulerCancel, TheIdCanBeSubmittedAgain) {
    Scheduler s(config(64, 16, 4, 256));
    s.submit(make(0, 10, 5));
    give_words(s);
    s.cancel(0);
    s.submit(make(0, 12, 2));
    EXPECT_EQ(s.request(0).output.size(), 0u);
    EXPECT_EQ(s.request(0).state, RequestState::Waiting);
    give_words(s);
    give_words(s);
    const auto done = s.take_finished();
    ASSERT_EQ(done.size(), 1u);
    EXPECT_EQ(done[0].prompt.size(), 12u);
}

TEST(SchedulerCancel, RandomCancelsKeepTheOtherAnswersAndReturnEveryBlock) {
    std::mt19937 rng(7);
    for (int trial = 0; trial < 20; ++trial) {
        Scheduler s(config(24, 4, 4, 48));
        std::map<SeqId, Request> submitted;
        for (SeqId id = 0; id < 12; ++id) {
            Request r = make(id, 1 + int(rng() % 20), 1 + int(rng() % 28));
            submitted[id] = r;
            s.submit(r);
        }
        std::set<SeqId> cancelled;
        std::map<SeqId, std::vector<int32_t>> answers;
        while (s.num_running() + s.num_waiting() > 0) {
            const Step step = run_step(s);
            for (const SeqId seq : step.seqs) EXPECT_FALSE(cancelled.count(seq)) << seq;
            for (const Request& r : s.take_finished()) answers[r.id] = r.output;
            if (rng() % 4 == 0) {
                const SeqId id = SeqId(rng() % 12);
                if (cancelled.count(id) || answers.count(id)) continue;
                s.cancel(id);
                cancelled.insert(id);
            }
        }
        EXPECT_EQ(s.blocks().num_free(), 23);
        EXPECT_EQ(answers.size() + cancelled.size(), submitted.size());
        for (const auto& [id, out] : answers) EXPECT_EQ(out, solo(submitted[id])) << id;
    }
}
