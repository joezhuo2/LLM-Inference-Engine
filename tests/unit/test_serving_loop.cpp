#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <nlohmann/json.hpp>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "engine/server/serving_loop.h"
#include "support/scripted_backend.h"

using engine::Request;
using engine::SeqId;
using engine::server::FinishReason;
using engine::server::ServingLoop;
using engine::server::Submission;
using engine::server::Ticket;
using engine::server::TokenEvent;
using engine::test::kEos;
using engine::test::ScriptedBackend;
using engine::test::word_decode;
using std::chrono::milliseconds;

namespace {

std::vector<TokenEvent> drain(const Ticket& ticket) {
    std::vector<TokenEvent> events;
    while (auto e = ticket.events->pop(milliseconds(0))) events.push_back(*e);
    return events;
}

std::string text_of(const std::vector<TokenEvent>& events) {
    std::string out;
    for (const TokenEvent& e : events) out += e.text;
    return out;
}

void run_until_idle(ServingLoop& loop) {
    for (int i = 0; i < 10000 && loop.run_once(); ++i) {
    }
}

Submission ask(int prompt_len, int max_tokens, std::vector<std::string> stop = {}) {
    Submission s{.max_tokens = max_tokens, .stop = std::move(stop)};
    for (int i = 0; i < prompt_len; ++i) s.prompt.push_back(100 + i);
    return s;
}

}  // namespace

TEST(ServingLoop, StreamsEachTokenWithItsStepAndEndsWithLength) {
    ScriptedBackend backend([](const Request&) { return 11; });
    ServingLoop loop(backend, word_decode, kEos);
    const Ticket t = loop.submit(ask(5, 3));
    run_until_idle(loop);
    const auto events = drain(t);
    ASSERT_EQ(events.size(), 4u);
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(events[i].text, " w11");
        EXPECT_EQ(events[i].step, i);
        EXPECT_FALSE(events[i].finish.has_value());
    }
    EXPECT_EQ(events[3].finish, FinishReason::Length);
    EXPECT_EQ(events[3].completion_tokens, 3);
    EXPECT_EQ(events[3].step, 2);
    EXPECT_TRUE(t.events->closed());
}

TEST(ServingLoop, EosEndsWithStopAndAddsNoText) {
    ScriptedBackend backend([](const Request& r) { return r.output.size() == 1 ? kEos : 12; });
    ServingLoop loop(backend, word_decode, kEos);
    const Ticket t = loop.submit(ask(5, 50));
    run_until_idle(loop);
    const auto events = drain(t);
    ASSERT_EQ(events.size(), 2u);
    EXPECT_EQ(events[0].text, " w12");
    EXPECT_EQ(events[1].finish, FinishReason::Stop);
    EXPECT_EQ(events[1].text, "");
    EXPECT_EQ(events[1].completion_tokens, 2);
}

TEST(ServingLoop, AStopStringEndsTheAnswerBeforeIt) {
    ScriptedBackend backend([](const Request& r) { return int32_t(10 + r.output.size()); });
    ServingLoop loop(backend, word_decode, kEos);
    const Ticket t = loop.submit(ask(5, 20, {" w13"}));
    run_until_idle(loop);
    const auto events = drain(t);
    EXPECT_EQ(text_of(events), " w10 w11 w12");
    ASSERT_FALSE(events.empty());
    EXPECT_EQ(events.back().finish, FinishReason::Stop);
    EXPECT_EQ(events.back().completion_tokens, 4);
    EXPECT_EQ(backend.scheduler().blocks().num_free(), backend.num_blocks() - 1);
}

TEST(ServingLoop, ACancelledRequestGetsNoMoreEventsAndItsBlocksComeBack) {
    ScriptedBackend backend([](const Request&) { return 11; });
    ServingLoop loop(backend, word_decode, kEos);
    const Ticket t = loop.submit(ask(40, 30));
    loop.run_once();
    loop.run_once();
    EXPECT_EQ(drain(t).size(), 2u);
    loop.cancel(t.id);
    run_until_idle(loop);
    EXPECT_TRUE(drain(t).empty());
    EXPECT_TRUE(t.events->closed());
    const auto m = loop.metrics();
    EXPECT_EQ(m.counters.finished_total, 0);
    EXPECT_EQ(m.counters.running, 0);
    EXPECT_EQ(m.counters.blocks_free, backend.num_blocks() - 1);
}

TEST(ServingLoop, ACancelForAFinishedRequestIsIgnored) {
    ScriptedBackend backend([](const Request&) { return 11; });
    ServingLoop loop(backend, word_decode, kEos);
    const Ticket t = loop.submit(ask(5, 1));
    run_until_idle(loop);
    loop.cancel(t.id);
    loop.cancel(12345);
    EXPECT_FALSE(loop.run_once());
    EXPECT_EQ(drain(t).back().finish, FinishReason::Length);
}

TEST(ServingLoop, PublishesEveryStepAsTheTraceStepObject) {
    ScriptedBackend backend([](const Request& r) { return r.id == 0 ? 11 : 12; });
    ServingLoop loop(backend, word_decode, kEos);
    const auto steps = loop.steps().subscribe();
    loop.submit(ask(5, 2));
    loop.submit(ask(7, 3));
    run_until_idle(loop);
    std::vector<nlohmann::json> seen;
    while (auto s = steps->pop(milliseconds(0))) seen.push_back(nlohmann::json::parse(*s));
    ASSERT_EQ(seen.size(), 3u);
    EXPECT_EQ(seen[0]["index"], 0);
    EXPECT_EQ(seen[0]["kind"], "prefill");
    EXPECT_EQ(seen[0]["seqs"], (std::vector<SeqId>{0, 1}));
    EXPECT_EQ(seen[0]["tokens"][1]["text"], " w12");
    EXPECT_EQ(seen[1]["kind"], "decode");
    EXPECT_EQ(seen[1]["finished"], (std::vector<SeqId>{0}));
    EXPECT_EQ(seen[2]["index"], 2);
    EXPECT_EQ(seen[2]["seqs"], (std::vector<SeqId>{1}));
    EXPECT_EQ(seen[2]["running"], 0);
    EXPECT_GE(seen[2]["start_s"].get<double>(), 0.0);
    EXPECT_GE(seen[2]["end_s"].get<double>(), seen[2]["start_s"].get<double>());
}

TEST(ServingLoop, CountsStepsTokensRequestsAndLatency) {
    ScriptedBackend backend([](const Request&) { return 11; });
    ServingLoop loop(backend, word_decode, kEos);
    loop.submit(ask(5, 4));
    loop.submit(ask(5, 1));
    run_until_idle(loop);
    const auto m = loop.metrics();
    EXPECT_EQ(m.counters.steps_total, 4);
    EXPECT_EQ(m.counters.output_total, 5);
    EXPECT_EQ(m.counters.finished_total, 2);
    EXPECT_EQ(m.counters.waiting, 0);
    EXPECT_EQ(m.counters.running, 0);
    EXPECT_EQ(m.blocks_total, backend.num_blocks());
    EXPECT_GT(m.output_per_s, 0.0);
    ASSERT_TRUE(m.ttft_s.has_value());
    EXPECT_GE(m.ttft_s->p50, 0.0);
    ASSERT_TRUE(m.tpot_s.has_value());
    EXPECT_GT(m.time_s, 0.0);
}

TEST(ServingLoop, LatencyIsMissingBeforeAnyRequestFinishes) {
    ScriptedBackend backend([](const Request&) { return 11; });
    ServingLoop loop(backend, word_decode, kEos);
    const auto m = loop.metrics();
    EXPECT_FALSE(m.ttft_s.has_value());
    EXPECT_FALSE(m.tpot_s.has_value());
    EXPECT_EQ(m.counters.blocks_free, backend.num_blocks() - 1);
}

TEST(ServingLoop, CountsAPreemptionWhenASequenceIsPrefilledAgain) {
    ScriptedBackend backend([](const Request&) { return 11; }, 6, 4, 16);
    ServingLoop loop(backend, word_decode, kEos);
    const Ticket a = loop.submit(ask(4, 12));
    const Ticket b = loop.submit(ask(4, 12));
    run_until_idle(loop);
    EXPECT_GT(backend.preemptions(), 0);
    EXPECT_EQ(loop.metrics().counters.preemptions_total, backend.preemptions());
    for (const Ticket& t : {a, b}) {
        const auto events = drain(t);
        ASSERT_FALSE(events.empty());
        EXPECT_EQ(events.back().completion_tokens, 12);
        EXPECT_EQ(int(events.size()), 13);
    }
}

TEST(ServingLoop, ServesRequestsSubmittedFromManyThreadsWhileItRuns) {
    ScriptedBackend backend([](const Request& r) { return int32_t(10 + r.id % 5); });
    ServingLoop loop(backend, word_decode, kEos);
    std::jthread engine([&](std::stop_token stop) { loop.run(stop); });
    std::vector<std::jthread> clients;
    std::vector<int> tokens(16, 0);
    for (int c = 0; c < 16; ++c)
        clients.emplace_back([&, c] {
            const Ticket t = loop.submit(ask(3 + c, 5 + c));
            while (auto e = t.events->pop(std::chrono::seconds(10)))
                if (e->finish) tokens[c] = e->completion_tokens;
        });
    clients.clear();
    for (int c = 0; c < 16; ++c) EXPECT_EQ(tokens[c], 5 + c) << c;
}
