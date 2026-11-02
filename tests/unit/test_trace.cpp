#include <gtest/gtest.h>

#include <chrono>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "engine/bench/trace.h"

using engine::Clock;
using engine::Request;
using engine::SeqId;
using engine::step_json;
using engine::StepTrace;
using engine::trace_json;
using std::chrono::milliseconds;

namespace {

const Clock::time_point kStart = Clock::time_point{} + std::chrono::hours(1);

StepTrace step(int start_ms, int end_ms, bool prefill, std::vector<SeqId> seqs,
               std::vector<int32_t> tokens) {
    StepTrace s;
    s.start = kStart + milliseconds(start_ms);
    s.end = kStart + milliseconds(end_ms);
    s.prefill = prefill;
    s.seqs = std::move(seqs);
    s.tokens = std::move(tokens);
    return s;
}

std::string echo(SeqId seq, int32_t token) {
    return std::to_string(seq) + ":" + std::to_string(token);
}

}  // namespace

TEST(TraceJson, HasTheFormatVersionAndMetaFirst) {
    const auto j = trace_json({{"model", "tinyllama"}, {"max_batch", 16}}, {}, {}, kStart, echo);
    EXPECT_EQ(j["format"], "llmie-trace");
    EXPECT_EQ(j["version"], 1);
    EXPECT_EQ(j["meta"]["model"], "tinyllama");
    EXPECT_EQ(j["meta"]["max_batch"], 16);
    EXPECT_TRUE(j["requests"].is_array() && j["requests"].empty());
    EXPECT_TRUE(j["steps"].is_array() && j["steps"].empty());
    EXPECT_EQ(j.begin().key(), "format");
}

TEST(TraceJson, StepsCarryRelativeTimesKindsAndAlignedTokens) {
    std::vector<StepTrace> steps = {step(0, 20, true, {4, 7}, {11, 12}),
                                    step(20, 30, false, {4, 7}, {13, 2})};
    steps[1].finished = {7};
    steps[1].waiting = 3;
    steps[1].running = 1;
    steps[1].free_blocks = 99;
    const auto j = trace_json({}, {}, steps, kStart, echo);

    ASSERT_EQ(j["steps"].size(), 2u);
    const auto& first = j["steps"][0];
    EXPECT_EQ(first["index"], 0);
    EXPECT_DOUBLE_EQ(first["start_s"].get<double>(), 0.0);
    EXPECT_DOUBLE_EQ(first["end_s"].get<double>(), 0.020);
    EXPECT_EQ(first["kind"], "prefill");
    EXPECT_EQ(first["seqs"], (std::vector<SeqId>{4, 7}));
    EXPECT_EQ(first["tokens"][1]["id"], 12);
    EXPECT_EQ(first["tokens"][1]["text"], "7:12");

    const auto& second = j["steps"][1];
    EXPECT_EQ(second["index"], 1);
    EXPECT_EQ(second["kind"], "decode");
    EXPECT_EQ(second["finished"], (std::vector<SeqId>{7}));
    EXPECT_EQ(second["waiting"], 3);
    EXPECT_EQ(second["running"], 1);
    EXPECT_EQ(second["free_blocks"], 99);
}

TEST(TraceJson, AsksForTheTextOfEveryTokenInStepOrder) {
    const std::vector<StepTrace> steps = {step(0, 1, true, {1, 2}, {10, 20}),
                                          step(1, 2, false, {2, 1}, {21, 11})};
    std::vector<std::string> calls;
    trace_json({}, {}, steps, kStart, [&](SeqId seq, int32_t token) {
        calls.push_back(echo(seq, token));
        return std::string();
    });
    EXPECT_EQ(calls, (std::vector<std::string>{"1:10", "2:20", "2:21", "1:11"}));
}

TEST(TraceJson, RequestsAreListedByIdWithTheirTimes) {
    std::vector<Request> requests(2);
    requests[0].id = 5;
    requests[1].id = 2;
    requests[1].prompt = {1, 2, 3};
    requests[1].output = {4, 5};
    requests[1].preemptions = 1;
    requests[1].arrived = kStart + milliseconds(100);
    requests[1].first_token = kStart + milliseconds(150);
    requests[1].finished = kStart + milliseconds(400);
    const auto j = trace_json({}, requests, {}, kStart, echo);

    ASSERT_EQ(j["requests"].size(), 2u);
    EXPECT_EQ(j["requests"][0]["id"], 2);
    EXPECT_EQ(j["requests"][1]["id"], 5);
    const auto& r = j["requests"][0];
    EXPECT_DOUBLE_EQ(r["arrival_s"].get<double>(), 0.100);
    EXPECT_EQ(r["prompt_tokens"], 3);
    EXPECT_EQ(r["output_tokens"], 2);
    EXPECT_EQ(r["preemptions"], 1);
    EXPECT_DOUBLE_EQ(r["first_token_s"].get<double>(), 0.150);
    EXPECT_DOUBLE_EQ(r["finished_s"].get<double>(), 0.400);
}

TEST(TraceJson, RejectsAStepWithoutOneTokenPerSequence) {
    const std::vector<StepTrace> steps = {step(0, 1, false, {1, 2}, {10})};
    EXPECT_THROW(trace_json({}, {}, steps, kStart, echo), std::invalid_argument);
}

TEST(StepJson, IsTheSameObjectTheTraceWrites) {
    StepTrace s = step(5, 12, false, {3, 8}, {40, 2});
    s.finished = {8};
    s.waiting = 1;
    s.running = 1;
    s.free_blocks = 7;
    const std::vector<std::string> texts = {"3:40", "8:2"};
    const auto j = trace_json({}, {}, std::vector<StepTrace>(4, s), kStart, echo);
    EXPECT_EQ(step_json(s, 3, kStart, texts).dump(), j["steps"][3].dump());
}

TEST(StepJson, RejectsAStepWithoutOneTextPerSequence) {
    const StepTrace s = step(0, 1, false, {1, 2}, {10, 11});
    const std::vector<std::string> texts = {"a"};
    EXPECT_THROW(step_json(s, 0, kStart, texts), std::invalid_argument);
}
