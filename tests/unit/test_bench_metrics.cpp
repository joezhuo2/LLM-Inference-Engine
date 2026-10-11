#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine/bench/csv.h"
#include "engine/bench/metrics.h"

using engine::Clock;
using engine::percentile;
using engine::Request;
using engine::RequestMetrics;
using engine::RequestState;
using std::chrono::milliseconds;

namespace {

const Clock::time_point kStart = Clock::time_point{} + std::chrono::hours(1);

Request finished(int id, int outputs, int arrive_ms, int first_ms, int end_ms) {
    Request r{.id = id, .prompt = {1, 2, 3}, .max_new_tokens = outputs};
    r.output.assign(size_t(outputs), 7);
    r.state = RequestState::Finished;
    r.arrived = kStart + milliseconds(arrive_ms);
    r.first_token = kStart + milliseconds(first_ms);
    r.finished = kStart + milliseconds(end_ms);
    return r;
}

std::vector<std::string> lines(const std::string& text) {
    std::vector<std::string> out;
    std::istringstream in(text);
    for (std::string line; std::getline(in, line);) out.push_back(line);
    return out;
}

}  // namespace

TEST(RequestMetrics, FollowsTheBuildPlanDefinitions) {
    auto r = finished(4, 11, 100, 350, 1350);
    r.preemptions = 2;
    const auto m = engine::request_metrics(r, kStart);
    EXPECT_EQ(m.id, 4);
    EXPECT_EQ(m.prompt_tokens, 3);
    EXPECT_EQ(m.output_tokens, 11);
    EXPECT_EQ(m.preemptions, 2);
    EXPECT_DOUBLE_EQ(m.arrival_s, 0.1);
    EXPECT_DOUBLE_EQ(m.ttft_s, 0.25);
    EXPECT_DOUBLE_EQ(m.tpot_s, 0.1);
    EXPECT_DOUBLE_EQ(m.latency_s, 1.25);
}

TEST(RequestMetrics, OneTokenHasNoTpot) {
    const auto m = engine::request_metrics(finished(0, 1, 0, 40, 40), kStart);
    EXPECT_TRUE(std::isnan(m.tpot_s));
    EXPECT_DOUBLE_EQ(m.ttft_s, 0.04);
    EXPECT_DOUBLE_EQ(m.latency_s, 0.04);
}

TEST(RequestMetrics, RejectsUnfinishedOrDisorderedRequests) {
    auto running = finished(0, 3, 0, 10, 20);
    running.state = RequestState::Running;
    auto empty = finished(0, 3, 0, 10, 20);
    empty.output.clear();
    for (const auto& r : {running, empty, finished(0, 3, -1, 10, 20), finished(0, 3, 11, 10, 20),
                          finished(0, 3, 0, 21, 20)})
        EXPECT_THROW(engine::request_metrics(r, kStart), std::invalid_argument);
}

TEST(Percentile, InterpolatesLinearlyLikeNumpy) {
    const std::vector<double> v = {5, 1, 4, 2, 3};
    EXPECT_DOUBLE_EQ(percentile(v, 0), 1);
    EXPECT_DOUBLE_EQ(percentile(v, 50), 3);
    EXPECT_DOUBLE_EQ(percentile(v, 90), 4.6);
    EXPECT_DOUBLE_EQ(percentile(v, 100), 5);
    EXPECT_DOUBLE_EQ(percentile({1, 2, 3, 4}, 50), 2.5);
    EXPECT_DOUBLE_EQ(percentile({1, 2, 3, 4}, 99), 3.97);
    EXPECT_DOUBLE_EQ(percentile({8}, 90), 8);
    EXPECT_TRUE(std::isnan(percentile({}, 50)));
    EXPECT_THROW(percentile(v, -1), std::invalid_argument);
    EXPECT_THROW(percentile(v, 101), std::invalid_argument);
    EXPECT_THROW(percentile(v, std::nan("")), std::invalid_argument);
}

TEST(Summary, AddsUpTokensAndThroughput) {
    std::vector<RequestMetrics> m;
    for (const auto& r :
         {finished(0, 5, 0, 100, 500), finished(1, 1, 0, 200, 200), finished(2, 9, 0, 300, 1100)})
        m.push_back(engine::request_metrics(r, kStart));
    const auto s = engine::summarize(m, 2.0);
    EXPECT_EQ(s.requests, 3);
    EXPECT_EQ(s.prompt_tokens, 9);
    EXPECT_EQ(s.output_tokens, 15);
    EXPECT_DOUBLE_EQ(s.request_throughput, 1.5);
    EXPECT_DOUBLE_EQ(s.output_throughput, 7.5);
    EXPECT_DOUBLE_EQ(s.ttft_s.mean, 0.2);
    EXPECT_DOUBLE_EQ(s.ttft_s.p50, 0.2);
    EXPECT_DOUBLE_EQ(s.tpot_s.mean, 0.1);
    EXPECT_DOUBLE_EQ(s.tpot_s.p90, 0.1);
    EXPECT_DOUBLE_EQ(s.latency_s.p50, 0.5);
    EXPECT_DOUBLE_EQ(s.latency_s.p90, 0.98);
}

TEST(Summary, TpotIsMissingWhenEveryRequestHasOneToken) {
    const std::vector m = {engine::request_metrics(finished(0, 1, 0, 10, 10), kStart)};
    const auto s = engine::summarize(m, 1.0);
    EXPECT_TRUE(std::isnan(s.tpot_s.mean));
    EXPECT_TRUE(std::isnan(s.tpot_s.p50));
}

TEST(Summary, RejectsNoRequestsOrABadWallTime) {
    const std::vector m = {engine::request_metrics(finished(0, 2, 0, 10, 20), kStart)};
    EXPECT_THROW(engine::summarize({}, 1.0), std::invalid_argument);
    for (const double wall : {0.0, -1.0, std::numeric_limits<double>::infinity(), std::nan("")})
        EXPECT_THROW(engine::summarize(m, wall), std::invalid_argument);
}

TEST(Csv, QuotesOnlyWhenNeeded) {
    EXPECT_EQ(engine::csv_escape("plain text"), "plain text");
    EXPECT_EQ(engine::csv_escape("a,b"), "\"a,b\"");
    EXPECT_EQ(engine::csv_escape("say \"hi\""), "\"say \"\"hi\"\"\"");
    EXPECT_EQ(engine::csv_escape("two\nlines"), "\"two\nlines\"");
    EXPECT_EQ(engine::csv_escape(""), "");
}

TEST(Csv, NumbersRoundTripAndNanIsEmpty) {
    EXPECT_EQ(engine::csv_number(0.25), "0.25");
    EXPECT_EQ(engine::csv_number(123456.789), "123456.789");
    EXPECT_EQ(engine::csv_number(1.0 / 3.0), "0.333333333");
    EXPECT_EQ(engine::csv_number(std::nan("")), "");
}

TEST(Csv, WritesOneRowPerRequest) {
    std::vector<RequestMetrics> m = {
        engine::request_metrics(finished(0, 11, 100, 350, 1350), kStart),
        engine::request_metrics(finished(1, 1, 0, 40, 40), kStart)};
    std::ostringstream out;
    engine::write_request_csv(out, m);
    EXPECT_EQ(lines(out.str()),
              (std::vector<std::string>{
                  "id,prompt_tokens,output_tokens,preemptions,arrival_s,ttft_s,tpot_s,latency_s",
                  "0,3,11,0,0.1,0.25,0.1,1.25", "1,3,1,0,0,0.04,,0.04"}));
}

TEST(Csv, WritesAHeaderAndOneRow) {
    std::ostringstream out;
    engine::write_csv_row(out, {{"model", "tiny,llama"}, {"max_batch", "1"}});
    EXPECT_EQ(lines(out.str()), (std::vector<std::string>{"model,max_batch", "\"tiny,llama\",1"}));
}

TEST(Csv, SummaryFieldsKeepAFixedOrder) {
    std::vector<std::string> names;
    for (const auto& [name, value] : engine::summary_fields({})) names.push_back(name);
    EXPECT_EQ(names,
              (std::vector<std::string>{"requests", "prompt_tokens", "output_tokens", "wall_s",
                                        "request_throughput", "output_throughput", "ttft_s_mean",
                                        "ttft_s_p50", "ttft_s_p90", "ttft_s_p99", "tpot_s_mean",
                                        "tpot_s_p50", "tpot_s_p90", "tpot_s_p99", "latency_s_mean",
                                        "latency_s_p50", "latency_s_p90", "latency_s_p99"}));
}
