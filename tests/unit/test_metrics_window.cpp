#include <gtest/gtest.h>

#include <optional>

#include "engine/server/metrics_window.h"

using engine::server::MetricsWindow;

TEST(MetricsWindow, OutputRateCountsTheLastTenSeconds) {
    MetricsWindow w;
    w.add_tokens(1.0, 100);
    w.add_tokens(12.0, 30);
    w.add_tokens(15.0, 20);
    EXPECT_DOUBLE_EQ(w.output_per_s(15.0), 5.0);
    EXPECT_DOUBLE_EQ(w.output_per_s(22.0), 2.0);
    EXPECT_DOUBLE_EQ(w.output_per_s(25.0), 0.0);
}

TEST(MetricsWindow, AStepExactlyTenSecondsOldHasLeftTheWindow) {
    MetricsWindow w;
    w.add_tokens(10.0, 50);
    EXPECT_DOUBLE_EQ(w.output_per_s(19.99), 5.0);
    EXPECT_DOUBLE_EQ(w.output_per_s(20.0), 0.0);
}

TEST(MetricsWindow, OutputRateInTheFirstSecondsDividesByTheTimeSinceStart) {
    MetricsWindow w;
    EXPECT_DOUBLE_EQ(w.output_per_s(0.0), 0.0);
    w.add_tokens(1.0, 40);
    EXPECT_DOUBLE_EQ(w.output_per_s(2.0), 20.0);
}

TEST(MetricsWindow, LatencyIsNullWithNoRequestsInTheLastMinute) {
    MetricsWindow w;
    EXPECT_EQ(w.ttft_s(5.0), std::nullopt);
    EXPECT_EQ(w.tpot_s(5.0), std::nullopt);
    w.add_request(10.0, 0.1, 0.01);
    EXPECT_TRUE(w.ttft_s(69.9).has_value());
    EXPECT_EQ(w.ttft_s(70.0), std::nullopt);
    EXPECT_EQ(w.tpot_s(70.0), std::nullopt);
}

TEST(MetricsWindow, LatencyPercentilesCoverTheRequestsOfTheLastMinute) {
    MetricsWindow w;
    w.add_request(1.0, 9.0, 9.0);
    for (int i = 1; i <= 101; ++i) w.add_request(70.0, 0.001 * i, 0.0001 * i);
    const auto ttft = w.ttft_s(70.0);
    ASSERT_TRUE(ttft.has_value());
    EXPECT_DOUBLE_EQ(ttft->p50, 0.051);
    EXPECT_DOUBLE_EQ(ttft->p99, 0.100);
    EXPECT_DOUBLE_EQ(w.tpot_s(70.0)->p50, 0.0051);
}

TEST(MetricsWindow, OneTokenAnswersHaveATtftButNoTpot) {
    MetricsWindow w;
    w.add_request(1.0, 0.2, std::nullopt);
    EXPECT_DOUBLE_EQ(w.ttft_s(1.0)->p50, 0.2);
    EXPECT_EQ(w.tpot_s(1.0), std::nullopt);
}
