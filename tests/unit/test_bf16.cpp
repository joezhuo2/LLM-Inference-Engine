#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "support/bf16.h"

using engine::test::from_bf16;
using engine::test::to_bf16;

TEST(Bf16, ExactValuesRoundTrip) {
    EXPECT_EQ(to_bf16(1.0f), 0x3F80);
    EXPECT_EQ(to_bf16(-2.0f), 0xC000);
    EXPECT_EQ(to_bf16(0.0f), 0x0000);
    EXPECT_EQ(from_bf16(0x3F80), 1.0f);
    EXPECT_EQ(from_bf16(to_bf16(0.15625f)), 0.15625f);
}

TEST(Bf16, TiesRoundToEven) {
    EXPECT_EQ(from_bf16(to_bf16(1.0f + 0x1p-8f)), 1.0f);
    EXPECT_EQ(from_bf16(to_bf16(1.0f + 3 * 0x1p-8f)), 1.0f + 0x1p-6f);
}

TEST(Bf16, AboveHalfRoundsUp) {
    EXPECT_EQ(from_bf16(to_bf16(1.0f + 0x1p-8f + 0x1p-20f)), 1.0f + 0x1p-7f);
}

TEST(Bf16, NaNAndInfinity) {
    EXPECT_TRUE(std::isnan(from_bf16(to_bf16(std::numeric_limits<float>::quiet_NaN()))));
    EXPECT_TRUE(std::isinf(from_bf16(to_bf16(std::numeric_limits<float>::infinity()))));
}

TEST(Bf16, UlpDistanceCountsRepresentableStepsAcrossZero) {
    using engine::test::bf16_ulps;
    EXPECT_EQ(bf16_ulps(0x3F80, 0x3F80), 0);
    EXPECT_EQ(bf16_ulps(0x3F80, 0x3F81), 1);
    EXPECT_EQ(bf16_ulps(0x0000, 0x8000), 0);
    EXPECT_EQ(bf16_ulps(0x0001, 0x8001), 2);
    EXPECT_EQ(bf16_ulps(0xBF80, 0xBF82), 2);
}

TEST(Bf16, CompareReportsCountWorstAndFirst) {
    const auto d = engine::test::compare_bf16({0x3F80, 0x4000, 0x4001, 0xC000},
                                              {0x3F80, 0x4003, 0x4000, 0xC000});
    EXPECT_EQ(d.mismatches, 2u);
    EXPECT_EQ(d.max_ulps, 3);
    EXPECT_EQ(d.first, 1u);
}
