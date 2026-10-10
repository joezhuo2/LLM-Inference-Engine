#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>

#include "engine/sampling/params.h"

using engine::SamplingParams;
using engine::validate;

TEST(SamplingParams, DefaultsToGreedy) {
    const SamplingParams p;
    EXPECT_EQ(p.temperature, 0.0f);
    EXPECT_NO_THROW(validate(p));
}

TEST(SamplingParams, AcceptsZeroAndPositiveTemperatures) {
    for (const float t : {0.0f, 1e-6f, 0.7f, 1.0f, 100.0f}) EXPECT_NO_THROW(validate({t, 0})) << t;
}

TEST(SamplingParams, RejectsNegativeInfiniteAndNaNTemperatures) {
    for (const float t :
         {-1.0f, -0.0001f, INFINITY, -INFINITY, std::numeric_limits<float>::quiet_NaN()})
        EXPECT_THROW(validate({t, 0}), std::invalid_argument) << t;
}
