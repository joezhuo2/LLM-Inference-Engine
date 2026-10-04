#include <cuda_bf16.h>
#include <gtest/gtest.h>

#include <bit>
#include <cstdint>
#include <random>

#include "support/bf16.h"

TEST(Bf16, MatchesCudaRounding) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint32_t> bits;
    for (int i = 0; i < 1'000'000; ++i) {
        float f = std::bit_cast<float>(bits(rng));
        if (std::isnan(f)) continue;
        uint16_t expected = std::bit_cast<uint16_t>(__float2bfloat16(f));
        ASSERT_EQ(engine::test::to_bf16(f), expected)
            << "input bits " << std::bit_cast<uint32_t>(f);
    }
}
