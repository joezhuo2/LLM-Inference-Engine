#pragma once

#include <bit>
#include <cmath>
#include <cstdint>

namespace engine::test {

inline uint16_t to_bf16(float f) {
    if (std::isnan(f)) return 0x7FC0;
    uint32_t u = std::bit_cast<uint32_t>(f);
    u += 0x7FFF + ((u >> 16) & 1);
    return uint16_t(u >> 16);
}

inline float from_bf16(uint16_t h) {
    return std::bit_cast<float>(uint32_t(h) << 16);
}

inline float round_bf16(float f) {
    return from_bf16(to_bf16(f));
}

}  // namespace engine::test
