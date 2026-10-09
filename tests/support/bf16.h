#pragma once

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <random>
#include <vector>

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

inline std::vector<uint16_t> random_bf16(size_t n, uint32_t seed, float lo = -1.0f,
                                         float hi = 1.0f) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<uint16_t> out(n);
    for (auto& v : out) v = to_bf16(dist(rng));
    return out;
}

inline int bf16_ulps(uint16_t a, uint16_t b) {
    const auto key = [](uint16_t h) { return h & 0x8000 ? -int(h & 0x7FFF) : int(h); };
    return std::abs(key(a) - key(b));
}

struct Bf16Diff {
    size_t mismatches = 0;
    int max_ulps = 0;
    size_t first = 0;
};

inline Bf16Diff compare_bf16(const std::vector<uint16_t>& actual,
                             const std::vector<uint16_t>& expected) {
    Bf16Diff d;
    for (size_t i = 0; i < actual.size() && i < expected.size(); ++i) {
        const int ulps = bf16_ulps(actual[i], expected[i]);
        if (ulps == 0) continue;
        if (d.mismatches++ == 0) d.first = i;
        if (ulps > d.max_ulps) d.max_ulps = ulps;
    }
    return d;
}

inline std::ostream& operator<<(std::ostream& os, const Bf16Diff& d) {
    return os << d.mismatches << " mismatches, max " << d.max_ulps << " ulps, first at index "
              << d.first;
}

}  // namespace engine::test
