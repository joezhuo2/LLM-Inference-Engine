#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

#include "support/bf16.h"

namespace engine::test {

inline std::vector<int32_t> argmax_rows(const std::vector<uint16_t>& logits, int64_t vocab) {
    const int64_t rows = int64_t(logits.size()) / vocab;
    std::vector<int32_t> ids(size_t(rows), 0);
    for (int64_t t = 0; t < rows; ++t) {
        const uint16_t* row = logits.data() + t * vocab;
        for (int64_t i = 1; i < vocab; ++i)
            if (from_bf16(row[i]) > from_bf16(row[ids[size_t(t)]])) ids[size_t(t)] = int32_t(i);
    }
    return ids;
}

inline size_t count_equal(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
    size_t n = 0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) n += a[i] == b[i];
    return n;
}

inline double mean_abs_diff(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        sum += std::abs(double(from_bf16(a[i])) - from_bf16(b[i]));
    return a.empty() ? 0.0 : sum / double(a.size());
}

inline double relative_l2(const std::vector<uint16_t>& actual,
                          const std::vector<uint16_t>& expected) {
    double diff = 0.0, norm = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double e = from_bf16(expected[i]);
        const double d = from_bf16(actual[i]) - e;
        diff += d * d;
        norm += e * e;
    }
    return std::sqrt(diff / norm);
}

}  // namespace engine::test
