#pragma once

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <vector>

#include "engine/loader/checksum.h"
#include "support/paths.h"

namespace engine::test {

inline std::filesystem::path golden_checksums_path() {
    return golden_dir() / "checksums.json";
}

inline std::vector<TensorChecksum> load_golden_checksums() {
    std::ifstream in(golden_checksums_path());
    return checksums_from_json(nlohmann::json::parse(in));
}

inline void expect_matches_golden(const std::vector<TensorChecksum>& actual,
                                  const std::vector<TensorChecksum>& golden) {
    ASSERT_EQ(actual.size(), golden.size());
    for (size_t i = 0; i < golden.size(); ++i) {
        const TensorChecksum& a = actual[i];
        const TensorChecksum& g = golden[i];
        SCOPED_TRACE(g.name);
        ASSERT_EQ(a.name, g.name);
        EXPECT_EQ(a.dtype, g.dtype);
        EXPECT_EQ(a.shape, g.shape);
        EXPECT_LE(std::fabs(a.sum - g.sum), 1e-6 * g.abs_sum);
        EXPECT_LE(std::fabs(a.abs_sum - g.abs_sum), 1e-6 * g.abs_sum);
    }
}

}  // namespace engine::test
