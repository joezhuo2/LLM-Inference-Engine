#include <gtest/gtest.h>

#include <climits>
#include <cstddef>
#include <stdexcept>

#include "engine/kv/cache_size.h"
#include "engine/model/config.h"

using engine::kv_block_bytes;
using engine::kv_cache_blocks;
using engine::ModelConfig;

namespace {

ModelConfig tinyllama() {
    ModelConfig c;
    c.hidden = 2048;
    c.layers = 22;
    c.heads = 32;
    c.kv_heads = 4;
    c.head_dim = 64;
    return c;
}

ModelConfig tiny() {
    ModelConfig c;
    c.layers = 1;
    c.kv_heads = 1;
    c.head_dim = 1;
    return c;
}

constexpr size_t kGiB = size_t(1) << 30;

}  // namespace

TEST(KvCacheSize, TinyLlamaBlockHoldsSixteenTokensOfEveryLayer) {
    EXPECT_EQ(kv_block_bytes(tinyllama(), 16), 360'448);
    EXPECT_EQ(kv_block_bytes(tinyllama(), 1), 22'528);
}

TEST(KvCacheSize, TakesTheFractionOfFreeMemory) {
    EXPECT_EQ(kv_cache_blocks(tinyllama(), 16, 4 * kGiB), 10'724);
    EXPECT_EQ(kv_cache_blocks(tinyllama(), 16, 4 * kGiB, 1.0), 11'915);
    EXPECT_EQ(kv_cache_blocks(tinyllama(), 16, 4 * kGiB, 0.5), 5'957);
}

TEST(KvCacheSize, NeedsTheNullBlockAndOneMore) {
    const int64_t block = kv_block_bytes(tinyllama(), 16);
    EXPECT_EQ(kv_cache_blocks(tinyllama(), 16, size_t(2 * block), 1.0), 2);
    EXPECT_EQ(kv_cache_blocks(tinyllama(), 16, size_t(4 * block), 0.5), 2);
    EXPECT_THROW(kv_cache_blocks(tinyllama(), 16, size_t(2 * block - 1), 1.0), std::runtime_error);
    EXPECT_THROW(kv_cache_blocks(tinyllama(), 16, 0), std::runtime_error);
}

TEST(KvCacheSize, CapsTheBlockCountAtIntMax) {
    EXPECT_EQ(kv_cache_blocks(tiny(), 1, size_t(1) << 40, 1.0), INT_MAX);
}

TEST(KvCacheSize, RejectsBadArguments) {
    EXPECT_THROW(kv_block_bytes(tinyllama(), 0), std::invalid_argument);
    EXPECT_THROW(kv_cache_blocks(tinyllama(), 0, 4 * kGiB), std::invalid_argument);
    EXPECT_THROW(kv_cache_blocks(tinyllama(), 16, 4 * kGiB, 0.0), std::invalid_argument);
    EXPECT_THROW(kv_cache_blocks(tinyllama(), 16, 4 * kGiB, 1.5), std::invalid_argument);
    EXPECT_THROW(kv_cache_blocks(tinyllama(), 16, 4 * kGiB, -0.5), std::invalid_argument);
}
