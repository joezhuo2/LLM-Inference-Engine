#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "engine/kv/cache_size.h"
#include "engine/model/config.h"
#include "engine/runtime/kv_cache.h"
#include "kernels/cuda_check.h"

using engine::device_free_bytes;
using engine::DType;
using engine::kv_block_bytes;
using engine::kv_cache_blocks;
using engine::KvCache;
using engine::ModelConfig;
using engine::Tensor;

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

ModelConfig small() {
    ModelConfig c;
    c.layers = 3;
    c.kv_heads = 2;
    c.head_dim = 8;
    return c;
}

constexpr size_t kMiB = size_t(1) << 20;

}  // namespace

TEST(KvCache, EveryLayerHasBlockMajorKAndV) {
    KvCache cache(tinyllama(), 10);
    EXPECT_EQ(cache.layers(), 22);
    EXPECT_EQ(cache.num_blocks(), 10);
    EXPECT_EQ(cache.block_size(), 16);
    EXPECT_EQ(cache.bytes(), size_t(10 * kv_block_bytes(tinyllama(), 16)));
    for (int l = 0; l < cache.layers(); ++l) {
        for (const Tensor& t : {cache.k(l), cache.v(l)}) {
            EXPECT_EQ(t.dtype, DType::BF16);
            ASSERT_EQ(t.ndim, 4);
            EXPECT_EQ(t.shape[0], 10);
            EXPECT_EQ(t.shape[1], 4);
            EXPECT_EQ(t.shape[2], 16);
            EXPECT_EQ(t.shape[3], 64);
        }
    }
}

TEST(KvCache, LayersTileTheBufferWithoutOverlap) {
    KvCache cache(small(), 5, 4);
    std::vector<std::byte*> starts;
    for (int l = 0; l < cache.layers(); ++l) {
        starts.push_back(static_cast<std::byte*>(cache.k(l).data));
        starts.push_back(static_cast<std::byte*>(cache.v(l).data));
    }
    const size_t plane = cache.k(0).bytes();
    EXPECT_EQ(plane * starts.size(), cache.bytes());
    std::vector<std::byte*> sorted = starts;
    std::sort(sorted.begin(), sorted.end());
    EXPECT_EQ(sorted, starts);
    for (size_t i = 1; i < starts.size(); ++i) EXPECT_EQ(size_t(starts[i] - starts[i - 1]), plane);
}

TEST(KvCache, WritingOnePlaneLeavesTheOthers) {
    KvCache cache(small(), 3, 2);
    for (int l = 0; l < cache.layers(); ++l) {
        CUDA_CHECK(cudaMemset(cache.k(l).data, 2 * l + 1, cache.k(l).bytes()));
        CUDA_CHECK(cudaMemset(cache.v(l).data, 2 * l + 2, cache.v(l).bytes()));
    }
    for (int l = 0; l < cache.layers(); ++l) {
        for (int which = 0; which < 2; ++which) {
            const Tensor t = which == 0 ? cache.k(l) : cache.v(l);
            std::vector<uint8_t> host(t.bytes());
            CUDA_CHECK(cudaMemcpy(host.data(), t.data, t.bytes(), cudaMemcpyDeviceToHost));
            EXPECT_TRUE(std::all_of(host.begin(), host.end(),
                                    [&](uint8_t b) { return b == 2 * l + which + 1; }))
                << "layer " << l << (which == 0 ? " K" : " V");
        }
    }
}

TEST(KvCache, AllocationShowsInFreeMemory) {
    const size_t before = device_free_bytes();
    {
        KvCache cache(tinyllama(), 200);
        const size_t during = device_free_bytes();
        ASSERT_LT(during, before);
        EXPECT_GE(before - during, cache.bytes());
        EXPECT_LE(before - during, cache.bytes() + 16 * kMiB);
    }
    EXPECT_GE(device_free_bytes() + 16 * kMiB, before);
}

TEST(KvCache, FitsInAFractionOfFreeMemory) {
    const size_t free = device_free_bytes();
    const int blocks = kv_cache_blocks(tinyllama(), 16, free, 0.25);
    KvCache cache(tinyllama(), blocks);
    EXPECT_LE(cache.bytes(), free / 4);
    EXPECT_GT(cache.bytes() + size_t(kv_block_bytes(tinyllama(), 16)), free / 4);
}

TEST(KvCache, RejectsBadArguments) {
    EXPECT_THROW(KvCache(small(), 1), std::invalid_argument);
    EXPECT_THROW(KvCache(small(), 4, 0), std::invalid_argument);
    KvCache cache(small(), 2);
    EXPECT_THROW(cache.k(-1), std::invalid_argument);
    EXPECT_THROW(cache.v(3), std::invalid_argument);
}
