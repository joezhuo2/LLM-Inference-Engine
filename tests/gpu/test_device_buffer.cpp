#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

#include "engine/runtime/device_buffer.h"
#include "kernels/cuda_check.h"

using engine::DeviceBuffer;

static_assert(!std::is_copy_constructible_v<DeviceBuffer>);
static_assert(!std::is_copy_assignable_v<DeviceBuffer>);
static_assert(std::is_nothrow_move_constructible_v<DeviceBuffer>);
static_assert(std::is_nothrow_move_assignable_v<DeviceBuffer>);

TEST(DeviceBuffer, AllocatesRequestedSize) {
    DeviceBuffer b(1 << 20);
    EXPECT_NE(b.data(), nullptr);
    EXPECT_EQ(b.size(), size_t(1) << 20);
}

TEST(DeviceBuffer, ZeroSizeHasNoAllocation) {
    DeviceBuffer b(0);
    EXPECT_EQ(b.data(), nullptr);
    EXPECT_EQ(b.size(), 0u);
}

TEST(DeviceBuffer, RoundTripsBytes) {
    std::vector<int32_t> in(4096), out(in.size());
    std::iota(in.begin(), in.end(), -100);
    DeviceBuffer b(in.size() * sizeof(int32_t));
    CUDA_CHECK(cudaMemcpy(b.data(), in.data(), b.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(out.data(), b.data(), b.size(), cudaMemcpyDeviceToHost));
    EXPECT_EQ(in, out);
}

TEST(DeviceBuffer, MoveConstructTransfersOwnership) {
    DeviceBuffer a(256);
    void* p = a.data();
    DeviceBuffer b(std::move(a));
    EXPECT_EQ(b.data(), p);
    EXPECT_EQ(b.size(), 256u);
    EXPECT_EQ(a.data(), nullptr);
    EXPECT_EQ(a.size(), 0u);
}

TEST(DeviceBuffer, MoveAssignTakesOverAndEmptiesSource) {
    DeviceBuffer a(256), b(512);
    void* p = b.data();
    a = std::move(b);
    EXPECT_EQ(a.data(), p);
    EXPECT_EQ(a.size(), 512u);
    EXPECT_EQ(b.data(), nullptr);
}

TEST(DeviceBuffer, SelfMoveAssignKeepsAllocation) {
    DeviceBuffer a(256);
    void* p = a.data();
    DeviceBuffer& alias = a;
    a = std::move(alias);
    EXPECT_EQ(a.data(), p);
    EXPECT_EQ(a.size(), 256u);
}

TEST(DeviceBuffer, DebugBuildsFillWithNaN) {
#ifdef NDEBUG
    GTEST_SKIP() << "NaN fill is debug only";
#else
    DeviceBuffer b(64);
    std::vector<uint32_t> words(16);
    CUDA_CHECK(cudaMemcpy(words.data(), b.data(), b.size(), cudaMemcpyDeviceToHost));
    for (uint32_t w : words) {
        EXPECT_TRUE(std::isnan(std::bit_cast<float>(w)));
        EXPECT_EQ(w >> 16, 0xFFFFu);
    }
#endif
}
