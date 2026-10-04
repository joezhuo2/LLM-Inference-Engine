#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <type_traits>
#include <utility>

#include "engine/runtime/device_buffer.h"
#include "engine/runtime/pinned_buffer.h"
#include "kernels/cuda_check.h"

using engine::DeviceBuffer;
using engine::PinnedBuffer;

static_assert(!std::is_copy_constructible_v<PinnedBuffer>);
static_assert(!std::is_copy_assignable_v<PinnedBuffer>);
static_assert(std::is_nothrow_move_constructible_v<PinnedBuffer>);
static_assert(std::is_nothrow_move_assignable_v<PinnedBuffer>);

TEST(PinnedBuffer, IsPageLockedHostMemory) {
    PinnedBuffer b(1 << 20);
    ASSERT_NE(b.data(), nullptr);
    cudaPointerAttributes attr;
    CUDA_CHECK(cudaPointerGetAttributes(&attr, b.data()));
    EXPECT_EQ(attr.type, cudaMemoryTypeHost);
}

TEST(PinnedBuffer, ZeroSizeHasNoAllocation) {
    PinnedBuffer b(0);
    EXPECT_EQ(b.data(), nullptr);
    EXPECT_EQ(b.size(), 0u);
}

TEST(PinnedBuffer, AsyncRoundTripThroughDevice) {
    const size_t n = 1 << 16;
    PinnedBuffer src(n), dst(n);
    auto* s = static_cast<uint8_t*>(src.data());
    for (size_t i = 0; i < n; ++i) s[i] = uint8_t(i * 7);
    std::memset(dst.data(), 0, n);

    DeviceBuffer dev(n);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMemcpyAsync(dev.data(), src.data(), n, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(dst.data(), dev.data(), n, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    EXPECT_EQ(std::memcmp(src.data(), dst.data(), n), 0);
}

TEST(PinnedBuffer, MoveTransfersOwnership) {
    PinnedBuffer a(128), b(256);
    void* p = b.data();
    a = std::move(b);
    EXPECT_EQ(a.data(), p);
    EXPECT_EQ(a.size(), 256u);
    EXPECT_EQ(b.data(), nullptr);

    PinnedBuffer c(std::move(a));
    EXPECT_EQ(c.data(), p);
    EXPECT_EQ(a.data(), nullptr);
    EXPECT_EQ(a.size(), 0u);
}
