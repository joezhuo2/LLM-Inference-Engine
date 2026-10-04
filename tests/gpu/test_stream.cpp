#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <type_traits>

#include "engine/core/stream.h"

static_assert(std::is_same_v<engine::Stream, cudaStream_t>);

TEST(Stream, NullIsTheLegacyDefaultStream) {
    engine::Stream s = nullptr;
    EXPECT_EQ(s, cudaStream_t{0});
}
