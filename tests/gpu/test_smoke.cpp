#include <gtest/gtest.h>

#include "engine/kernels/smoke.h"

TEST(Smoke, VectorAdd) {
    EXPECT_EQ(engine::kernels::smoke_vector_add(1 << 20), 0.0f);
}
