#include <gtest/gtest.h>

#include "engine/core/version.h"

TEST(Version, IsSet) {
    EXPECT_FALSE(engine::version().empty());
}
