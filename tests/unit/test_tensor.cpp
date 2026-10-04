#include <gtest/gtest.h>

#include <stdexcept>

#include "engine/core/tensor.h"

using engine::DType;
using engine::Tensor;

TEST(DType, Sizes) {
    EXPECT_EQ(engine::dtype_size(DType::BF16), 2u);
    EXPECT_EQ(engine::dtype_size(DType::F16), 2u);
    EXPECT_EQ(engine::dtype_size(DType::F32), 4u);
    EXPECT_EQ(engine::dtype_size(DType::I32), 4u);
    EXPECT_EQ(engine::dtype_size(DType::I64), 8u);
}

TEST(Tensor, NumelAndBytes) {
    Tensor t(nullptr, DType::BF16, {2560, 2048});
    EXPECT_EQ(t.ndim, 2);
    EXPECT_EQ(t.numel(), 2560 * 2048);
    EXPECT_EQ(t.bytes(), 2560u * 2048u * 2u);

    Tensor kv(nullptr, DType::F32, {7, 4, 16, 64});
    EXPECT_EQ(kv.numel(), 7 * 4 * 16 * 64);
    EXPECT_EQ(kv.bytes(), size_t(kv.numel()) * 4);
}

TEST(Tensor, ScalarHasOneElement) {
    Tensor t(nullptr, DType::I32, {});
    EXPECT_EQ(t.ndim, 0);
    EXPECT_EQ(t.numel(), 1);
}

TEST(Tensor, ZeroDimMeansEmpty) {
    Tensor t(nullptr, DType::BF16, {0, 2048});
    EXPECT_EQ(t.numel(), 0);
    EXPECT_EQ(t.bytes(), 0u);
}

TEST(Tensor, RejectsMoreThanFourDims) {
    EXPECT_THROW(Tensor(nullptr, DType::BF16, {1, 2, 3, 4, 5}), std::invalid_argument);
}
