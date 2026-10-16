#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "engine/kernels/swiglu.h"
#include "support/bf16.h"
#include "support/device.h"
#include "support/guarded.h"

using engine::DeviceBuffer;
using engine::DType;
using engine::Tensor;
using engine::test::CudaStream;
using engine::test::download;
using engine::test::from_bf16;
using engine::test::GuardedBf16;
using engine::test::random_bf16;
using engine::test::to_bf16;
using engine::test::upload;

namespace {

bool same_bits_or_both_nan(uint16_t a, uint16_t b) {
    const bool a_nan = std::isnan(from_bf16(a)), b_nan = std::isnan(from_bf16(b));
    return a == b || (a_nan && b_nan);
}

size_t count_mismatches(const std::vector<uint16_t>& got, const std::vector<uint16_t>& want,
                        size_t* first) {
    size_t mismatches = 0;
    for (size_t i = 0; i < got.size(); ++i)
        if (!same_bits_or_both_nan(got[i], want[i]) && mismatches++ == 0) *first = i;
    return mismatches;
}

void check(int64_t tokens, int64_t ff, const std::vector<uint16_t>& gate_up, size_t shift) {
    CudaStream stream;
    GuardedBf16 in_naive(gate_up, shift, false), in_fast(gate_up, shift, false);
    const std::vector<uint16_t> out_shape(size_t(tokens * ff), 0);
    GuardedBf16 want(out_shape, shift, true), got(out_shape, shift, true);
    engine::naive::swiglu(want.tensor({tokens, ff}), in_naive.tensor({tokens, 2 * ff}), stream);
    engine::fast::swiglu(got.tensor({tokens, ff}), in_fast.tensor({tokens, 2 * ff}), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    size_t first = 0;
    EXPECT_EQ(count_mismatches(got.values(), want.values(), &first), 0u)
        << "first mismatch at row " << first / size_t(ff) << ", column " << first % size_t(ff);
    EXPECT_TRUE(got.guards_intact()) << "a write outside out";
    EXPECT_TRUE(in_fast.guards_intact());
    EXPECT_EQ(in_fast.values(), gate_up) << "gate_up was modified";
}

struct Shape {
    int64_t tokens, ff;
    size_t shift;
};

class SwigluShapeTest : public ::testing::TestWithParam<Shape> {};

}  // namespace

TEST_P(SwigluShapeTest, MatchesNaive) {
    const auto [tokens, ff, shift] = GetParam();
    check(tokens, ff, random_bf16(size_t(tokens * 2 * ff), 17, -12.0f, 12.0f), shift);
}

INSTANTIATE_TEST_SUITE_P(Shapes, SwigluShapeTest,
                         ::testing::Values(Shape{1, 1, 0}, Shape{1, 3, 0}, Shape{1, 7, 0},
                                           Shape{1, 8, 0}, Shape{1, 9, 0}, Shape{3, 100, 0},
                                           Shape{7, 300, 0}, Shape{1, 5631, 0}, Shape{1, 5632, 0},
                                           Shape{16, 5632, 0}, Shape{64, 5632, 0}, Shape{1, 1, 1},
                                           Shape{3, 100, 1}, Shape{1, 5632, 1}, Shape{16, 5632, 1},
                                           Shape{4, 5631, 1}, Shape{4, 5632, 3}, Shape{70000, 8, 0},
                                           Shape{70000, 3, 0}),
                         [](const auto& info) {
                             return "T" + std::to_string(info.param.tokens) + "_ff" +
                                    std::to_string(info.param.ff) + "_shift" +
                                    std::to_string(info.param.shift);
                         });

TEST(FastSwiglu, MatchesNaiveOnLargeMagnitudes) {
    const int64_t tokens = 8, ff = 5632;
    check(tokens, ff, random_bf16(size_t(tokens * 2 * ff), 71, -100.0f, 100.0f), 0);
    check(tokens, ff, random_bf16(size_t(tokens * 2 * ff), 72, -1e-3f, 1e-3f), 0);
}

TEST(FastSwiglu, MatchesNaiveOnSpecialValues) {
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const std::vector<float> values = {0.0f,  -0.0f,  1.0f,    -1.0f,  88.0f,  -88.0f,
                                       89.0f, -89.0f, -100.0f, 1e30f,  -1e30f, inf,
                                       -inf,  nan,    1e-38f,  -1e-38f};
    const int64_t ff = int64_t(values.size());
    std::vector<uint16_t> gate_up;
    for (size_t g = 0; g < values.size(); ++g) {
        for (const float v : values) gate_up.push_back(to_bf16(v));
        for (size_t j = 0; j < values.size(); ++j)
            gate_up.push_back(to_bf16(values[(g + j) % values.size()]));
    }
    check(ff, ff, gate_up, 0);
    check(ff, ff, gate_up, 1);
}

TEST(FastSwiglu, RowResultsDoNotDependOnTheBatch) {
    const int64_t tokens = 16, ff = 5632;
    const auto gate_up = random_bf16(size_t(tokens * 2 * ff), 81, -12.0f, 12.0f);
    CudaStream stream;
    DeviceBuffer din = upload(gate_up), dout(size_t(tokens * ff) * 2);
    engine::fast::swiglu(Tensor(dout.data(), DType::BF16, {tokens, ff}),
                         Tensor(din.data(), DType::BF16, {tokens, 2 * ff}), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto batch = download<uint16_t>(dout);
    for (const int64_t row : {int64_t(0), int64_t(7), tokens - 1}) {
        SCOPED_TRACE("row " + std::to_string(row));
        const std::vector<uint16_t> one(gate_up.begin() + row * 2 * ff,
                                        gate_up.begin() + (row + 1) * 2 * ff);
        DeviceBuffer d1 = upload(one), o1(size_t(ff) * 2);
        engine::fast::swiglu(Tensor(o1.data(), DType::BF16, {1, ff}),
                             Tensor(d1.data(), DType::BF16, {1, 2 * ff}), stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        EXPECT_EQ(download<uint16_t>(o1),
                  std::vector<uint16_t>(batch.begin() + row * ff, batch.begin() + (row + 1) * ff));
    }
}

TEST(FastSwiglu, IsDeterministic) {
    const int64_t tokens = 4, ff = 5632;
    const auto gate_up = random_bf16(size_t(tokens * 2 * ff), 91, -12.0f, 12.0f);
    CudaStream stream;
    DeviceBuffer din = upload(gate_up);
    std::vector<uint16_t> first;
    for (int i = 0; i < 3; ++i) {
        DeviceBuffer dout(size_t(tokens * ff) * 2);
        engine::fast::swiglu(Tensor(dout.data(), DType::BF16, {tokens, ff}),
                             Tensor(din.data(), DType::BF16, {tokens, 2 * ff}), stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (i == 0) first = download<uint16_t>(dout);
        EXPECT_EQ(download<uint16_t>(dout), first);
    }
}

TEST(FastSwiglu, ZeroTokensIsANoOp) {
    CudaStream stream;
    engine::fast::swiglu(Tensor(nullptr, DType::BF16, {0, 8}),
                         Tensor(nullptr, DType::BF16, {0, 16}), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TEST(FastSwiglu, RejectsBadArguments) {
    DeviceBuffer buf(1 << 14);
    void* p = buf.data();
    CudaStream stream;
    const auto call = [&](const Tensor& out, const Tensor& gate_up) {
        engine::fast::swiglu(out, gate_up, stream);
    };
    const Tensor out(p, DType::BF16, {3, 8});
    const Tensor gate_up(p, DType::BF16, {3, 16});
    EXPECT_THROW(call(Tensor(p, DType::F32, {3, 8}), gate_up), std::invalid_argument);
    EXPECT_THROW(call(out, Tensor(p, DType::F32, {3, 16})), std::invalid_argument);
    EXPECT_THROW(call(out, Tensor(p, DType::BF16, {48})), std::invalid_argument);
    EXPECT_THROW(call(Tensor(p, DType::BF16, {24}), gate_up), std::invalid_argument);
    EXPECT_THROW(call(Tensor(p, DType::BF16, {2, 8}), gate_up), std::invalid_argument);
    EXPECT_THROW(call(Tensor(p, DType::BF16, {3, 9}), gate_up), std::invalid_argument);
    EXPECT_THROW(call(Tensor(p, DType::BF16, {3, 7}), gate_up), std::invalid_argument);
    EXPECT_THROW(call(out, Tensor(p, DType::BF16, {3, 17})), std::invalid_argument);
}
