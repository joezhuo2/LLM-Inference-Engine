#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine/kernels/gemm_fp8.h"
#include "kernels/cuda_check.h"
#include "support/bf16.h"
#include "support/device.h"
#include "support/fp8.h"

using engine::DeviceBuffer;
using engine::DType;
using engine::gemm_fp8;
using engine::Tensor;
using engine::test::CudaStream;
using engine::test::download;
using engine::test::from_bf16;
using engine::test::from_fp8;
using engine::test::random_bf16;
using engine::test::to_bf16;
using engine::test::to_fp8;
using engine::test::upload;

namespace {

constexpr int64_t kGuard = 256;

struct Problem {
    int64_t b, in, out;
    std::vector<uint16_t> x;
    std::vector<uint8_t> w;
    std::vector<float> row_scale;
};

Problem make_problem(int64_t b, int64_t in, int64_t out, uint32_t seed, float weight_range = 1.0f) {
    std::mt19937 rng(seed);
    Problem p{b, in, out, random_bf16(size_t(b * in), seed + 1), {}, {}};
    p.w.resize(size_t(out * in));
    p.row_scale.resize(size_t(out));
    std::vector<float> row(static_cast<size_t>(in));
    for (int64_t o = 0; o < out; ++o) {
        const float reach = weight_range * std::uniform_real_distribution<float>(0.01f, 1.0f)(rng);
        std::uniform_real_distribution<float> dist(-reach, reach);
        float largest = 0.0f;
        for (auto& v : row) largest = std::max(largest, std::abs(v = dist(rng)));
        const float scale = largest > 0.0f ? largest / 448.0f : 1.0f;
        p.row_scale[size_t(o)] = scale;
        for (int64_t i = 0; i < in; ++i) p.w[size_t(o * in + i)] = to_fp8(row[size_t(i)] / scale);
    }
    return p;
}

std::vector<uint16_t> run(const Problem& p, uint32_t seed) {
    const int64_t n = p.b * p.out;
    const auto init = random_bf16(size_t(n + 2 * kGuard), seed);
    DeviceBuffer y = upload(init);
    DeviceBuffer x = upload(p.x.empty() ? std::vector<uint16_t>{0} : p.x);
    DeviceBuffer w = upload(p.w), scale = upload(p.row_scale);

    CudaStream stream;
    gemm_fp8(Tensor(static_cast<uint16_t*>(y.data()) + kGuard, DType::BF16, {p.b, p.out}),
             Tensor(x.data(), DType::BF16, {p.b, p.in}),
             Tensor(w.data(), DType::F8E4M3, {p.out, p.in}),
             Tensor(scale.data(), DType::F32, {p.out}), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (!p.x.empty()) {
        EXPECT_EQ(download<uint16_t>(x), p.x) << "x was modified";
    }
    EXPECT_EQ(download<uint8_t>(w), p.w) << "w was modified";
    EXPECT_EQ(download<float>(scale), p.row_scale) << "row_scale was modified";
    const auto after = download<uint16_t>(y);
    for (int64_t i = 0; i < kGuard; ++i) {
        EXPECT_EQ(after[size_t(i)], init[size_t(i)]) << "write before y at " << i - kGuard;
        EXPECT_EQ(after[size_t(kGuard + n + i)], init[size_t(kGuard + n + i)])
            << "write after y at " << i;
    }
    return {after.begin() + kGuard, after.begin() + kGuard + n};
}

void check(const Problem& p, uint32_t seed) {
    const auto y = run(p, seed);
    size_t outside = 0;
    int64_t first_row = -1, first_col = -1;
    for (int64_t i = 0; i < p.b; ++i)
        for (int64_t o = 0; o < p.out; ++o) {
            double ref = 0.0, spread = 0.0;
            for (int64_t k = 0; k < p.in; ++k) {
                const double t = double(from_bf16(p.x[size_t(i * p.in + k)])) *
                                 from_fp8(p.w[size_t(o * p.in + k)]) * p.row_scale[size_t(o)];
                ref += t;
                spread += std::fabs(t);
            }
            const double tol = 0x1p-7 * std::fabs(ref) + 1e-4 * spread + 1e-6;
            const double got = from_bf16(y[size_t(i * p.out + o)]);
            if (!(std::fabs(got - ref) <= tol) && outside++ == 0) {
                first_row = i;
                first_col = o;
            }
        }
    EXPECT_EQ(outside, 0u) << "first at row " << first_row << ", col " << first_col;
}

std::vector<uint16_t> bf16s(const std::vector<float>& v) {
    std::vector<uint16_t> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) out[i] = to_bf16(v[i]);
    return out;
}

std::vector<uint8_t> fp8s(const std::vector<float>& v) {
    std::vector<uint8_t> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) out[i] = to_fp8(v[i]);
    return out;
}

struct Shape {
    int64_t b, in, out;
};

class GemmFp8ShapeTest : public ::testing::TestWithParam<Shape> {};

}  // namespace

TEST(GemmFp8, HandExampleIsExact) {
    const Problem p{3,
                    4,
                    5,
                    bf16s({1, 2, 3, 4, 0, -1, 2, 1, 3, 0, -2, 5}),
                    fp8s({1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 2, -1, 0, 3, -1, 2, -3, 1}),
                    {1.0f, 1.0f, 1.0f, 1.0f, 1.0f}};
    EXPECT_EQ(run(p, 1), bf16s({1, 2, 10, 12, -2, 0, -1, 2, 4, -7, 3, 0, 6, 21, 8}));
}

TEST(GemmFp8, EachRowIsMultipliedByItsOwnScale) {
    const Problem p{3,
                    4,
                    5,
                    bf16s({1, 2, 3, 4, 0, -1, 2, 1, 3, 0, -2, 5}),
                    fp8s({1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 2, -1, 0, 3, -1, 2, -3, 1}),
                    {2.0f, 0.5f, -1.0f, 0.0f, 0.25f}};
    EXPECT_EQ(run(p, 2), bf16s({2, 1, -10, 0, -0.5f, 0, -0.5f, -2, 0, -1.75f, 6, 0, -6, 0, 2}));
}

TEST(GemmFp8, SaturatedAndSmallestWeights) {
    Problem p = make_problem(4, 37, 9, 3);
    std::mt19937 rng(4);
    const uint8_t pick[] = {0x7E, 0xFE, 0x00, 0x80, 0x01, 0x81, 0x38};
    for (auto& b : p.w) b = pick[rng() % 7];
    check(p, 5);
}

TEST(GemmFp8, ANanWeightReachesOnlyItsOwnRow) {
    Problem p = make_problem(2, 40, 6, 6);
    p.w[size_t(3 * 40 + 39)] = 0x7F;
    const auto y = run(p, 7);
    for (int64_t i = 0; i < p.b; ++i)
        for (int64_t o = 0; o < p.out; ++o)
            EXPECT_EQ(std::isnan(from_bf16(y[size_t(i * p.out + o)])), o == 3)
                << "row " << i << ", col " << o;
}

TEST(GemmFp8, TwoIdenticalCallsAgree) {
    const Problem p = make_problem(16, 2048, 256, 8);
    EXPECT_EQ(run(p, 9), run(p, 9));
}

TEST(GemmFp8, NoRowsIsANoOp) {
    Problem p = make_problem(0, 2048, 64, 10);
    EXPECT_TRUE(run(p, 11).empty());
}

TEST(GemmFp8, RejectsBadArguments) {
    DeviceBuffer buf(4096);
    void* p = buf.data();
    const Tensor y(p, DType::BF16, {2, 5});
    const Tensor x(p, DType::BF16, {2, 3});
    const Tensor w(p, DType::F8E4M3, {5, 3});
    const Tensor scale(p, DType::F32, {5});
    const engine::Stream s = nullptr;

    EXPECT_THROW(gemm_fp8(y, x, Tensor(p, DType::BF16, {5, 3}), scale, s), std::invalid_argument);
    EXPECT_THROW(gemm_fp8(y, x, Tensor(p, DType::F8E4M3, {5, 4}), scale, s), std::invalid_argument);
    EXPECT_THROW(gemm_fp8(y, x, Tensor(p, DType::F8E4M3, {15}), scale, s), std::invalid_argument);
    EXPECT_THROW(gemm_fp8(Tensor(p, DType::BF16, {2, 4}), x, w, scale, s), std::invalid_argument);
    EXPECT_THROW(gemm_fp8(Tensor(p, DType::BF16, {3, 5}), x, w, scale, s), std::invalid_argument);
    EXPECT_THROW(gemm_fp8(Tensor(p, DType::F32, {2, 5}), x, w, scale, s), std::invalid_argument);
    EXPECT_THROW(gemm_fp8(Tensor(p, DType::BF16, {10}), x, w, scale, s), std::invalid_argument);
    EXPECT_THROW(gemm_fp8(y, Tensor(p, DType::F8E4M3, {2, 3}), w, scale, s), std::invalid_argument);
    EXPECT_THROW(gemm_fp8(y, Tensor(p, DType::BF16, {6}), w, scale, s), std::invalid_argument);
    EXPECT_THROW(gemm_fp8(y, x, w, Tensor(p, DType::BF16, {5}), s), std::invalid_argument);
    EXPECT_THROW(gemm_fp8(y, x, w, Tensor(p, DType::F32, {4}), s), std::invalid_argument);
    EXPECT_THROW(gemm_fp8(y, x, w, Tensor(p, DType::F32, {5, 1}), s), std::invalid_argument);
}

TEST_P(GemmFp8ShapeTest, MatchesCpuReference) {
    const auto [b, in, out] = GetParam();
    check(make_problem(b, in, out, 1234), 12);
}

TEST_P(GemmFp8ShapeTest, MatchesCpuReferenceWithLargeWeights) {
    const auto [b, in, out] = GetParam();
    if (in * out > (int64_t(1) << 22)) GTEST_SKIP() << "covered by the small weights case";
    check(make_problem(b, in, out, 4321, 300.0f), 13);
}

INSTANTIATE_TEST_SUITE_P(ShapesFp8, GemmFp8ShapeTest,
                         ::testing::Values(Shape{1, 1, 1}, Shape{7, 13, 5}, Shape{17, 129, 33},
                                           Shape{3, 15, 4}, Shape{2, 17, 3}, Shape{1, 2048, 2560},
                                           Shape{1, 5632, 2048}, Shape{1, 2048, 11264},
                                           Shape{16, 2048, 2560}, Shape{64, 2048, 256},
                                           Shape{1, 2048, 32000}),
                         [](const auto& info) {
                             const Shape& s = info.param;
                             return "B" + std::to_string(s.b) + "_in" + std::to_string(s.in) +
                                    "_out" + std::to_string(s.out);
                         });
