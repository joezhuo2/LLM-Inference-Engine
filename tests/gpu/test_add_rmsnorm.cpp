#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine/kernels/add.h"
#include "engine/kernels/add_rmsnorm.h"
#include "engine/kernels/rmsnorm.h"
#include "support/bf16.h"
#include "support/device.h"
#include "support/guarded.h"

using engine::DeviceBuffer;
using engine::DType;
using engine::Tensor;
using engine::fast::add_rmsnorm;
using engine::test::close_up_to_reduction_order;
using engine::test::compare_bf16;
using engine::test::CudaStream;
using engine::test::download;
using engine::test::from_bf16;
using engine::test::GuardedBf16;
using engine::test::random_bf16;
using engine::test::to_bf16;
using engine::test::upload;

namespace {

constexpr float kEps = 1e-5f;

std::vector<uint16_t> cpu_add(const std::vector<uint16_t>& x, const std::vector<uint16_t>& r) {
    std::vector<uint16_t> out(x.size());
    for (size_t i = 0; i < x.size(); ++i) out[i] = to_bf16(from_bf16(x[i]) + from_bf16(r[i]));
    return out;
}

std::vector<uint16_t> cpu_rmsnorm(const std::vector<uint16_t>& x, const std::vector<uint16_t>& w,
                                  int64_t hidden, float eps) {
    std::vector<uint16_t> out(x.size());
    for (size_t row = 0; row < x.size(); row += size_t(hidden)) {
        double sum = 0.0;
        for (int64_t i = 0; i < hidden; ++i)
            sum += double(from_bf16(x[row + i])) * from_bf16(x[row + i]);
        const float rstd = 1.0f / std::sqrt(float(sum / double(hidden)) + eps);
        for (int64_t i = 0; i < hidden; ++i) {
            const float normed = from_bf16(to_bf16(from_bf16(x[row + i]) * rstd));
            out[row + i] = to_bf16(from_bf16(w[i]) * normed);
        }
    }
    return out;
}

struct Result {
    std::vector<uint16_t> sum, norm;
};

Result run(const std::vector<uint16_t>& x, const std::vector<uint16_t>& r,
           const std::vector<uint16_t>& w, int64_t hidden, size_t shift) {
    const int64_t tokens = int64_t(x.size()) / hidden;
    CudaStream stream;
    GuardedBf16 gx(x, shift, false), gr(r, shift, false), gw(w, shift, false);
    GuardedBf16 gsum(x, shift, true), gnorm(x, shift, true);
    add_rmsnorm(gsum.tensor({tokens, hidden}), gnorm.tensor({tokens, hidden}),
                gx.tensor({tokens, hidden}), gr.tensor({tokens, hidden}), gw.tensor({hidden}), kEps,
                stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    EXPECT_TRUE(gx.guards_intact() && gr.guards_intact() && gw.guards_intact());
    EXPECT_TRUE(gsum.guards_intact() && gnorm.guards_intact()) << "a write outside the outputs";
    EXPECT_EQ(gx.values(), x) << "x was modified";
    EXPECT_EQ(gr.values(), r) << "residual was modified";
    EXPECT_EQ(gw.values(), w) << "weight was modified";
    return {gsum.values(), gnorm.values()};
}

struct Shape {
    int64_t tokens, hidden;
    size_t shift;
};

class AddRmsnormShapeTest : public ::testing::TestWithParam<Shape> {};

}  // namespace

TEST_P(AddRmsnormShapeTest, MatchesReference) {
    const auto [tokens, hidden, shift] = GetParam();
    const size_t n = size_t(tokens * hidden);
    const auto x = random_bf16(n, 21, -16.0f, 16.0f);
    const auto r = random_bf16(n, 22, -16.0f, 16.0f);
    const auto w = random_bf16(size_t(hidden), 23, -2.0f, 2.0f);
    const auto got = run(x, r, w, hidden, shift);
    const auto sum = cpu_add(x, r);
    EXPECT_EQ(compare_bf16(got.sum, sum).mismatches, 0u) << compare_bf16(got.sum, sum);
    const auto diff = compare_bf16(got.norm, cpu_rmsnorm(sum, w, hidden, kEps));
    EXPECT_TRUE(close_up_to_reduction_order(diff, n)) << diff;
}

INSTANTIATE_TEST_SUITE_P(Shapes, AddRmsnormShapeTest,
                         ::testing::Values(Shape{1, 1, 0}, Shape{1, 2, 0}, Shape{3, 8, 0},
                                           Shape{3, 100, 0}, Shape{7, 300, 0}, Shape{1, 2047, 0},
                                           Shape{2, 2049, 0}, Shape{1, 2048, 0}, Shape{16, 2048, 0},
                                           Shape{64, 2048, 0}, Shape{5, 5632, 0}, Shape{2, 8192, 0},
                                           Shape{3, 24, 0}, Shape{4, 264, 0}, Shape{1, 16384, 0},
                                           Shape{1, 1, 1}, Shape{3, 100, 1}, Shape{1, 2047, 1},
                                           Shape{1, 2048, 1}, Shape{16, 2048, 1},
                                           Shape{5, 5632, 1}),
                         [](const auto& info) {
                             return "T" + std::to_string(info.param.tokens) + "_hidden" +
                                    std::to_string(info.param.hidden) + "_shift" +
                                    std::to_string(info.param.shift);
                         });

TEST(AddRmsnorm, MatchesTheUnfusedNaivePairOnTheGpu) {
    const int64_t tokens = 16, hidden = 2048;
    const size_t n = size_t(tokens * hidden);
    const auto x = random_bf16(n, 31, -8.0f, 8.0f);
    const auto r = random_bf16(n, 32, -8.0f, 8.0f);
    const auto w = random_bf16(size_t(hidden), 33, 0.0f, 2.0f);
    const auto got = run(x, r, w, hidden, 0);

    CudaStream stream;
    DeviceBuffer dx = upload(x), dr = upload(r), dw = upload(w), dnorm(n * 2);
    const Tensor tx(dx.data(), DType::BF16, {tokens, hidden});
    engine::naive::add(tx, tx, Tensor(dr.data(), DType::BF16, {tokens, hidden}), stream);
    engine::naive::rmsnorm(Tensor(dnorm.data(), DType::BF16, {tokens, hidden}), tx,
                           Tensor(dw.data(), DType::BF16, {hidden}), kEps, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    EXPECT_EQ(compare_bf16(got.sum, download<uint16_t>(dx)).mismatches, 0u);
    const auto diff = compare_bf16(got.norm, download<uint16_t>(dnorm));
    EXPECT_TRUE(close_up_to_reduction_order(diff, n)) << diff;
}

TEST(AddRmsnorm, SumMayAliasXOrResidual) {
    const int64_t tokens = 8, hidden = 2048;
    const size_t n = size_t(tokens * hidden);
    const auto x = random_bf16(n, 41, -8.0f, 8.0f);
    const auto r = random_bf16(n, 42, -8.0f, 8.0f);
    const auto w = random_bf16(size_t(hidden), 43, -2.0f, 2.0f);
    const auto want = run(x, r, w, hidden, 0);
    for (const bool alias_x : {true, false}) {
        SCOPED_TRACE(alias_x ? "sum aliases x" : "sum aliases residual");
        CudaStream stream;
        GuardedBf16 gx(x, 0, false), gr(r, 0, false), gw(w, 0, false), gnorm(x, 0, true);
        GuardedBf16& target = alias_x ? gx : gr;
        const Tensor sum = target.tensor({tokens, hidden});
        add_rmsnorm(sum, gnorm.tensor({tokens, hidden}), gx.tensor({tokens, hidden}),
                    gr.tensor({tokens, hidden}), gw.tensor({hidden}), kEps, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        EXPECT_EQ(target.values(), want.sum);
        EXPECT_EQ(gnorm.values(), want.norm);
        EXPECT_EQ((alias_x ? gr : gx).values(), alias_x ? r : x);
        EXPECT_TRUE(gx.guards_intact() && gr.guards_intact() && gnorm.guards_intact());
    }
}

TEST(AddRmsnorm, RowResultsDoNotDependOnTheBatch) {
    const int64_t tokens = 16, hidden = 2048;
    const size_t n = size_t(tokens * hidden);
    const auto x = random_bf16(n, 51, -8.0f, 8.0f);
    const auto r = random_bf16(n, 52, -8.0f, 8.0f);
    const auto w = random_bf16(size_t(hidden), 53, -2.0f, 2.0f);
    const auto batch = run(x, r, w, hidden, 0);
    for (const int64_t row : {int64_t(0), int64_t(5), tokens - 1}) {
        SCOPED_TRACE("row " + std::to_string(row));
        const auto first = x.begin() + row * hidden;
        const auto first_r = r.begin() + row * hidden;
        const auto solo = run({first, first + hidden}, {first_r, first_r + hidden}, w, hidden, 0);
        const auto sum_begin = batch.sum.begin() + row * hidden;
        const auto norm_begin = batch.norm.begin() + row * hidden;
        EXPECT_EQ(solo.sum, std::vector<uint16_t>(sum_begin, sum_begin + hidden));
        EXPECT_EQ(solo.norm, std::vector<uint16_t>(norm_begin, norm_begin + hidden));
    }
}

TEST(AddRmsnorm, IsDeterministic) {
    const int64_t tokens = 4, hidden = 5632;
    const size_t n = size_t(tokens * hidden);
    const auto x = random_bf16(n, 61, -8.0f, 8.0f);
    const auto r = random_bf16(n, 62, -8.0f, 8.0f);
    const auto w = random_bf16(size_t(hidden), 63, -2.0f, 2.0f);
    const auto first = run(x, r, w, hidden, 0);
    for (int i = 0; i < 3; ++i) {
        const auto again = run(x, r, w, hidden, 0);
        EXPECT_EQ(again.sum, first.sum);
        EXPECT_EQ(again.norm, first.norm);
    }
}

TEST(AddRmsnorm, AnAllZeroRowGivesZeros) {
    const int64_t hidden = 2048;
    std::vector<uint16_t> x(size_t(2 * hidden), 0), r(size_t(2 * hidden), 0);
    for (int64_t i = 0; i < hidden; ++i) x[size_t(hidden + i)] = to_bf16(float(i % 7) - 3.0f);
    const auto w = random_bf16(size_t(hidden), 71, 0.5f, 2.0f);
    const auto got = run(x, r, w, hidden, 0);
    for (int64_t i = 0; i < hidden; ++i) {
        ASSERT_EQ(from_bf16(got.sum[size_t(i)]), 0.0f);
        ASSERT_EQ(from_bf16(got.norm[size_t(i)]), 0.0f) << "column " << i;
    }
}

TEST(AddRmsnorm, ZeroTokensIsANoOp) {
    DeviceBuffer w(8 * sizeof(uint16_t));
    CudaStream stream;
    const Tensor empty(nullptr, DType::BF16, {0, 8});
    add_rmsnorm(empty, empty, empty, empty, Tensor(w.data(), DType::BF16, {8}), kEps, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TEST(AddRmsnorm, RejectsBadArguments) {
    DeviceBuffer buf(1 << 16);
    void* p = buf.data();
    CudaStream stream;
    const auto call = [&](const Tensor& sum, const Tensor& norm, const Tensor& x,
                          const Tensor& residual, const Tensor& weight) {
        add_rmsnorm(sum, norm, x, residual, weight, kEps, stream);
    };
    const Tensor ok(p, DType::BF16, {3, 8});
    const Tensor w(p, DType::BF16, {8});
    EXPECT_THROW(call(Tensor(p, DType::F32, {3, 8}), ok, ok, ok, w), std::invalid_argument);
    EXPECT_THROW(call(ok, Tensor(p, DType::F32, {3, 8}), ok, ok, w), std::invalid_argument);
    EXPECT_THROW(call(ok, ok, Tensor(p, DType::F32, {3, 8}), ok, w), std::invalid_argument);
    EXPECT_THROW(call(ok, ok, ok, Tensor(p, DType::F32, {3, 8}), w), std::invalid_argument);
    EXPECT_THROW(call(ok, ok, ok, ok, Tensor(p, DType::F32, {8})), std::invalid_argument);
    EXPECT_THROW(call(ok, ok, Tensor(p, DType::BF16, {24}), ok, w), std::invalid_argument);
    EXPECT_THROW(call(ok, ok, ok, Tensor(p, DType::BF16, {2, 8}), w), std::invalid_argument);
    EXPECT_THROW(call(Tensor(p, DType::BF16, {3, 4}), ok, ok, ok, w), std::invalid_argument);
    EXPECT_THROW(call(ok, Tensor(p, DType::BF16, {4, 8}), ok, ok, w), std::invalid_argument);
    EXPECT_THROW(call(ok, ok, ok, ok, Tensor(p, DType::BF16, {7})), std::invalid_argument);
    EXPECT_THROW(call(ok, ok, ok, ok, Tensor(p, DType::BF16, {1, 8})), std::invalid_argument);
}
