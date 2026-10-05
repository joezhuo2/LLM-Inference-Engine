#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine/kernels/gemm.h"
#include "engine/runtime/device_buffer.h"
#include "kernels/cuda_check.h"
#include "support/bf16.h"

using engine::Blas;
using engine::DeviceBuffer;
using engine::DType;
using engine::Tensor;
using engine::test::from_bf16;
using engine::test::to_bf16;

namespace {

constexpr size_t kWorkspaceBytes = size_t(4) << 20;

std::vector<uint16_t> to_bf16(const std::vector<float>& v) {
    std::vector<uint16_t> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) out[i] = engine::test::to_bf16(v[i]);
    return out;
}

DeviceBuffer upload(const std::vector<uint16_t>& h) {
    DeviceBuffer d(h.size() * sizeof(uint16_t));
    CUDA_CHECK(cudaMemcpy(d.data(), h.data(), d.size(), cudaMemcpyHostToDevice));
    return d;
}

std::vector<float> download(const DeviceBuffer& d) {
    std::vector<uint16_t> h(d.size() / sizeof(uint16_t));
    CUDA_CHECK(cudaMemcpy(h.data(), d.data(), d.size(), cudaMemcpyDeviceToHost));
    std::vector<float> out(h.size());
    for (size_t i = 0; i < h.size(); ++i) out[i] = from_bf16(h[i]);
    return out;
}

class GemmTest : public ::testing::Test {
protected:
    void SetUp() override {
        CUDA_CHECK(cudaStreamCreate(&stream_));
        workspace_ = DeviceBuffer(kWorkspaceBytes);
        blas_ = std::make_unique<Blas>(stream_, workspace_.data(), workspace_.size());
    }

    void TearDown() override {
        blas_.reset();
        CUDA_CHECK(cudaStreamDestroy(stream_));
    }

    std::vector<float> run(const std::vector<uint16_t>& x, const std::vector<uint16_t>& w,
                           int64_t b, int64_t in, int64_t out) {
        DeviceBuffer dx = upload(x), dw = upload(w), dy(size_t(b * out) * sizeof(uint16_t));
        engine::gemm(*blas_, Tensor(dy.data(), DType::BF16, {b, out}),
                     Tensor(dx.data(), DType::BF16, {b, in}),
                     Tensor(dw.data(), DType::BF16, {out, in}));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        return download(dy);
    }

    cudaStream_t stream_ = nullptr;
    DeviceBuffer workspace_;
    std::unique_ptr<Blas> blas_;
};

}  // namespace

TEST(Blas, BindsTheGivenStream) {
    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));
    {
        DeviceBuffer workspace(kWorkspaceBytes);
        Blas blas(s, workspace.data(), workspace.size());
        ASSERT_NE(blas.handle(), nullptr);
        cudaStream_t bound;
        CUBLAS_CHECK(cublasGetStream(blas.handle(), &bound));
        EXPECT_EQ(bound, s);
        EXPECT_EQ(blas.stream(), s);
    }
    CUDA_CHECK(cudaStreamDestroy(s));
}

TEST_F(GemmTest, HandExampleIsExact) {
    const std::vector<float> x = {1, 2, 3, 4, 0, -1, 2, 1, 3, 0, -2, 5};
    const std::vector<float> w = {1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 2, -1, 0, 3, -1, 2, -3, 1};
    const std::vector<float> expected = {1, 2, 10, 12, -2, 0, -1, 2, 4, -7, 3, 0, 6, 21, 8};
    EXPECT_EQ(run(to_bf16(x), to_bf16(w), 3, 4, 5), expected);
}

TEST_F(GemmTest, RejectsMismatchedShapes) {
    DeviceBuffer buf(4096);
    void* p = buf.data();
    EXPECT_THROW(engine::gemm(*blas_, Tensor(p, DType::BF16, {2, 5}),
                              Tensor(p, DType::BF16, {2, 3}), Tensor(p, DType::BF16, {5, 4})),
                 std::invalid_argument);
    EXPECT_THROW(engine::gemm(*blas_, Tensor(p, DType::BF16, {2, 4}),
                              Tensor(p, DType::BF16, {2, 3}), Tensor(p, DType::BF16, {5, 3})),
                 std::invalid_argument);
    EXPECT_THROW(engine::gemm(*blas_, Tensor(p, DType::F32, {2, 5}), Tensor(p, DType::BF16, {2, 3}),
                              Tensor(p, DType::BF16, {5, 3})),
                 std::invalid_argument);
}

struct Shape {
    int64_t b, in, out;
};

class GemmShapeTest : public GemmTest, public ::testing::WithParamInterface<Shape> {};

TEST_P(GemmShapeTest, MatchesCpuReference) {
    const auto [b, in, out] = GetParam();
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<uint16_t> x(size_t(b * in)), w(size_t(out * in));
    for (auto& v : x) v = to_bf16(dist(rng));
    for (auto& v : w) v = to_bf16(dist(rng));

    const std::vector<float> y = run(x, w, b, in, out);

    for (int64_t i = 0; i < b; ++i) {
        for (int64_t j = 0; j < out; ++j) {
            double ref = 0.0, scale = 0.0;
            for (int64_t k = 0; k < in; ++k) {
                const double t = double(from_bf16(x[i * in + k])) * from_bf16(w[j * in + k]);
                ref += t;
                scale += std::fabs(t);
            }
            const double tol = 0x1p-7 * std::fabs(ref) + 1e-4 * scale + 1e-6;
            ASSERT_NEAR(y[i * out + j], ref, tol) << "at row " << i << ", col " << j;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(Shapes, GemmShapeTest,
                         ::testing::Values(Shape{1, 1, 1}, Shape{7, 13, 5}, Shape{17, 129, 33},
                                           Shape{1, 2048, 2560}, Shape{1, 5632, 2048},
                                           Shape{16, 2048, 2560}, Shape{64, 2048, 256},
                                           Shape{1, 2048, 32000}),
                         [](const auto& info) {
                             const Shape& s = info.param;
                             return "B" + std::to_string(s.b) + "_in" + std::to_string(s.in) +
                                    "_out" + std::to_string(s.out);
                         });
