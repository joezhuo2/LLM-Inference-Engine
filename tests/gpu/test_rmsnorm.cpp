#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine/kernels/rmsnorm.h"
#include "engine/model/config.h"
#include "support/bf16.h"
#include "support/device.h"
#include "support/paths.h"
#include "support/reference.h"
#include "support/safetensors_file.h"

using engine::DeviceBuffer;
using engine::DType;
using engine::Tensor;
using engine::naive::rmsnorm;
using engine::test::compare_bf16;
using engine::test::CudaStream;
using engine::test::download;
using engine::test::from_bf16;
using engine::test::to_bf16;
using engine::test::upload;

namespace {

constexpr float kEps = 1e-5f;

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

std::vector<uint16_t> gpu_rmsnorm(const std::vector<uint16_t>& x, const std::vector<uint16_t>& w,
                                  int64_t hidden, float eps) {
    const int64_t tokens = int64_t(x.size()) / hidden;
    CudaStream stream;
    DeviceBuffer dx = upload(x), dw = upload(w), out(x.size() * sizeof(uint16_t));
    rmsnorm(Tensor(out.data(), DType::BF16, {tokens, hidden}),
            Tensor(dx.data(), DType::BF16, {tokens, hidden}),
            Tensor(dw.data(), DType::BF16, {hidden}), eps, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return download<uint16_t>(out);
}

struct Shape {
    int64_t tokens, hidden;
};

class RmsnormShapeTest : public ::testing::TestWithParam<Shape> {};

}  // namespace

TEST_P(RmsnormShapeTest, MatchesCpuReference) {
    const auto [tokens, hidden] = GetParam();
    const auto x = engine::test::random_bf16(size_t(tokens * hidden), 3, -4.0f, 4.0f);
    const auto w = engine::test::random_bf16(size_t(hidden), 4, -2.0f, 2.0f);
    const auto diff =
        compare_bf16(gpu_rmsnorm(x, w, hidden, kEps), cpu_rmsnorm(x, w, hidden, kEps));
    EXPECT_EQ(diff.mismatches, 0u) << diff;
}

INSTANTIATE_TEST_SUITE_P(Shapes, RmsnormShapeTest,
                         ::testing::Values(Shape{1, 1}, Shape{3, 100}, Shape{7, 300},
                                           Shape{1, 2048}, Shape{64, 2048}, Shape{5, 5632}),
                         [](const auto& info) {
                             return "T" + std::to_string(info.param.tokens) + "_hidden" +
                                    std::to_string(info.param.hidden);
                         });

TEST(Rmsnorm, WorksInPlace) {
    const int64_t tokens = 4, hidden = 2048;
    const auto x = engine::test::random_bf16(size_t(tokens * hidden), 5, -4.0f, 4.0f);
    const auto w = engine::test::random_bf16(size_t(hidden), 6, -2.0f, 2.0f);
    CudaStream stream;
    DeviceBuffer dx = upload(x), dw = upload(w);
    const Tensor t(dx.data(), DType::BF16, {tokens, hidden});
    rmsnorm(t, t, Tensor(dw.data(), DType::BF16, {hidden}), kEps, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto diff = compare_bf16(download<uint16_t>(dx), cpu_rmsnorm(x, w, hidden, kEps));
    EXPECT_EQ(diff.mismatches, 0u) << diff;
}

TEST(Rmsnorm, ZeroTokensIsANoOp) {
    DeviceBuffer w(8 * sizeof(uint16_t));
    CudaStream stream;
    rmsnorm(Tensor(nullptr, DType::BF16, {0, 8}), Tensor(nullptr, DType::BF16, {0, 8}),
            Tensor(w.data(), DType::BF16, {8}), kEps, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TEST(Rmsnorm, RejectsMismatchedShapesAndDtypes) {
    DeviceBuffer buf(4096);
    void* p = buf.data();
    CudaStream stream;
    EXPECT_THROW(rmsnorm(Tensor(p, DType::BF16, {3, 8}), Tensor(p, DType::BF16, {3, 8}),
                         Tensor(p, DType::BF16, {7}), kEps, stream),
                 std::invalid_argument);
    EXPECT_THROW(rmsnorm(Tensor(p, DType::BF16, {2, 8}), Tensor(p, DType::BF16, {3, 8}),
                         Tensor(p, DType::BF16, {8}), kEps, stream),
                 std::invalid_argument);
    EXPECT_THROW(rmsnorm(Tensor(p, DType::BF16, {3, 8}), Tensor(p, DType::F32, {3, 8}),
                         Tensor(p, DType::BF16, {8}), kEps, stream),
                 std::invalid_argument);
    EXPECT_THROW(rmsnorm(Tensor(p, DType::BF16, {24}), Tensor(p, DType::BF16, {3, 8}),
                         Tensor(p, DType::BF16, {8}), kEps, stream),
                 std::invalid_argument);
}

TEST(Rmsnorm, MatchesHuggingFaceLn1OnEveryTracedLayer) {
    if (const auto why = engine::test::reference_skip_reason(); !why.empty()) GTEST_SKIP() << why;
    const engine::test::SafetensorsFile weights(engine::test::model_dir() / "model.safetensors");
    const auto config = engine::load_config(engine::test::model_dir() / "config.json");
    const engine::test::SafetensorsFile ref(engine::test::traced_reference_prompt().file);
    for (int layer = 0; layer < config.layers; ++layer) {
        SCOPED_TRACE("layer " + std::to_string(layer));
        const std::string l = "layers." + std::to_string(layer) + ".";
        const auto x = ref.read<uint16_t>(
            layer == 0 ? "embed" : "layers." + std::to_string(layer - 1) + ".out");
        const auto w = weights.read<uint16_t>("model." + l + "input_layernorm.weight");
        const auto diff = compare_bf16(gpu_rmsnorm(x, w, config.hidden, float(config.rms_eps)),
                                       ref.read<uint16_t>(l + "ln1"));
        EXPECT_EQ(diff.mismatches, 0u) << diff;
    }
}

TEST(Rmsnorm, MatchesHuggingFaceFinalNormOnEveryPrompt) {
    if (const auto why = engine::test::reference_skip_reason(); !why.empty()) GTEST_SKIP() << why;
    const engine::test::SafetensorsFile weights(engine::test::model_dir() / "model.safetensors");
    const auto config = engine::load_config(engine::test::model_dir() / "config.json");
    const auto w = weights.read<uint16_t>("model.norm.weight");
    const std::string last = "layers." + std::to_string(config.layers - 1) + ".out";
    for (const auto& prompt : engine::test::load_reference_manifest()) {
        SCOPED_TRACE(prompt.name);
        const engine::test::SafetensorsFile ref(prompt.file);
        const auto diff = compare_bf16(
            gpu_rmsnorm(ref.read<uint16_t>(last), w, config.hidden, float(config.rms_eps)),
            ref.read<uint16_t>("final_norm"));
        EXPECT_EQ(diff.mismatches, 0u) << diff;
    }
}
