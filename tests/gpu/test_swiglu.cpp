#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine/kernels/swiglu.h"
#include "engine/model/config.h"
#include "support/bf16.h"
#include "support/device.h"
#include "support/paths.h"
#include "support/reference.h"
#include "support/safetensors_file.h"

using engine::DeviceBuffer;
using engine::DType;
using engine::Tensor;
using engine::naive::swiglu;
using engine::test::compare_bf16;
using engine::test::CudaStream;
using engine::test::download;
using engine::test::from_bf16;
using engine::test::round_bf16;
using engine::test::to_bf16;
using engine::test::upload;

namespace {

std::vector<uint16_t> cpu_swiglu(const std::vector<uint16_t>& gate_up, int64_t ff) {
    std::vector<uint16_t> out(gate_up.size() / 2);
    for (size_t idx = 0; idx < out.size(); ++idx) {
        const size_t t = idx / size_t(ff), j = idx % size_t(ff);
        const float g = from_bf16(gate_up[t * 2 * ff + j]);
        const float u = from_bf16(gate_up[t * 2 * ff + ff + j]);
        out[idx] = to_bf16(round_bf16(g / (1.0f + std::exp(-g))) * u);
    }
    return out;
}

std::vector<uint16_t> gpu_swiglu(const std::vector<uint16_t>& gate_up, int64_t ff) {
    const int64_t tokens = int64_t(gate_up.size()) / (2 * ff);
    CudaStream stream;
    DeviceBuffer dgu = upload(gate_up), out(gate_up.size() / 2 * sizeof(uint16_t));
    swiglu(Tensor(out.data(), DType::BF16, {tokens, ff}),
           Tensor(dgu.data(), DType::BF16, {tokens, 2 * ff}), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return download<uint16_t>(out);
}

}  // namespace

TEST(Swiglu, MatchesCpuReference) {
    for (const auto& [tokens, ff] :
         {std::pair<int64_t, int64_t>{1, 1}, {3, 100}, {7, 300}, {1, 5632}, {16, 5632}}) {
        SCOPED_TRACE("T = " + std::to_string(tokens) + ", ff = " + std::to_string(ff));
        const auto gate_up = engine::test::random_bf16(size_t(tokens * 2 * ff), 13, -12.0f, 12.0f);
        const auto diff = compare_bf16(gpu_swiglu(gate_up, ff), cpu_swiglu(gate_up, ff));
        EXPECT_EQ(diff.mismatches, 0u) << diff;
    }
}

TEST(Swiglu, ReadsGateFromTheFirstHalfAndUpFromTheSecond) {
    const std::vector<float> gate_up = {0.0f, 1.0f, 2.0f, 3.0f};
    std::vector<uint16_t> h;
    for (const float v : gate_up) h.push_back(to_bf16(v));
    const auto out = gpu_swiglu(h, 2);
    EXPECT_EQ(from_bf16(out[0]), 0.0f);
    EXPECT_EQ(out[1], to_bf16(round_bf16(1.0f / (1.0f + std::exp(-1.0f))) * 3.0f));
}

TEST(Swiglu, ZeroTokensIsANoOp) {
    CudaStream stream;
    swiglu(Tensor(nullptr, DType::BF16, {0, 8}), Tensor(nullptr, DType::BF16, {0, 16}), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TEST(Swiglu, RejectsMismatchedShapesAndDtypes) {
    DeviceBuffer buf(4096);
    void* p = buf.data();
    CudaStream stream;
    EXPECT_THROW(swiglu(Tensor(p, DType::BF16, {3, 8}), Tensor(p, DType::BF16, {3, 8}), stream),
                 std::invalid_argument);
    EXPECT_THROW(swiglu(Tensor(p, DType::BF16, {2, 8}), Tensor(p, DType::BF16, {3, 16}), stream),
                 std::invalid_argument);
    EXPECT_THROW(swiglu(Tensor(p, DType::BF16, {3, 8}), Tensor(p, DType::F32, {3, 16}), stream),
                 std::invalid_argument);
    EXPECT_THROW(swiglu(Tensor(p, DType::BF16, {3, 8}), Tensor(p, DType::BF16, {3, 17}), stream),
                 std::invalid_argument);
}

TEST(Swiglu, MatchesHuggingFaceActOnEveryTracedLayer) {
    if (const auto why = engine::test::reference_skip_reason(); !why.empty()) GTEST_SKIP() << why;
    const auto config = engine::load_config(engine::test::model_dir() / "config.json");
    const engine::test::SafetensorsFile ref(engine::test::traced_reference_prompt().file);
    const int64_t ff = config.intermediate;
    for (int layer = 0; layer < config.layers; ++layer) {
        SCOPED_TRACE("layer " + std::to_string(layer));
        const std::string l = "layers." + std::to_string(layer) + ".";
        const auto gate = ref.read<uint16_t>(l + "gate");
        const auto up = ref.read<uint16_t>(l + "up");
        std::vector<uint16_t> gate_up;
        gate_up.reserve(gate.size() * 2);
        for (size_t row = 0; row < gate.size(); row += size_t(ff)) {
            gate_up.insert(gate_up.end(), gate.begin() + row, gate.begin() + row + ff);
            gate_up.insert(gate_up.end(), up.begin() + row, up.begin() + row + ff);
        }
        const auto diff = compare_bf16(gpu_swiglu(gate_up, ff), ref.read<uint16_t>(l + "act"));
        EXPECT_EQ(diff.mismatches, 0u) << diff;
    }
}
