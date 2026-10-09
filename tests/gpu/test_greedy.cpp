#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine/sampling/greedy.h"
#include "support/bf16.h"
#include "support/device.h"
#include "support/reference.h"
#include "support/safetensors_file.h"

using engine::DeviceBuffer;
using engine::DType;
using engine::greedy;
using engine::Tensor;
using engine::test::CudaStream;
using engine::test::download;
using engine::test::from_bf16;
using engine::test::to_bf16;
using engine::test::upload;

namespace {

std::vector<int32_t> cpu_greedy(const std::vector<uint16_t>& logits, int64_t vocab) {
    const int64_t tokens = int64_t(logits.size()) / vocab;
    std::vector<int32_t> ids(size_t(tokens), 0);
    for (int64_t t = 0; t < tokens; ++t) {
        float best = from_bf16(logits[size_t(t * vocab)]);
        for (int64_t i = 1; i < vocab; ++i) {
            const float v = from_bf16(logits[size_t(t * vocab + i)]);
            if (v > best) {
                best = v;
                ids[size_t(t)] = int32_t(i);
            }
        }
    }
    return ids;
}

std::vector<int32_t> run_greedy(const std::vector<uint16_t>& logits, int64_t vocab) {
    const int64_t tokens = int64_t(logits.size()) / vocab;
    CudaStream stream;
    DeviceBuffer dlogits = upload(logits);
    DeviceBuffer dids(size_t(tokens) * sizeof(int32_t));
    greedy(Tensor(dids.data(), DType::I32, {tokens}),
           Tensor(dlogits.data(), DType::BF16, {tokens, vocab}), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return download<int32_t>(dids);
}

}  // namespace

TEST(Greedy, MatchesCpuArgmax) {
    for (const int64_t vocab : {1, 7, 1000, 1024, 1025, 32000}) {
        SCOPED_TRACE("vocab " + std::to_string(vocab));
        const auto logits = engine::test::random_bf16(size_t(5 * vocab), uint32_t(vocab), -8, 8);
        EXPECT_EQ(run_greedy(logits, vocab), cpu_greedy(logits, vocab));
    }
}

TEST(Greedy, TiesPickTheLowestIndex) {
    const int64_t vocab = 4000;
    std::vector<uint16_t> logits(size_t(4 * vocab), to_bf16(-1.0f));
    for (int64_t i = 0; i < vocab; ++i) logits[size_t(i)] = to_bf16(2.0f);
    for (const int64_t i : {3000, 40}) logits[size_t(vocab + i)] = to_bf16(5.0f);
    for (const int64_t i : {40 + 1024, 40}) logits[size_t(2 * vocab + i)] = to_bf16(5.0f);
    for (const int64_t i : {3999, 1023}) logits[size_t(3 * vocab + i)] = to_bf16(5.0f);
    EXPECT_EQ(run_greedy(logits, vocab), (std::vector<int32_t>{0, 40, 40, 1023}));
}

TEST(Greedy, ARowOfNegativeInfinityPicksTheFirstIndex) {
    const std::vector<uint16_t> logits(3000, to_bf16(-INFINITY));
    EXPECT_EQ(run_greedy(logits, 3000), (std::vector<int32_t>{0}));
}

TEST(Greedy, ZeroTokensIsANoOp) {
    CudaStream stream;
    EXPECT_NO_THROW(
        greedy(Tensor(nullptr, DType::I32, {0}), Tensor(nullptr, DType::BF16, {0, 32000}), stream));
}

TEST(Greedy, RejectsBadShapesAndDtypes) {
    CudaStream stream;
    int dummy = 0;
    void* p = &dummy;
    EXPECT_THROW(greedy(Tensor(p, DType::I32, {2}), Tensor(p, DType::F32, {2, 10}), stream),
                 std::invalid_argument);
    EXPECT_THROW(greedy(Tensor(p, DType::I32, {2}), Tensor(p, DType::BF16, {2, 0}), stream),
                 std::invalid_argument);
    EXPECT_THROW(greedy(Tensor(p, DType::I32, {3}), Tensor(p, DType::BF16, {2, 10}), stream),
                 std::invalid_argument);
    EXPECT_THROW(greedy(Tensor(p, DType::I64, {2}), Tensor(p, DType::BF16, {2, 10}), stream),
                 std::invalid_argument);
}

TEST(Greedy, MatchesCpuArgmaxOnHuggingFaceLogitsOfEveryPrompt) {
    if (const auto why = engine::test::reference_skip_reason(); !why.empty()) GTEST_SKIP() << why;
    for (const auto& prompt : engine::test::load_reference_manifest()) {
        SCOPED_TRACE(prompt.name);
        const engine::test::SafetensorsFile ref(prompt.file);
        const int64_t vocab = ref.entry("logits").shape[1];
        const auto logits = ref.read<uint16_t>("logits");
        EXPECT_EQ(run_greedy(logits, vocab), cpu_greedy(logits, vocab));
    }
}
