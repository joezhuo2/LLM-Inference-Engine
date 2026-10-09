#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "engine/kernels/embedding.h"
#include "support/bf16.h"
#include "support/device.h"
#include "support/paths.h"
#include "support/reference.h"
#include "support/safetensors_file.h"

using engine::DeviceBuffer;
using engine::DType;
using engine::Tensor;
using engine::naive::embedding;
using engine::test::CudaStream;
using engine::test::download;
using engine::test::upload;

TEST(Embedding, CopiesTheRowOfEachId) {
    const int64_t vocab = 37, hidden = 300;
    const std::vector<int32_t> ids = {5, 0, 36, 5, 17, 1, 0};
    const int64_t tokens = int64_t(ids.size());
    const auto table = engine::test::random_bf16(size_t(vocab * hidden), 11);
    CudaStream stream;
    DeviceBuffer dtable = upload(table), dids = upload(ids);
    DeviceBuffer dout(size_t(tokens * hidden) * sizeof(uint16_t));

    embedding(Tensor(dout.data(), DType::BF16, {tokens, hidden}),
              Tensor(dids.data(), DType::I32, {tokens}),
              Tensor(dtable.data(), DType::BF16, {vocab, hidden}), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const auto out = download<uint16_t>(dout);
    for (int64_t t = 0; t < tokens; ++t)
        for (int64_t i = 0; i < hidden; ++i)
            ASSERT_EQ(out[t * hidden + i], table[ids[t] * hidden + i])
                << "token " << t << ", column " << i;
}

TEST(Embedding, ZeroTokensIsANoOp) {
    DeviceBuffer table(4 * 8 * sizeof(uint16_t));
    CudaStream stream;
    embedding(Tensor(nullptr, DType::BF16, {0, 8}), Tensor(nullptr, DType::I32, {0}),
              Tensor(table.data(), DType::BF16, {4, 8}), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TEST(Embedding, RejectsMismatchedShapesAndDtypes) {
    DeviceBuffer buf(4096);
    void* p = buf.data();
    CudaStream stream;
    EXPECT_THROW(embedding(Tensor(p, DType::BF16, {3, 8}), Tensor(p, DType::I64, {3}),
                           Tensor(p, DType::BF16, {4, 8}), stream),
                 std::invalid_argument);
    EXPECT_THROW(embedding(Tensor(p, DType::BF16, {3, 8}), Tensor(p, DType::I32, {2}),
                           Tensor(p, DType::BF16, {4, 8}), stream),
                 std::invalid_argument);
    EXPECT_THROW(embedding(Tensor(p, DType::BF16, {3, 6}), Tensor(p, DType::I32, {3}),
                           Tensor(p, DType::BF16, {4, 8}), stream),
                 std::invalid_argument);
    EXPECT_THROW(embedding(Tensor(p, DType::BF16, {3, 8}), Tensor(p, DType::I32, {3}),
                           Tensor(p, DType::F32, {4, 8}), stream),
                 std::invalid_argument);
}

TEST(Embedding, MatchesHuggingFaceOnEveryPrompt) {
    if (const auto why = engine::test::reference_skip_reason(); !why.empty()) GTEST_SKIP() << why;

    const engine::test::SafetensorsFile weights(engine::test::model_dir() / "model.safetensors");
    const auto& entry = weights.entry("model.embed_tokens.weight");
    const int64_t vocab = entry.shape[0], hidden = entry.shape[1];
    DeviceBuffer table = upload(weights.read<uint16_t>("model.embed_tokens.weight"));
    CudaStream stream;
    for (const auto& prompt : engine::test::load_reference_manifest()) {
        SCOPED_TRACE(prompt.name);
        const engine::test::SafetensorsFile ref(prompt.file);
        const auto ids = ref.read<int32_t>("input_ids");
        const int64_t tokens = int64_t(ids.size());
        DeviceBuffer dids = upload(ids), out(size_t(tokens * hidden) * sizeof(uint16_t));

        embedding(Tensor(out.data(), DType::BF16, {tokens, hidden}),
                  Tensor(dids.data(), DType::I32, {tokens}),
                  Tensor(table.data(), DType::BF16, {vocab, hidden}), stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const auto expected = ref.read<uint16_t>("embed");
        const auto actual = download<uint16_t>(out);
        ASSERT_EQ(actual.size(), expected.size());
        const auto diff = engine::test::compare_bf16(actual, expected);
        EXPECT_EQ(diff.mismatches, 0u) << diff;
    }
}
