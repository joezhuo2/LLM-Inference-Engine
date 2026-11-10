#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine/kernels/embedding.h"
#include "support/bf16.h"
#include "support/device.h"
#include "support/guarded.h"

using engine::DeviceBuffer;
using engine::DType;
using engine::Tensor;
using engine::test::CudaStream;
using engine::test::download;
using engine::test::GuardedBf16;
using engine::test::random_bf16;
using engine::test::upload;

namespace {

std::vector<int32_t> random_ids(int64_t tokens, int64_t vocab, uint32_t seed) {
    std::vector<int32_t> ids(static_cast<size_t>(tokens));
    uint32_t x = seed;
    for (auto& id : ids) {
        x = x * 1664525u + 1013904223u;
        id = int32_t((x >> 8) % uint32_t(vocab));
    }
    return ids;
}

void check(int64_t tokens, int64_t vocab, int64_t hidden, size_t table_shift, size_t out_shift) {
    const auto table = random_bf16(size_t(vocab * hidden), 23);
    auto ids = random_ids(tokens, vocab, 7);
    if (tokens > 1) {
        ids.front() = 0;
        ids.back() = int32_t(vocab - 1);
    }
    CudaStream stream;
    DeviceBuffer dids = upload(ids);
    GuardedBf16 dtable(table, table_shift, false);
    const std::vector<uint16_t> out_shape(size_t(tokens * hidden), 0);
    GuardedBf16 want(out_shape, out_shift, true), got(out_shape, out_shift, true);
    const Tensor tids(dids.data(), DType::I32, {tokens});
    engine::naive::embedding(want.tensor({tokens, hidden}), tids, dtable.tensor({vocab, hidden}),
                             stream);
    engine::fast::embedding(got.tensor({tokens, hidden}), tids, dtable.tensor({vocab, hidden}),
                            stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    EXPECT_EQ(got.values(), want.values());
    EXPECT_TRUE(got.guards_intact()) << "a write outside out";
    EXPECT_TRUE(dtable.guards_intact());
    EXPECT_EQ(dtable.values(), table) << "the table was modified";
    const auto out = got.values();
    for (int64_t t = 0; t < tokens; ++t)
        for (int64_t i = 0; i < hidden; ++i)
            ASSERT_EQ(out[size_t(t * hidden + i)], table[size_t(ids[size_t(t)] * hidden + i)])
                << "token " << t << ", column " << i;
}

struct Shape {
    int64_t tokens, vocab, hidden;
    size_t table_shift, out_shift;
};

class FastEmbeddingShapeTest : public ::testing::TestWithParam<Shape> {};

}  // namespace

TEST_P(FastEmbeddingShapeTest, MatchesNaive) {
    const auto [tokens, vocab, hidden, table_shift, out_shift] = GetParam();
    check(tokens, vocab, hidden, table_shift, out_shift);
}

INSTANTIATE_TEST_SUITE_P(Shapes, FastEmbeddingShapeTest,
                         ::testing::Values(Shape{1, 1, 8, 0, 0}, Shape{1, 37, 2048, 0, 0},
                                           Shape{16, 37, 2048, 0, 0}, Shape{512, 300, 2048, 0, 0},
                                           Shape{2048, 64, 2048, 0, 0}, Shape{7, 37, 300, 0, 0},
                                           Shape{5, 37, 1, 0, 0}, Shape{9, 37, 7, 0, 0},
                                           Shape{9, 37, 9, 0, 0}, Shape{16, 37, 2048, 1, 0},
                                           Shape{16, 37, 2048, 0, 1}, Shape{16, 37, 2048, 4, 4},
                                           Shape{16, 37, 2048, 8, 8}, Shape{3, 37, 2047, 0, 0},
                                           Shape{70000, 5, 8, 0, 0}),
                         [](const auto& info) {
                             const Shape& s = info.param;
                             return "T" + std::to_string(s.tokens) + "_V" +
                                    std::to_string(s.vocab) + "_H" + std::to_string(s.hidden) +
                                    "_shift" + std::to_string(s.table_shift) + "_" +
                                    std::to_string(s.out_shift);
                         });

TEST(FastEmbedding, RepeatedIdsCopyTheSameRow) {
    const int64_t vocab = 4, hidden = 2048;
    const std::vector<int32_t> ids = {2, 2, 2, 0, 2, 3, 3, 2};
    const int64_t tokens = int64_t(ids.size());
    const auto table = random_bf16(size_t(vocab * hidden), 31);
    CudaStream stream;
    DeviceBuffer dtable = upload(table), dids = upload(ids);
    DeviceBuffer dout(size_t(tokens * hidden) * sizeof(uint16_t));
    engine::fast::embedding(Tensor(dout.data(), DType::BF16, {tokens, hidden}),
                            Tensor(dids.data(), DType::I32, {tokens}),
                            Tensor(dtable.data(), DType::BF16, {vocab, hidden}), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto out = download<uint16_t>(dout);
    for (int64_t t = 0; t < tokens; ++t)
        EXPECT_EQ(std::vector<uint16_t>(out.begin() + t * hidden, out.begin() + (t + 1) * hidden),
                  std::vector<uint16_t>(table.begin() + ids[size_t(t)] * hidden,
                                        table.begin() + (ids[size_t(t)] + 1) * hidden))
            << "token " << t;
}

TEST(FastEmbedding, ZeroTokensIsANoOp) {
    DeviceBuffer table(4 * 8 * sizeof(uint16_t));
    CudaStream stream;
    engine::fast::embedding(Tensor(nullptr, DType::BF16, {0, 8}), Tensor(nullptr, DType::I32, {0}),
                            Tensor(table.data(), DType::BF16, {4, 8}), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TEST(FastEmbedding, RejectsMismatchedShapesAndDtypes) {
    DeviceBuffer buf(4096);
    void* p = buf.data();
    CudaStream stream;
    const auto call = [&](const Tensor& out, const Tensor& ids, const Tensor& table) {
        engine::fast::embedding(out, ids, table, stream);
    };
    const Tensor out(p, DType::BF16, {3, 8}), ids(p, DType::I32, {3}),
        table(p, DType::BF16, {4, 8});
    EXPECT_THROW(call(out, Tensor(p, DType::I64, {3}), table), std::invalid_argument);
    EXPECT_THROW(call(out, Tensor(p, DType::I32, {2}), table), std::invalid_argument);
    EXPECT_THROW(call(out, Tensor(p, DType::I32, {3, 1}), table), std::invalid_argument);
    EXPECT_THROW(call(Tensor(p, DType::BF16, {3, 6}), ids, table), std::invalid_argument);
    EXPECT_THROW(call(Tensor(p, DType::F32, {3, 8}), ids, table), std::invalid_argument);
    EXPECT_THROW(call(Tensor(p, DType::BF16, {24}), ids, table), std::invalid_argument);
    EXPECT_THROW(call(out, ids, Tensor(p, DType::F32, {4, 8})), std::invalid_argument);
    EXPECT_THROW(call(out, ids, Tensor(p, DType::BF16, {32})), std::invalid_argument);
}
