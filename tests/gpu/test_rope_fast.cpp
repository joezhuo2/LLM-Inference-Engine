#include <gtest/gtest.h>

#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine/kernels/rope.h"
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

struct Layout {
    int heads, kv_heads, head_dim;
    int64_t width() const { return int64_t(heads + 2 * kv_heads) * head_dim; }
};

std::vector<int32_t> consecutive(int64_t tokens, int32_t first) {
    std::vector<int32_t> p(static_cast<size_t>(tokens));
    for (int64_t i = 0; i < tokens; ++i) p[size_t(i)] = first + int32_t(i);
    return p;
}

std::vector<int32_t> random_positions(int64_t tokens, int32_t max, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int32_t> dist(0, max);
    std::vector<int32_t> p(static_cast<size_t>(tokens));
    for (auto& v : p) v = dist(rng);
    return p;
}

void check(const Layout& l, const std::vector<int32_t>& positions, float theta, size_t shift,
           uint32_t seed) {
    const int64_t tokens = int64_t(positions.size());
    const int64_t width = l.width();
    const auto qkv = random_bf16(size_t(tokens * width), seed, -6.0f, 6.0f);

    CudaStream stream;
    GuardedBf16 expected(qkv, shift, false), actual(qkv, shift, false);
    DeviceBuffer dpos = upload(positions);
    const Tensor pos(dpos.data(), DType::I32, {tokens});
    engine::naive::rope(expected.tensor({tokens, width}), pos, l.heads, l.kv_heads, theta, stream);
    engine::fast::rope(actual.tensor({tokens, width}), pos, l.heads, l.kv_heads, theta, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const auto want = expected.values();
    const auto got = actual.values();
    size_t mismatches = 0, first = 0;
    for (size_t i = 0; i < got.size(); ++i)
        if (got[i] != want[i] && mismatches++ == 0) first = i;
    EXPECT_EQ(mismatches, 0u) << "first mismatch at row " << first / size_t(width) << ", column "
                              << first % size_t(width);
    EXPECT_TRUE(actual.guards_intact()) << "a write outside qkv";
    EXPECT_EQ(download<int32_t>(dpos), positions) << "positions were modified";

    const int64_t v_begin = int64_t(l.heads + l.kv_heads) * l.head_dim;
    for (int64_t t = 0; t < tokens; ++t)
        for (int64_t c = v_begin; c < width; ++c)
            ASSERT_EQ(got[size_t(t * width + c)], qkv[size_t(t * width + c)])
                << "V changed at row " << t << ", column " << c;
}

}  // namespace

TEST(FastRope, MatchesNaiveOnTinyLlamaLayout) {
    const Layout l{32, 4, 64};
    check(l, consecutive(1, 0), 10000.0f, 0, 1);
    check(l, consecutive(1, 1000), 10000.0f, 0, 2);
    check(l, consecutive(16, 0), 10000.0f, 0, 3);
    check(l, consecutive(2048, 0), 10000.0f, 0, 4);
    check(l, random_positions(16, 2047, 5), 10000.0f, 0, 6);
    check(l, random_positions(257, 2047, 7), 10000.0f, 0, 8);
}

TEST(FastRope, MatchesNaiveOnOtherLayouts) {
    const Layout layouts[] = {{1, 1, 2},  {4, 2, 8},   {3, 3, 6},   {2, 1, 10}, {6, 2, 50},
                              {8, 2, 16}, {8, 8, 128}, {4, 1, 256}, {5, 1, 12}, {2, 2, 62}};
    uint32_t seed = 100;
    for (const Layout& l : layouts) {
        SCOPED_TRACE("heads " + std::to_string(l.heads) + ", kv_heads " +
                     std::to_string(l.kv_heads) + ", head_dim " + std::to_string(l.head_dim));
        const auto positions = random_positions(9, 2047, seed++);
        check(l, positions, 10000.0f, 0, seed++);
    }
}

TEST(FastRope, MatchesNaiveWithAnotherTheta) {
    check({32, 4, 64}, random_positions(8, 2047, 11), 500000.0f, 0, 12);
    check({4, 2, 8}, random_positions(8, 2047, 13), 1.0f, 0, 14);
}

TEST(FastRope, MatchesNaiveOnAnUnalignedBasePointer) {
    check({32, 4, 64}, random_positions(16, 2047, 21), 10000.0f, 1, 22);
    check({4, 2, 8}, random_positions(5, 2047, 23), 10000.0f, 1, 24);
    check({3, 1, 6}, random_positions(5, 2047, 25), 10000.0f, 3, 26);
}

TEST(FastRope, EqualPositionsGiveEqualRotations) {
    const Layout l{32, 4, 64};
    const int64_t tokens = 6, width = l.width();
    std::vector<uint16_t> row = random_bf16(size_t(width), 31, -6.0f, 6.0f);
    std::vector<uint16_t> qkv;
    for (int64_t t = 0; t < tokens; ++t) qkv.insert(qkv.end(), row.begin(), row.end());
    const std::vector<int32_t> positions(size_t(tokens), 777);
    CudaStream stream;
    DeviceBuffer dqkv = upload(qkv), dpos = upload(positions);
    engine::fast::rope(Tensor(dqkv.data(), DType::BF16, {tokens, width}),
                       Tensor(dpos.data(), DType::I32, {tokens}), l.heads, l.kv_heads, 10000.0f,
                       stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto out = download<uint16_t>(dqkv);
    for (int64_t t = 1; t < tokens; ++t)
        EXPECT_EQ(std::vector<uint16_t>(out.begin() + t * width, out.begin() + (t + 1) * width),
                  std::vector<uint16_t>(out.begin(), out.begin() + width));
}

TEST(FastRope, PositionZeroIsTheIdentity) {
    const Layout l{32, 4, 64};
    const int64_t width = l.width();
    const auto qkv = random_bf16(size_t(width), 41, -6.0f, 6.0f);
    CudaStream stream;
    DeviceBuffer dqkv = upload(qkv), dpos = upload(std::vector<int32_t>{0});
    engine::fast::rope(Tensor(dqkv.data(), DType::BF16, {1, width}),
                       Tensor(dpos.data(), DType::I32, {1}), l.heads, l.kv_heads, 10000.0f, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    EXPECT_EQ(download<uint16_t>(dqkv), qkv);
}

TEST(FastRope, IsDeterministic) {
    const Layout l{32, 4, 64};
    const int64_t tokens = 32, width = l.width();
    const auto qkv = random_bf16(size_t(tokens * width), 51, -6.0f, 6.0f);
    const auto positions = random_positions(tokens, 2047, 52);
    DeviceBuffer dpos = upload(positions);
    std::vector<uint16_t> first;
    for (int i = 0; i < 3; ++i) {
        CudaStream stream;
        DeviceBuffer dqkv = upload(qkv);
        engine::fast::rope(Tensor(dqkv.data(), DType::BF16, {tokens, width}),
                           Tensor(dpos.data(), DType::I32, {tokens}), l.heads, l.kv_heads, 10000.0f,
                           stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (i == 0) first = download<uint16_t>(dqkv);
        EXPECT_EQ(download<uint16_t>(dqkv), first);
    }
}

TEST(FastRope, ZeroTokensIsANoOp) {
    CudaStream stream;
    engine::fast::rope(Tensor(nullptr, DType::BF16, {0, 2560}), Tensor(nullptr, DType::I32, {0}),
                       32, 4, 10000.0f, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TEST(FastRope, RejectsBadArguments) {
    DeviceBuffer buf(1 << 16);
    void* p = buf.data();
    CudaStream stream;
    const auto call = [&](const Tensor& qkv, const Tensor& pos, int heads, int kv_heads) {
        engine::fast::rope(qkv, pos, heads, kv_heads, 10000.0f, stream);
    };
    const Tensor qkv(p, DType::BF16, {3, 48});
    const Tensor pos(p, DType::I32, {3});
    EXPECT_NO_THROW(call(qkv, pos, 4, 2));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    EXPECT_THROW(call(qkv, pos, 0, 2), std::invalid_argument);
    EXPECT_THROW(call(qkv, pos, 4, 0), std::invalid_argument);
    EXPECT_THROW(call(qkv, pos, -4, 2), std::invalid_argument);
    EXPECT_THROW(call(Tensor(p, DType::F32, {3, 48}), pos, 4, 2), std::invalid_argument);
    EXPECT_THROW(call(Tensor(p, DType::BF16, {144}), pos, 4, 2), std::invalid_argument);
    EXPECT_THROW(call(qkv, Tensor(p, DType::BF16, {3}), 4, 2), std::invalid_argument);
    EXPECT_THROW(call(qkv, Tensor(p, DType::I32, {2}), 4, 2), std::invalid_argument);
    EXPECT_THROW(call(qkv, Tensor(p, DType::I32, {1, 3}), 4, 2), std::invalid_argument);
    EXPECT_THROW(call(qkv, pos, 4, 3), std::invalid_argument);
    EXPECT_THROW(call(Tensor(p, DType::BF16, {3, 40}), pos, 4, 1), std::invalid_argument);
    EXPECT_THROW(call(Tensor(p, DType::BF16, {3, 40}), pos, 4, 2), std::invalid_argument);
    EXPECT_THROW(call(Tensor(p, DType::BF16, {1, 15}), Tensor(p, DType::I32, {1}), 1, 1),
                 std::invalid_argument);
    EXPECT_THROW(call(Tensor(p, DType::BF16, {1, 3 * 258}), Tensor(p, DType::I32, {1}), 1, 1),
                 std::invalid_argument);
    EXPECT_NO_THROW(call(Tensor(p, DType::BF16, {1, 3 * 256}), Tensor(p, DType::I32, {1}), 1, 1));
}
