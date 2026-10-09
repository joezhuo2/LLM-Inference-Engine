#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine/kernels/rope.h"
#include "support/bf16.h"
#include "support/device.h"

using engine::DeviceBuffer;
using engine::DType;
using engine::Tensor;
using engine::naive::rope;
using engine::test::compare_bf16;
using engine::test::CudaStream;
using engine::test::download;
using engine::test::from_bf16;
using engine::test::round_bf16;
using engine::test::to_bf16;
using engine::test::upload;

namespace {

constexpr float kTheta = 10000.0f;

struct Layout {
    int heads, kv_heads, head_dim;
    int width() const { return (heads + 2 * kv_heads) * head_dim; }
};

std::vector<uint16_t> cpu_rope(std::vector<uint16_t> qkv, const std::vector<int32_t>& positions,
                               const Layout& l, float theta) {
    const int half = l.head_dim / 2;
    for (size_t t = 0; t < positions.size(); ++t) {
        for (int head = 0; head < l.heads + l.kv_heads; ++head) {
            uint16_t* x = qkv.data() + t * l.width() + head * l.head_dim;
            for (int i = 0; i < half; ++i) {
                const float inv_freq = 1.0f / std::pow(theta, float(2 * i) / float(l.head_dim));
                const float angle = float(positions[t]) * inv_freq;
                const float c = round_bf16(std::cos(angle)), s = round_bf16(std::sin(angle));
                const float lo = from_bf16(x[i]), hi = from_bf16(x[i + half]);
                x[i] = to_bf16(round_bf16(lo * c) + round_bf16(-hi * s));
                x[i + half] = to_bf16(round_bf16(hi * c) + round_bf16(lo * s));
            }
        }
    }
    return qkv;
}

std::vector<uint16_t> gpu_rope(const std::vector<uint16_t>& qkv,
                               const std::vector<int32_t>& positions, const Layout& l,
                               float theta) {
    const int64_t tokens = int64_t(positions.size());
    CudaStream stream;
    DeviceBuffer dqkv = upload(qkv), dpos = upload(positions);
    rope(Tensor(dqkv.data(), DType::BF16, {tokens, l.width()}),
         Tensor(dpos.data(), DType::I32, {tokens}), l.heads, l.kv_heads, theta, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return download<uint16_t>(dqkv);
}

std::vector<int32_t> spread_positions() {
    std::vector<int32_t> positions;
    for (int p = 0; p < 2048; p += 3) positions.push_back(p);
    positions.push_back(2047);
    return positions;
}

class RopeLayoutTest : public ::testing::TestWithParam<Layout> {};

}  // namespace

TEST_P(RopeLayoutTest, MatchesCpuReference) {
    const Layout l = GetParam();
    const auto positions = spread_positions();
    const auto qkv = engine::test::random_bf16(positions.size() * l.width(), 12, -3.0f, 3.0f);
    const auto diff =
        compare_bf16(gpu_rope(qkv, positions, l, kTheta), cpu_rope(qkv, positions, l, kTheta));
    EXPECT_EQ(diff.mismatches, 0u) << diff;
}

TEST_P(RopeLayoutTest, LeavesVUntouched) {
    const Layout l = GetParam();
    const auto positions = spread_positions();
    const auto qkv = engine::test::random_bf16(positions.size() * l.width(), 14);
    const auto out = gpu_rope(qkv, positions, l, kTheta);
    const size_t v_begin = size_t(l.heads + l.kv_heads) * l.head_dim;
    for (size_t t = 0; t < positions.size(); ++t)
        for (size_t i = v_begin; i < size_t(l.width()); ++i)
            ASSERT_EQ(out[t * l.width() + i], qkv[t * l.width() + i]) << "token " << t;
}

TEST_P(RopeLayoutTest, PositionZeroIsTheIdentity) {
    const Layout l = GetParam();
    const std::vector<int32_t> positions(5, 0);
    const auto qkv = engine::test::random_bf16(positions.size() * l.width(), 15);
    EXPECT_EQ(gpu_rope(qkv, positions, l, kTheta), qkv);
}

INSTANTIATE_TEST_SUITE_P(Layouts, RopeLayoutTest,
                         ::testing::Values(Layout{4, 2, 8}, Layout{3, 3, 2}, Layout{32, 4, 64}),
                         [](const auto& info) {
                             const Layout& l = info.param;
                             return "heads" + std::to_string(l.heads) + "_kv" +
                                    std::to_string(l.kv_heads) + "_dim" +
                                    std::to_string(l.head_dim);
                         });

TEST(Rope, ZeroTokensIsANoOp) {
    CudaStream stream;
    rope(Tensor(nullptr, DType::BF16, {0, 64}), Tensor(nullptr, DType::I32, {0}), 4, 2, kTheta,
         stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TEST(Rope, RejectsBadLayouts) {
    DeviceBuffer buf(4096);
    void* p = buf.data();
    CudaStream stream;
    const Tensor pos(p, DType::I32, {3});
    EXPECT_THROW(rope(Tensor(p, DType::BF16, {3, 63}), pos, 4, 2, kTheta, stream),
                 std::invalid_argument);
    EXPECT_THROW(rope(Tensor(p, DType::BF16, {3, 24}), pos, 4, 2, kTheta, stream),
                 std::invalid_argument);
    EXPECT_THROW(
        rope(Tensor(p, DType::BF16, {3, 64}), Tensor(p, DType::I32, {2}), 4, 2, kTheta, stream),
        std::invalid_argument);
    EXPECT_THROW(
        rope(Tensor(p, DType::BF16, {3, 64}), Tensor(p, DType::I64, {3}), 4, 2, kTheta, stream),
        std::invalid_argument);
    EXPECT_THROW(rope(Tensor(p, DType::F32, {3, 64}), pos, 4, 2, kTheta, stream),
                 std::invalid_argument);
    EXPECT_THROW(rope(Tensor(p, DType::BF16, {3, 1028}), pos, 1, 1, kTheta, stream),
                 std::invalid_argument);
    EXPECT_THROW(rope(Tensor(p, DType::BF16, {3, 64}), pos, 0, 2, kTheta, stream),
                 std::invalid_argument);
}
