#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "engine/kernels/attention.h"
#include "support/bf16.h"
#include "support/device.h"

using engine::DeviceBuffer;
using engine::DType;
using engine::Tensor;
using engine::naive::store_kv;
using engine::test::CudaStream;
using engine::test::download;
using engine::test::upload;

TEST(StoreKv, CopiesKAndVRowsToTheirCachePositions) {
    constexpr int64_t tokens = 3, q_width = 8, kv_width = 4, width = q_width + 2 * kv_width;
    constexpr int64_t rows = 6, start = 2;
    const auto qkv = engine::test::random_bf16(size_t(tokens * width), 21);
    const auto k_init = engine::test::random_bf16(size_t(rows * kv_width), 22);
    const auto v_init = engine::test::random_bf16(size_t(rows * kv_width), 23);

    CudaStream stream;
    DeviceBuffer dqkv = upload(qkv), dk = upload(k_init), dv = upload(v_init);
    store_kv(Tensor(dk.data(), DType::BF16, {rows, kv_width}),
             Tensor(dv.data(), DType::BF16, {rows, kv_width}),
             Tensor(dqkv.data(), DType::BF16, {tokens, width}), start, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto k = download<uint16_t>(dk), v = download<uint16_t>(dv);

    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t i = 0; i < kv_width; ++i) {
            const size_t c = size_t(r * kv_width + i);
            if (r < start || r >= start + tokens) {
                EXPECT_EQ(k[c], k_init[c]) << "row " << r;
                EXPECT_EQ(v[c], v_init[c]) << "row " << r;
            } else {
                const size_t row = size_t((r - start) * width);
                EXPECT_EQ(k[c], qkv[row + q_width + i]) << "row " << r;
                EXPECT_EQ(v[c], qkv[row + q_width + kv_width + i]) << "row " << r;
            }
        }
    }
}

TEST(StoreKv, ZeroTokensIsANoOp) {
    DeviceBuffer buf(64);
    CudaStream stream;
    const Tensor cache(buf.data(), DType::BF16, {4, 4});
    store_kv(cache, cache, Tensor(nullptr, DType::BF16, {0, 16}), 4, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TEST(StoreKv, RejectsBadShapesAndRanges) {
    DeviceBuffer buf(4096);
    void* p = buf.data();
    CudaStream stream;
    const Tensor cache(p, DType::BF16, {4, 4});
    const Tensor qkv(p, DType::BF16, {2, 16});
    EXPECT_THROW(store_kv(cache, cache, qkv, 3, stream), std::invalid_argument);
    EXPECT_THROW(store_kv(cache, cache, qkv, -1, stream), std::invalid_argument);
    EXPECT_THROW(store_kv(cache, cache, Tensor(p, DType::BF16, {2, 8}), 0, stream),
                 std::invalid_argument);
    EXPECT_THROW(store_kv(cache, Tensor(p, DType::BF16, {4, 2}), qkv, 0, stream),
                 std::invalid_argument);
    EXPECT_THROW(store_kv(Tensor(p, DType::F32, {4, 4}), cache, qkv, 0, stream),
                 std::invalid_argument);
    EXPECT_THROW(store_kv(cache, cache, Tensor(p, DType::F32, {2, 16}), 0, stream),
                 std::invalid_argument);
}
