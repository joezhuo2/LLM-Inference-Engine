#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine/kernels/all_reduce.h"
#include "support/bf16.h"
#include "support/device.h"

using engine::all_reduce_add;
using engine::DeviceBuffer;
using engine::DType;
using engine::kMaxRanks;
using engine::Tensor;
using engine::test::compare_bf16;
using engine::test::CudaStream;
using engine::test::download;
using engine::test::from_bf16;
using engine::test::random_bf16;
using engine::test::to_bf16;
using engine::test::upload;

namespace {

std::vector<uint16_t> host_all_reduce(const std::vector<uint16_t>& x,
                                      const std::vector<std::vector<uint16_t>>& partials) {
    std::vector<uint16_t> out(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        float sum = from_bf16(partials[0][i]);
        for (size_t r = 1; r < partials.size(); ++r) sum += from_bf16(partials[r][i]);
        out[i] = to_bf16(from_bf16(x[i]) + sum);
    }
    return out;
}

}  // namespace

TEST(AllReduce, MatchesAHostSum) {
    for (const int ranks : {1, 2, 4, kMaxRanks}) {
        for (const int64_t rows : {1, 3, 512}) {
            SCOPED_TRACE("ranks " + std::to_string(ranks) + ", rows " + std::to_string(rows));
            const int64_t cols = 2048;
            const size_t n = size_t(rows * cols);
            const auto x = random_bf16(n, 11, -8.0f, 8.0f);
            std::vector<std::vector<uint16_t>> partials;
            std::vector<DeviceBuffer> device;
            std::vector<Tensor> tensors;
            for (int r = 0; r < ranks; ++r) {
                partials.push_back(random_bf16(n, 20 + uint32_t(r), -0.5f, 0.5f));
                device.push_back(upload(partials.back()));
                tensors.emplace_back(device.back().data(), DType::BF16,
                                     std::initializer_list<int64_t>{rows, cols});
            }
            CudaStream stream;
            DeviceBuffer dx = upload(x);
            all_reduce_add(Tensor(dx.data(), DType::BF16, {rows, cols}), tensors, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            const auto diff = compare_bf16(download<uint16_t>(dx), host_all_reduce(x, partials));
            EXPECT_EQ(diff.mismatches, 0u) << diff;
        }
    }
}

TEST(AllReduce, RejectsBadPartials) {
    CudaStream stream;
    DeviceBuffer a(4 * 8 * 2), b(4 * 8 * 2);
    const Tensor x(a.data(), DType::BF16, {4, 8});
    const Tensor p(b.data(), DType::BF16, {4, 8});
    EXPECT_THROW(all_reduce_add(x, {}, stream), std::invalid_argument);
    const std::vector<Tensor> too_many(size_t(kMaxRanks) + 1, p);
    EXPECT_THROW(all_reduce_add(x, too_many, stream), std::invalid_argument);
    const std::vector<Tensor> other_shape = {p, Tensor(b.data(), DType::BF16, {8, 4})};
    EXPECT_THROW(all_reduce_add(x, other_shape, stream), std::invalid_argument);
    const std::vector<Tensor> other_type = {Tensor(b.data(), DType::F32, {4, 4})};
    EXPECT_THROW(all_reduce_add(Tensor(a.data(), DType::F32, {4, 4}), other_type, stream),
                 std::invalid_argument);
    EXPECT_THROW(all_reduce_add(x, other_type, stream), std::invalid_argument);
    const std::vector<Tensor> one = {p};
    EXPECT_NO_THROW(all_reduce_add(x, one, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}
