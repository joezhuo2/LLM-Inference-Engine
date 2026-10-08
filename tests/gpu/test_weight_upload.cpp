#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "engine/loader/weight_layout.h"
#include "engine/runtime/device_buffer.h"
#include "engine/runtime/weights.h"
#include "kernels/cuda_check.h"
#include "support/fake_checkpoint.h"

using engine::CopyOp;
using engine::DeviceBuffer;
using engine::WeightLayout;

namespace {

constexpr uint64_t kDataOffset = 24;

class WeightUploadTest : public ::testing::TestWithParam<size_t> {
protected:
    void SetUp() override {
        const auto config = engine::test::tiny_config();
        const auto header = engine::test::fake_header(config, kDataOffset);
        layout_ = engine::plan_weight_layout(config, header);
        uint64_t end = kDataOffset;
        for (const auto& [name, t] : header.tensors) end = std::max(end, t.offset + t.bytes);
        file_.resize(end);
        for (size_t i = 0; i < file_.size(); ++i) file_[i] = std::byte((i * 131 + 7) & 0xFF);
        CUDA_CHECK(cudaStreamCreate(&stream_));
    }

    void TearDown() override { CUDA_CHECK(cudaStreamDestroy(stream_)); }

    std::vector<std::byte> file_;
    WeightLayout layout_;
    cudaStream_t stream_ = nullptr;
};

}  // namespace

TEST_P(WeightUploadTest, EveryTensorLandsAtItsDestination) {
    DeviceBuffer device(layout_.total_bytes);
    const auto stats = engine::upload_weights(file_, layout_, device.data(), stream_, GetParam());

    std::vector<std::byte> back(layout_.total_bytes);
    CUDA_CHECK(cudaMemcpy(back.data(), device.data(), back.size(), cudaMemcpyDeviceToHost));
    uint64_t total = 0;
    for (const CopyOp& c : layout_.copies) {
        EXPECT_EQ(std::memcmp(back.data() + c.dst, file_.data() + c.src, c.bytes), 0) << c.name;
        total += c.bytes;
    }
    EXPECT_EQ(stats.bytes, total);
    EXPECT_GE(stats.seconds, 0.0);
}

INSTANTIATE_TEST_SUITE_P(ChunkSizes, WeightUploadTest,
                         ::testing::Values(size_t(1), size_t(7), size_t(64), size_t(1) << 20),
                         [](const auto& info) { return "Chunk" + std::to_string(info.param); });

TEST(WeightUpload, RejectsZeroChunk) {
    const auto config = engine::test::tiny_config();
    const auto layout = engine::plan_weight_layout(config, engine::test::fake_header(config));
    std::vector<std::byte> file(16);
    EXPECT_THROW(engine::upload_weights(file, layout, nullptr, nullptr, 0), std::invalid_argument);
}
