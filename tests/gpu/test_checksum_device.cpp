#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "engine/loader/checksum.h"
#include "engine/loader/weight_layout.h"
#include "engine/runtime/checksum.h"
#include "engine/runtime/device_buffer.h"
#include "engine/runtime/weights.h"
#include "kernels/cuda_check.h"
#include "support/bf16.h"
#include "support/fake_checkpoint.h"

TEST(ChecksumDevice, MatchesTheFileAfterUpload) {
    const auto config = engine::test::tiny_config();
    const auto header = engine::test::fake_header(config, 8);
    const auto layout = engine::plan_weight_layout(config, header);

    uint64_t end = 8;
    for (const auto& [name, t] : header.tensors) end = std::max(end, t.offset + t.bytes);
    std::vector<std::byte> file(end);
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (size_t i = 8; i + 1 < file.size(); i += 2) {
        const uint16_t h = engine::test::to_bf16(dist(rng));
        file[i] = std::byte(h & 0xFF);
        file[i + 1] = std::byte(h >> 8);
    }

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    {
        engine::DeviceBuffer device(layout.total_bytes);
        engine::upload_weights(file, layout, device.data(), stream);

        const auto expected = engine::checksum_file(file, header);
        const auto actual = engine::checksum_device(layout, device.data());
        ASSERT_EQ(actual.size(), expected.size());
        for (size_t i = 0; i < actual.size(); ++i) {
            SCOPED_TRACE(expected[i].name);
            EXPECT_EQ(actual[i].name, expected[i].name);
            EXPECT_EQ(actual[i].shape, expected[i].shape);
            EXPECT_EQ(actual[i].sum, expected[i].sum);
            EXPECT_EQ(actual[i].abs_sum, expected[i].abs_sum);
        }
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
}
