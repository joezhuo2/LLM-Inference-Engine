#include <gtest/gtest.h>

#include <cstddef>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine/loader/checksum.h"
#include "engine/loader/safetensors.h"
#include "support/bf16.h"
#include "support/safetensors_bytes.h"

using engine::checksum_bf16;
using engine::DType;
using engine::TensorChecksum;

namespace {

std::string bf16_bytes(const std::vector<float>& values) {
    std::string out;
    for (float v : values) {
        const uint16_t h = engine::test::to_bf16(v);
        out += char(h & 0xFF);
        out += char(h >> 8);
    }
    return out;
}

std::span<const std::byte> bytes_of(const std::string& s) {
    return engine::test::as_bytes(s);
}

}  // namespace

TEST(Checksum, SumsAndAbsSumsInDouble) {
    const std::string data = bf16_bytes({1.0f, -2.0f, 0.5f, 0.0f});
    const TensorChecksum c = checksum_bf16("w", {2, 2}, bytes_of(data));
    EXPECT_EQ(c.name, "w");
    EXPECT_EQ(c.dtype, DType::BF16);
    EXPECT_EQ(c.shape, (std::vector<int64_t>{2, 2}));
    EXPECT_DOUBLE_EQ(c.sum, -0.5);
    EXPECT_DOUBLE_EQ(c.abs_sum, 3.5);
}

TEST(Checksum, ReadsMisalignedData) {
    const std::string data = "x" + bf16_bytes({3.0f, 0.25f});
    const TensorChecksum c = checksum_bf16("w", {2}, bytes_of(data).subspan(1));
    EXPECT_DOUBLE_EQ(c.sum, 3.25);
}

TEST(Checksum, RejectsWrongByteSize) {
    const std::string data = bf16_bytes({1.0f, 2.0f, 3.0f});
    EXPECT_THROW(checksum_bf16("w", {2}, bytes_of(data)), std::invalid_argument);
}

TEST(Checksum, CoversEveryTensorOfAFileInNameOrder) {
    const std::string b = bf16_bytes({1.0f, 2.0f}), a = bf16_bytes({-4.0f});
    const std::string header = R"({"b": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]},)"
                               R"( "a": {"dtype": "BF16", "shape": [1], "data_offsets": [4, 6]}})";
    const std::string file = engine::test::safetensors_file(header, b + a);
    const auto h = engine::parse_safetensors_header(bytes_of(file));

    const auto sums = engine::checksum_file(bytes_of(file), h);
    ASSERT_EQ(sums.size(), 2u);
    EXPECT_EQ(sums[0].name, "a");
    EXPECT_DOUBLE_EQ(sums[0].sum, -4.0);
    EXPECT_EQ(sums[1].name, "b");
    EXPECT_DOUBLE_EQ(sums[1].sum, 3.0);
    EXPECT_EQ(sums[1].shape, (std::vector<int64_t>{2}));
}

TEST(Checksum, RejectsNonBf16Tensors) {
    const std::string header = R"({"x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}})";
    const std::string file = engine::test::safetensors_file(header, std::string(4, '\0'));
    const auto h = engine::parse_safetensors_header(bytes_of(file));
    EXPECT_THROW(engine::checksum_file(bytes_of(file), h), std::runtime_error);
}

TEST(Checksum, JsonRoundTripIsExact) {
    const std::vector<TensorChecksum> in = {{"b", DType::BF16, {2, 3}, 0.1 + 0.2, 1e-300},
                                            {"a", DType::BF16, {}, -1.0 / 3.0, 12345.678}};
    const auto out =
        engine::checksums_from_json(nlohmann::json::parse(engine::checksums_to_json(in).dump()));
    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0].name, "a");
    EXPECT_EQ(out[0].shape, std::vector<int64_t>{});
    EXPECT_EQ(out[0].sum, in[1].sum);
    EXPECT_EQ(out[1].name, "b");
    EXPECT_EQ(out[1].shape, (std::vector<int64_t>{2, 3}));
    EXPECT_EQ(out[1].sum, in[0].sum);
    EXPECT_EQ(out[1].abs_sum, in[0].abs_sum);
}

TEST(Checksum, JsonRejectsUnknownDtype) {
    const auto j = nlohmann::json::parse(
        R"({"tensors": {"x": {"dtype": "F64", "shape": [1], "sum": 0, "abs_sum": 0}}})");
    EXPECT_THROW(engine::checksums_from_json(j), std::runtime_error);
}
