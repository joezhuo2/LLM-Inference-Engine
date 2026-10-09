#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine/loader/mapped_file.h"
#include "engine/loader/safetensors.h"
#include "support/paths.h"
#include "support/safetensors_bytes.h"

using engine::DType;
using engine::parse_safetensors_header;
using engine::SafetensorsHeader;
using engine::test::as_bytes;
using engine::test::safetensors_file;

namespace {

std::string entry(const std::string& dtype, const std::string& shape, uint64_t begin,
                  uint64_t end) {
    return R"({"dtype": ")" + dtype + R"(", "shape": )" + shape + R"(, "data_offsets": [)" +
           std::to_string(begin) + ", " + std::to_string(end) + "]}";
}

SafetensorsHeader parse(const std::string& file) {
    return parse_safetensors_header(as_bytes(file));
}

void expect_rejected(const std::string& file) {
    EXPECT_THROW(parse(file), std::runtime_error);
}

void expect_rejected_header(const std::string& header, size_t data_bytes) {
    expect_rejected(safetensors_file(header, std::string(data_bytes, '\0')));
}

}  // namespace

TEST(Safetensors, ParsesTwoTensors) {
    const std::string header = R"({"x": )" + entry("BF16", "[2, 3]", 0, 12) + R"(, "y": )" +
                               entry("F32", "[1]", 12, 16) + "}";
    const SafetensorsHeader h = parse(safetensors_file(header, std::string(16, '\0')));
    const uint64_t data = 8 + header.size();
    EXPECT_EQ(h.data_offset, data);
    ASSERT_EQ(h.tensors.size(), 2u);
    const auto& x = h.tensors.at("x");
    EXPECT_EQ(x.dtype, DType::BF16);
    EXPECT_EQ(x.shape, (std::vector<int64_t>{2, 3}));
    EXPECT_EQ(x.offset, data);
    EXPECT_EQ(x.bytes, 12u);
    const auto& y = h.tensors.at("y");
    EXPECT_EQ(y.dtype, DType::F32);
    EXPECT_EQ(y.offset, data + 12);
    EXPECT_EQ(y.bytes, 4u);
}

TEST(Safetensors, MapsEverySupportedDtype) {
    const std::string header =
        R"({"a": )" + entry("BF16", "[1]", 0, 2) + R"(, "b": )" + entry("F16", "[1]", 2, 4) +
        R"(, "c": )" + entry("F32", "[1]", 4, 8) + R"(, "d": )" + entry("I32", "[1]", 8, 12) +
        R"(, "e": )" + entry("I64", "[1]", 12, 20) + "}";
    const SafetensorsHeader h = parse(safetensors_file(header, std::string(20, '\0')));
    EXPECT_EQ(h.tensors.at("a").dtype, DType::BF16);
    EXPECT_EQ(h.tensors.at("b").dtype, DType::F16);
    EXPECT_EQ(h.tensors.at("c").dtype, DType::F32);
    EXPECT_EQ(h.tensors.at("d").dtype, DType::I32);
    EXPECT_EQ(h.tensors.at("e").dtype, DType::I64);
}

TEST(Safetensors, SkipsMetadata) {
    for (const std::string meta : {R"({"format": "pt"})", "null"}) {
        const std::string header =
            R"({"__metadata__": )" + meta + R"(, "x": )" + entry("BF16", "[2]", 0, 4) + "}";
        const SafetensorsHeader h = parse(safetensors_file(header, std::string(4, '\0')));
        EXPECT_EQ(h.tensors.size(), 1u) << meta;
        EXPECT_FALSE(h.tensors.contains("__metadata__")) << meta;
    }
}

TEST(Safetensors, AllowsWhitespacePaddingAroundTheObject) {
    const std::string header =
        std::string("  ") + R"({"x": )" + entry("BF16", "[2]", 0, 4) + "}" + "      ";
    const SafetensorsHeader h = parse(safetensors_file(header, std::string(4, '\0')));
    EXPECT_EQ(h.data_offset, 8 + header.size());
    EXPECT_EQ(h.tensors.at("x").offset, 8 + header.size());
}

TEST(Safetensors, IgnoresUnknownEntryFields) {
    const std::string header =
        R"({"x": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2], "extra": 1}})";
    EXPECT_EQ(parse(safetensors_file(header, std::string(2, '\0'))).tensors.at("x").bytes, 2u);
}

TEST(Safetensors, ScalarHasEmptyShape) {
    const std::string header = R"({"s": )" + entry("F32", "[]", 0, 4) + "}";
    const SafetensorsHeader h = parse(safetensors_file(header, std::string(4, '\0')));
    const auto& s = h.tensors.at("s");
    EXPECT_TRUE(s.shape.empty());
    EXPECT_EQ(s.bytes, 4u);
}

TEST(Safetensors, ZeroSizedTensorsAreValid) {
    const std::string header = R"({"empty": )" + entry("BF16", "[0, 3]", 0, 0) + R"(, "x": )" +
                               entry("BF16", "[1]", 0, 2) + "}";
    const SafetensorsHeader h = parse(safetensors_file(header, std::string(2, '\0')));
    EXPECT_EQ(h.tensors.at("empty").bytes, 0u);
    EXPECT_EQ(h.tensors.at("empty").shape, (std::vector<int64_t>{0, 3}));
    EXPECT_EQ(h.tensors.at("x").bytes, 2u);
}

TEST(Safetensors, EmptyDataSectionWithNoTensors) {
    const SafetensorsHeader h = parse(safetensors_file("{}"));
    EXPECT_TRUE(h.tensors.empty());
    EXPECT_EQ(h.data_offset, 10u);
}

TEST(Safetensors, JsonOrderDoesNotMatter) {
    const std::string header =
        R"({"y": )" + entry("BF16", "[1]", 2, 4) + R"(, "x": )" + entry("BF16", "[1]", 0, 2) + "}";
    const SafetensorsHeader h = parse(safetensors_file(header, std::string(4, '\0')));
    EXPECT_EQ(h.tensors.at("x").offset, h.data_offset);
    EXPECT_EQ(h.tensors.at("y").offset, h.data_offset + 2);
}

TEST(Safetensors, DuplicateKeyKeepsTheLastOccurrence) {
    const std::string header =
        R"({"x": )" + entry("BF16", "[1]", 0, 2) + R"(, "x": )" + entry("BF16", "[2]", 0, 4) + "}";
    const SafetensorsHeader h = parse(safetensors_file(header, std::string(4, '\0')));
    ASSERT_EQ(h.tensors.size(), 1u);
    EXPECT_EQ(h.tensors.at("x").shape, (std::vector<int64_t>{2}));
}

TEST(Safetensors, NamesSortAsPlainStrings) {
    const std::string header = R"({"layers.2": )" + entry("BF16", "[1]", 0, 2) +
                               R"(, "layers.10": )" + entry("BF16", "[1]", 2, 4) + "}";
    const SafetensorsHeader h = parse(safetensors_file(header, std::string(4, '\0')));
    EXPECT_EQ(h.tensors.begin()->first, "layers.10");
}

TEST(Safetensors, RejectsFilesShorterThanThePrefix) {
    expect_rejected("");
    expect_rejected(std::string(3, '\0'));
    expect_rejected(std::string(7, '\0'));
}

TEST(Safetensors, RejectsHeaderLengthPastTheEnd) {
    const std::string header = R"({"x": )" + entry("BF16", "[1]", 0, 2) + "}";
    expect_rejected(safetensors_file(header.size() + 3, header, "\0\0"));
}

TEST(Safetensors, RejectsHeaderLengthNearTwoToThe64) {
    expect_rejected(safetensors_file(UINT64_MAX, "{}", ""));
    expect_rejected(safetensors_file(UINT64_MAX - 7, "{}", ""));
}

TEST(Safetensors, RejectsHeadersOverTheSizeLimit) {
    std::string header = "{}";
    header.resize(100'000'001, ' ');
    expect_rejected(safetensors_file(header));
}

TEST(Safetensors, RejectsMalformedJson) {
    expect_rejected(safetensors_file(R"({"x": )"));
    expect_rejected(safetensors_file("    "));
    expect_rejected(safetensors_file(""));
    expect_rejected(safetensors_file("[]"));
    expect_rejected(safetensors_file("{\"x\xff\": 1}"));
}

TEST(Safetensors, RejectsMissingOrMistypedFields) {
    expect_rejected_header(R"({"x": 5})", 0);
    expect_rejected_header(R"({"x": {"shape": [1], "data_offsets": [0, 2]}})", 2);
    expect_rejected_header(R"({"x": {"dtype": "BF16", "data_offsets": [0, 2]}})", 2);
    expect_rejected_header(R"({"x": {"dtype": "BF16", "shape": [1]}})", 2);
    expect_rejected_header(R"({"x": {"dtype": 7, "shape": [1], "data_offsets": [0, 2]}})", 2);
    expect_rejected_header(R"({"x": {"dtype": "BF16", "shape": "1", "data_offsets": [0, 2]}})", 2);
}

TEST(Safetensors, RejectsNonIntegerNumbers) {
    expect_rejected_header(R"({"x": {"dtype": "BF16", "shape": [1.0], "data_offsets": [0, 2]}})",
                           2);
    expect_rejected_header(R"({"x": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2.0]}})",
                           2);
    expect_rejected_header(R"({"x": {"dtype": "BF16", "shape": [1e0], "data_offsets": [0, 2]}})",
                           2);
}

TEST(Safetensors, RejectsBadMetadata) {
    const std::string x = entry("BF16", "[1]", 0, 2);
    expect_rejected_header(R"({"__metadata__": {"k": 1}, "x": )" + x + "}", 2);
    expect_rejected_header(R"({"__metadata__": [], "x": )" + x + "}", 2);
}

TEST(Safetensors, RejectsUnsupportedDtypesByName) {
    for (const std::string dtype : {"F64", "I8", "U8", "BOOL", "F8_E4M3", "bf16", "Q4"}) {
        const std::string header = R"({"weight": )" + entry(dtype, "[1]", 0, 8) + "}";
        try {
            parse(safetensors_file(header, std::string(8, '\0')));
            ADD_FAILURE() << dtype << " was accepted";
        } catch (const std::runtime_error& e) {
            const std::string what = e.what();
            EXPECT_NE(what.find("weight"), std::string::npos) << what;
            EXPECT_NE(what.find(dtype), std::string::npos) << what;
        }
    }
}

TEST(Safetensors, RejectsNegativeDimensions) {
    expect_rejected_header(R"({"x": )" + entry("BF16", "[-2]", 0, 4) + "}", 4);
}

TEST(Safetensors, RejectsDataOffsetsWithWrongArity) {
    expect_rejected_header(R"({"x": {"dtype": "BF16", "shape": [1], "data_offsets": [2]}})", 2);
    expect_rejected_header(R"({"x": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2, 2]}})",
                           2);
    expect_rejected_header(R"({"x": {"dtype": "BF16", "shape": [1], "data_offsets": [-1, 1]}})", 2);
}

TEST(Safetensors, RejectsBeginAfterEnd) {
    expect_rejected_header(R"({"x": )" + entry("BF16", "[0]", 4, 2) + "}", 4);
}

TEST(Safetensors, RejectsRangesPastTheEnd) {
    expect_rejected_header(R"({"x": )" + entry("BF16", "[2]", 0, 4) + "}", 2);
    expect_rejected_header(R"({"x": )" + entry("BF16", "[1]", 0, 2) + "}", 0);
}

TEST(Safetensors, RejectsGapsOverlapsAndUncoveredBytes) {
    expect_rejected_header(
        R"({"x": )" + entry("BF16", "[1]", 0, 2) + R"(, "y": )" + entry("BF16", "[1]", 4, 6) + "}",
        6);
    expect_rejected_header(
        R"({"x": )" + entry("BF16", "[2]", 0, 4) + R"(, "y": )" + entry("BF16", "[1]", 2, 4) + "}",
        4);
    expect_rejected_header(R"({"x": )" + entry("BF16", "[1]", 0, 2) + "}", 4);
    expect_rejected_header(R"({"x": )" + entry("BF16", "[1]", 2, 4) + "}", 4);
}

TEST(Safetensors, RejectsSizeThatDisagreesWithShape) {
    expect_rejected_header(R"({"x": )" + entry("BF16", "[3]", 0, 4) + "}", 4);
    expect_rejected_header(R"({"x": )" + entry("F32", "[2]", 0, 4) + "}", 4);
}

TEST(Safetensors, RejectsShapesWhoseByteSizeOverflows) {
    expect_rejected_header(R"({"x": )" + entry("BF16", "[4611686018427387904, 4]", 0, 0) + "}", 0);
    expect_rejected_header(R"({"x": )" + entry("BF16", "[9223372036854775807, 2]", 0, 2) + "}", 2);
}

TEST(Safetensors, RealTinyLlamaHeader) {
    const auto path = engine::test::model_dir() / "model.safetensors";
    if (!std::filesystem::exists(path)) GTEST_SKIP() << path << " not found";
    const engine::MappedFile file(path);
    const SafetensorsHeader h = parse_safetensors_header(file.bytes());

    EXPECT_EQ(h.tensors.size(), 201u);
    uint64_t total = 0;
    for (const auto& [name, t] : h.tensors) {
        EXPECT_EQ(t.dtype, DType::BF16) << name;
        EXPECT_GE(t.offset, h.data_offset) << name;
        EXPECT_LE(t.offset + t.bytes, file.bytes().size()) << name;
        total += t.bytes;
    }
    EXPECT_EQ(total, file.bytes().size() - h.data_offset);

    EXPECT_EQ(h.tensors.at("model.embed_tokens.weight").shape, (std::vector<int64_t>{32000, 2048}));
    EXPECT_EQ(h.tensors.at("model.embed_tokens.weight").bytes, 32000u * 2048u * 2u);
    EXPECT_EQ(h.tensors.at("lm_head.weight").shape, (std::vector<int64_t>{32000, 2048}));
    EXPECT_EQ(h.tensors.at("model.norm.weight").shape, (std::vector<int64_t>{2048}));
    EXPECT_EQ(h.tensors.at("model.layers.21.self_attn.k_proj.weight").shape,
              (std::vector<int64_t>{256, 2048}));
    EXPECT_EQ(h.tensors.at("model.layers.0.mlp.down_proj.weight").shape,
              (std::vector<int64_t>{2048, 5632}));
}
