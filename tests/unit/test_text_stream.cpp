#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

#include "engine/tokenizer/text_stream.h"
#include "engine/tokenizer/tokenizer.h"
#include "support/paths.h"

using engine::TextStream;
using engine::Tokenizer;

namespace {

class TextStreamTest : public ::testing::Test {
protected:
    void SetUp() override {
        const auto path = engine::test::model_dir() / "tokenizer.model";
        if (!std::filesystem::exists(path)) GTEST_SKIP() << path << " not found";
        tok.emplace(path);
    }

    std::string streamed(const std::vector<int32_t>& ids) const {
        TextStream stream(*tok);
        std::string out;
        for (const int32_t id : ids) out += stream.push(id);
        return out;
    }

    std::optional<Tokenizer> tok;
};

}  // namespace

TEST_F(TextStreamTest, EmitsEachWordAsItArrives) {
    TextStream stream(*tok);
    EXPECT_EQ(stream.push(15043), "Hello");
    EXPECT_EQ(stream.push(3186), " world");
    EXPECT_EQ(stream.push(2), "");
}

TEST_F(TextStreamTest, HoldsBackAPartialUtf8Character) {
    const auto ids = tok->encode("\xF0\x9F\x99\x82", false);
    ASSERT_GE(ids.size(), 4u);
    TextStream stream(*tok);
    std::string out;
    for (size_t i = 0; i + 1 < ids.size(); ++i) out += stream.push(ids[i]);
    EXPECT_EQ(out.find("\xEF\xBF\xBD"), std::string::npos);
    out += stream.push(ids.back());
    EXPECT_EQ(out, tok->decode(ids));
}

TEST_F(TextStreamTest, ConcatenationEqualsDecodingTheWholeSequenceOnTheTokenizerGolden) {
    const auto golden = engine::test::golden_dir() / "tokenizer.json";
    if (!std::filesystem::exists(golden)) GTEST_SKIP() << golden << " not found";
    const auto cases = nlohmann::json::parse(std::ifstream(golden)).at("cases");
    int failures = 0, checked = 0;
    for (const auto& c : cases) {
        const auto ids = c.at("ids").get<std::vector<int32_t>>();
        const std::string whole = tok->decode(ids);
        if (whole.ends_with("\xEF\xBF\xBD")) continue;
        ++checked;
        if (streamed(ids) != whole && ++failures <= 10)
            ADD_FAILURE() << "ids " << c.at("ids").dump();
    }
    EXPECT_EQ(failures, 0) << "of " << checked << " cases";
    EXPECT_GT(checked, 5000);
}
