#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine/tokenizer/chat.h"
#include "engine/tokenizer/tokenizer.h"
#include "support/paths.h"

using engine::Tokenizer;
using Ids = std::vector<int32_t>;

namespace {

class TokenizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        const auto path = engine::test::model_dir() / "tokenizer.model";
        if (!std::filesystem::exists(path)) GTEST_SKIP() << path << " not found";
        tok.emplace(path);
    }

    std::optional<Tokenizer> tok;
};

}  // namespace

TEST_F(TokenizerTest, SpecialIdsAndVocab) {
    EXPECT_EQ(tok->vocab_size(), 32000);
    EXPECT_EQ(tok->bos_id(), 1);
    EXPECT_EQ(tok->eos_id(), 2);
}

TEST_F(TokenizerTest, EncodesLikeHuggingFace) {
    EXPECT_EQ(tok->encode("Hello world", true), (Ids{1, 15043, 3186}));
    EXPECT_EQ(tok->encode("Hello world", false), (Ids{15043, 3186}));
    EXPECT_EQ(tok->encode("\nnewline", true), (Ids{1, 29871, 13, 1482, 1220}));
    EXPECT_EQ(tok->encode("日本語 🙂", false),
              (Ids{29871, 30325, 30346, 30968, 29871, 243, 162, 156, 133}));
}

TEST_F(TokenizerTest, LeadingSpaceReplacesTheDummyPrefix) {
    EXPECT_EQ(tok->encode(" Hello", true), (Ids{1, 15043}));
    EXPECT_EQ(tok->encode("  leading spaces", true), (Ids{1, 29871, 8236, 8162}));
}

TEST_F(TokenizerTest, SpecialTokenTextBecomesItsId) {
    EXPECT_EQ(tok->encode("</s> literal", true), (Ids{1, 2, 16333}));
    EXPECT_EQ(tok->encode("a</s>b", true), (Ids{1, 263, 2, 29890}));
}

TEST_F(TokenizerTest, EmptyText) {
    EXPECT_EQ(tok->encode("", true), (Ids{1}));
    EXPECT_TRUE(tok->encode("", false).empty());
}

TEST_F(TokenizerTest, DecodesLikeHuggingFace) {
    EXPECT_EQ(tok->decode(Ids{15043, 3186}), "Hello world");
    EXPECT_EQ(tok->decode(Ids{1, 15043, 3186, 2}), "Hello world");
    EXPECT_EQ(tok->decode(Ids{29871, 15043}), " Hello");
    EXPECT_EQ(tok->decode(Ids{0, 263}), "a");
    EXPECT_EQ(tok->decode(Ids{30325, 30346, 30968, 29871, 243, 162, 156, 133}), "日本語 🙂");
}

TEST_F(TokenizerTest, DecodeRejectsOutOfRangeIds) {
    EXPECT_THROW(tok->decode(Ids{32000}), std::out_of_range);
    EXPECT_THROW(tok->decode(Ids{-1}), std::out_of_range);
}

TEST_F(TokenizerTest, ChatPromptMatchesApplyChatTemplate) {
    const std::vector<engine::ChatMessage> m{{"user", "Hi"}};
    EXPECT_EQ(tok->encode(engine::chat_prompt(m, true), false),
              (Ids{529, 29989, 1792, 29989, 29958, 13, 18567, 2, 13, 29966, 29989, 465, 22137,
                   29989, 29958, 13}));
}

TEST(Tokenizer, MissingModelThrows) {
    EXPECT_THROW(Tokenizer{std::filesystem::path("/nonexistent/tokenizer.model")},
                 std::runtime_error);
}
