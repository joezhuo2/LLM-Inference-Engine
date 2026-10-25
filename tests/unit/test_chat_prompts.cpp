#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine/bench/chat_prompts.h"
#include "engine/tokenizer/chat.h"
#include "engine/tokenizer/tokenizer.h"
#include "support/paths.h"

using engine::ChatMessage;
using engine::load_chat_prompts;
using engine::Tokenizer;

namespace {

class ChatPromptsTest : public ::testing::Test {
protected:
    void SetUp() override {
        const auto path = engine::test::model_dir() / "tokenizer.model";
        if (!std::filesystem::exists(path)) GTEST_SKIP() << path << " not found";
        tok.emplace(path);
    }

    std::filesystem::path write(const std::string& name, const std::string& text) {
        const auto path = std::filesystem::temp_directory_path() / ("llmie_" + name + ".json");
        std::ofstream(path) << text;
        return path;
    }

    std::optional<Tokenizer> tok;
};

}  // namespace

TEST_F(ChatPromptsTest, EachPromptIsTheTokenizedChatTemplate) {
    const auto path = write("two", R"({"prompts": [
        {"name": "a", "messages": [{"role": "user", "content": "What is 2 + 2?"}]},
        {"messages": [{"role": "system", "content": "Be brief."},
                      {"role": "user", "content": "Name a color."}]}]})");
    const auto prompts = load_chat_prompts(path, *tok);
    ASSERT_EQ(prompts.size(), 2u);
    const std::vector<ChatMessage> second{{"system", "Be brief."}, {"user", "Name a color."}};
    EXPECT_EQ(prompts[1], tok->encode(engine::chat_prompt(second, true), true));
    EXPECT_EQ(prompts[0].front(), tok->bos_id());
}

TEST_F(ChatPromptsTest, LoadsTheTwentyPromptsOfTheRepository) {
    const auto prompts =
        load_chat_prompts(std::filesystem::path(ENGINE_SCRIPTS_DIR) / "prompts.json", *tok);
    EXPECT_EQ(prompts.size(), 20u);
    for (const auto& p : prompts) EXPECT_GT(p.size(), 5u);
}

TEST_F(ChatPromptsTest, RejectsMalformedFiles) {
    for (const std::string& text :
         {std::string("not json"), std::string("[]"), std::string(R"({"prompts": []})"),
          std::string(R"({"prompts": [{"messages": []}]})"),
          std::string(R"({"prompts": [{"messages": [{"role": "user"}]}]})"),
          std::string(R"({"prompts": [{"messages": [{"role": "tool", "content": "x"}]}]})")}) {
        SCOPED_TRACE(text);
        EXPECT_THROW(load_chat_prompts(write("bad", text), *tok), std::invalid_argument);
    }
    EXPECT_THROW(load_chat_prompts("/nonexistent/prompts.json", *tok), std::invalid_argument);
}
