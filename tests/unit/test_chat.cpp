#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "engine/tokenizer/chat.h"

using engine::chat_prompt;
using engine::ChatMessage;

TEST(ChatPrompt, MatchesTheTinyLlamaTemplate) {
    const std::vector<ChatMessage> m{{"system", "You are a friendly chatbot."},
                                     {"user", "Explain KV caching"}};
    EXPECT_EQ(chat_prompt(m, true),
              "<|system|>\nYou are a friendly chatbot.</s>\n<|user|>\nExplain KV "
              "caching</s>\n<|assistant|>\n");
}

TEST(ChatPrompt, WithoutGenerationPrompt) {
    const std::vector<ChatMessage> m{{"user", "Hi"}};
    EXPECT_EQ(chat_prompt(m, false), "<|user|>\nHi</s>\n");
}

TEST(ChatPrompt, AssistantTurnsEndWithEos) {
    const std::vector<ChatMessage> m{{"user", "Hi"}, {"assistant", "Hello!"}};
    EXPECT_EQ(chat_prompt(m, false), "<|user|>\nHi</s>\n<|assistant|>\nHello!</s>\n");
}

TEST(ChatPrompt, NoMessagesGivesAnEmptyPrompt) {
    EXPECT_EQ(chat_prompt({}, true), "");
}

TEST(ChatPrompt, RejectsUnknownRoles) {
    const std::vector<ChatMessage> m{{"tool", "x"}};
    EXPECT_THROW(chat_prompt(m, true), std::invalid_argument);
}
