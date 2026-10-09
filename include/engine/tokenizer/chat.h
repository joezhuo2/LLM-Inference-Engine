#pragma once

#include <span>
#include <string>

namespace engine {

struct ChatMessage {
    std::string role;
    std::string content;
};

std::string chat_prompt(std::span<const ChatMessage> messages, bool add_generation_prompt);

}  // namespace engine
