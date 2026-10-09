#include "engine/tokenizer/chat.h"

#include <stdexcept>

namespace engine {

std::string chat_prompt(std::span<const ChatMessage> messages, bool add_generation_prompt) {
    std::string out;
    for (const ChatMessage& m : messages) {
        if (m.role != "system" && m.role != "user" && m.role != "assistant")
            throw std::invalid_argument("chat: unsupported role '" + m.role + "'");
        out += "<|" + m.role + "|>\n" + m.content + "</s>\n";
    }
    if (add_generation_prompt && !messages.empty()) out += "<|assistant|>\n";
    return out;
}

}  // namespace engine
