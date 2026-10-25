#include "engine/bench/chat_prompts.h"

#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

#include "engine/tokenizer/chat.h"

namespace engine {

std::vector<std::vector<int32_t>> load_chat_prompts(const std::filesystem::path& file,
                                                    const Tokenizer& tokenizer) {
    const auto reject = [&](const std::string& why) {
        throw std::invalid_argument("chat prompts " + file.string() + ": " + why);
    };
    std::ifstream in(file);
    if (!in) reject("cannot be read");
    const nlohmann::json j = nlohmann::json::parse(in, nullptr, false);
    if (j.is_discarded()) reject("is not valid JSON");
    if (!j.is_object() || !j.contains("prompts") || !j["prompts"].is_array())
        reject("expected an object with a \"prompts\" array");
    if (j["prompts"].empty()) reject("the \"prompts\" array is empty");

    std::vector<std::vector<int32_t>> prompts;
    for (const auto& prompt : j["prompts"]) {
        if (!prompt.is_object() || !prompt.contains("messages") || !prompt["messages"].is_array() ||
            prompt["messages"].empty())
            reject("every prompt needs a non-empty \"messages\" array");
        std::vector<ChatMessage> messages;
        for (const auto& m : prompt["messages"]) {
            if (!m.is_object() || !m.contains("role") || !m["role"].is_string() ||
                !m.contains("content") || !m["content"].is_string())
                reject("every message needs string \"role\" and \"content\"");
            messages.push_back({m["role"].get<std::string>(), m["content"].get<std::string>()});
        }
        prompts.push_back(tokenizer.encode(chat_prompt(messages, true), true));
    }
    return prompts;
}

}  // namespace engine
