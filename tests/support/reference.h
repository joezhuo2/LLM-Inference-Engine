#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "engine/tokenizer/chat.h"
#include "support/paths.h"

namespace engine::test {

struct ReferencePrompt {
    std::string name;
    std::filesystem::path file;
    std::vector<ChatMessage> messages;
    int prompt_len = 0;
    int total_len = 0;
    std::vector<int32_t> generated_ids;
    bool trace = false;
};

inline std::filesystem::path reference_dir() {
    return golden_dir() / "reference";
}

inline std::vector<ReferencePrompt> load_reference_manifest() {
    const auto dir = reference_dir();
    const auto j = nlohmann::json::parse(std::ifstream(dir / "manifest.json"));
    std::vector<ReferencePrompt> out;
    for (const auto& p : j.at("prompts")) {
        ReferencePrompt r;
        r.name = p.at("name").get<std::string>();
        r.file = dir / p.at("file").get<std::string>();
        for (const auto& m : p.at("messages"))
            r.messages.push_back(
                {m.at("role").get<std::string>(), m.at("content").get<std::string>()});
        r.prompt_len = p.at("prompt_len").get<int>();
        r.total_len = p.at("total_len").get<int>();
        r.generated_ids = p.at("generated_ids").get<std::vector<int32_t>>();
        r.trace = p.at("trace").get<bool>();
        out.push_back(std::move(r));
    }
    return out;
}

}  // namespace engine::test
