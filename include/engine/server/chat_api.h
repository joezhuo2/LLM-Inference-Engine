#pragma once

#include <cstdint>
#include <nlohmann/json.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "engine/kv/block_manager.h"
#include "engine/sampling/params.h"
#include "engine/tokenizer/chat.h"

namespace engine::server {

class ApiError : public std::runtime_error {
public:
    ApiError(int status, const std::string& message)
        : std::runtime_error(message), status_(status) {}

    int status() const { return status_; }

private:
    int status_;
};

struct SamplingDefaults {
    float temperature = 0.0f;
    int32_t top_k = 0;
    float top_p = 1.0f;
    int max_tokens = 256;
};

struct ChatRequest {
    std::vector<ChatMessage> messages{};
    std::optional<int> max_tokens{};
    SamplingParams params{};
    std::optional<uint64_t> seed{};
    std::vector<std::string> stop{};
    bool stream = false;
};

ChatRequest parse_chat_request(std::string_view body, const SamplingDefaults& defaults);

int resolve_max_tokens(std::optional<int> requested, int default_tokens, int prompt_tokens,
                       int max_context);

enum class FinishReason { Stop, Length };

struct Completion {
    SeqId seq = 0;
    int64_t created = 0;
    std::string model;
};

std::string completion_id(SeqId seq);

nlohmann::ordered_json role_chunk(const Completion& c);
nlohmann::ordered_json content_chunk(const Completion& c, const std::string& text, int64_t step);
nlohmann::ordered_json finish_chunk(const Completion& c, FinishReason reason, int64_t step);
nlohmann::ordered_json completion_json(const Completion& c, const std::string& text,
                                       FinishReason reason, int64_t step, int prompt_tokens,
                                       int completion_tokens);
nlohmann::ordered_json error_json(const std::string& message);

std::string sse_data(std::string_view data);
std::string sse_event(std::string_view event, std::string_view data);

}  // namespace engine::server
