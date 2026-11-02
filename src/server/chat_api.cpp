#include "engine/server/chat_api.h"

#include <algorithm>
#include <cstdint>
#include <utility>

namespace engine::server {

namespace {

using Json = nlohmann::json;

[[noreturn]] void reject(const std::string& message) {
    throw ApiError(400, message);
}

const Json* field(const Json& body, const char* name) {
    const auto it = body.find(name);
    return it == body.end() || it->is_null() ? nullptr : &*it;
}

std::vector<ChatMessage> parse_messages(const Json& body) {
    const Json* messages = field(body, "messages");
    if (!messages || !messages->is_array() || messages->empty())
        reject("\"messages\" must be a non-empty array");
    std::vector<ChatMessage> out;
    for (const Json& m : *messages) {
        if (!m.is_object()) reject("every message must be an object");
        const Json* role = field(m, "role");
        const Json* content = field(m, "content");
        if (!role || !role->is_string()) reject("every message needs a string \"role\"");
        if (!content || !content->is_string()) reject("every message needs a string \"content\"");
        ChatMessage message{role->get<std::string>(), content->get<std::string>()};
        if (message.role != "system" && message.role != "user" && message.role != "assistant")
            reject("unsupported role \"" + message.role + "\"; use system, user or assistant");
        out.push_back(std::move(message));
    }
    return out;
}

float number(const Json& body, const char* name, float fallback) {
    const Json* value = field(body, name);
    if (!value) return fallback;
    if (!value->is_number()) reject(std::string("\"") + name + "\" must be a number");
    return value->get<float>();
}

std::optional<int64_t> integer(const Json& body, const char* name) {
    const Json* value = field(body, name);
    if (!value) return std::nullopt;
    if (!value->is_number_integer()) reject(std::string("\"") + name + "\" must be an integer");
    return value->get<int64_t>();
}

std::vector<std::string> parse_stop(const Json& body) {
    const Json* stop = field(body, "stop");
    if (!stop) return {};
    if (stop->is_string()) {
        if (stop->get_ref<const std::string&>().empty()) return {};
        return {stop->get<std::string>()};
    }
    if (!stop->is_array()) reject("\"stop\" must be a string or a list of strings");
    std::vector<std::string> out;
    for (const Json& s : *stop) {
        if (!s.is_string()) reject("\"stop\" must be a string or a list of strings");
        if (!s.get_ref<const std::string&>().empty()) out.push_back(s.get<std::string>());
    }
    return out;
}

bool parse_stream(const Json& body) {
    const Json* stream = field(body, "stream");
    if (!stream) return false;
    if (!stream->is_boolean()) reject("\"stream\" must be true or false");
    return stream->get<bool>();
}

nlohmann::ordered_json chunk(const Completion& c, nlohmann::ordered_json delta,
                             nlohmann::ordered_json finish, nlohmann::ordered_json engine) {
    return {{"id", completion_id(c.seq)},
            {"object", "chat.completion.chunk"},
            {"created", c.created},
            {"model", c.model},
            {"choices",
             {{{"index", 0}, {"delta", std::move(delta)}, {"finish_reason", std::move(finish)}}}},
            {"engine", std::move(engine)}};
}

const char* reason_name(FinishReason reason) {
    return reason == FinishReason::Stop ? "stop" : "length";
}

}  // namespace

ChatRequest parse_chat_request(std::string_view body, const SamplingDefaults& defaults) {
    const Json j = Json::parse(body, nullptr, false);
    if (j.is_discarded()) reject("the body is not valid JSON");
    if (!j.is_object()) reject("the body must be a JSON object");

    ChatRequest r;
    r.messages = parse_messages(j);
    if (const auto max_tokens = integer(j, "max_tokens")) {
        if (*max_tokens < 1 || *max_tokens > INT32_MAX) reject("\"max_tokens\" must be positive");
        r.max_tokens = int(*max_tokens);
    }
    r.params.temperature = number(j, "temperature", defaults.temperature);
    r.params.top_p = number(j, "top_p", defaults.top_p);
    const auto top_k = integer(j, "top_k");
    if (top_k && (*top_k < 0 || *top_k > INT32_MAX)) reject("\"top_k\" must be zero or positive");
    r.params.top_k = top_k ? int32_t(*top_k) : defaults.top_k;
    if (const auto seed = integer(j, "seed")) r.seed = uint64_t(*seed);
    try {
        validate(r.params);
    } catch (const std::invalid_argument& e) {
        reject(e.what());
    }
    r.stop = parse_stop(j);
    r.stream = parse_stream(j);
    return r;
}

int resolve_max_tokens(std::optional<int> requested, int default_tokens, int prompt_tokens,
                       int max_context) {
    const int room = max_context - prompt_tokens + 1;
    if (room < 1)
        reject("the prompt has " + std::to_string(prompt_tokens) + " tokens; the context is " +
               std::to_string(max_context));
    if (!requested) return std::min(default_tokens, room);
    if (*requested > room)
        reject("the prompt (" + std::to_string(prompt_tokens) + " tokens) plus max_tokens (" +
               std::to_string(*requested) + ") does not fit in " + std::to_string(max_context) +
               " tokens of context");
    return *requested;
}

std::string completion_id(SeqId seq) {
    return "chatcmpl-" + std::to_string(seq);
}

nlohmann::ordered_json role_chunk(const Completion& c) {
    return chunk(c, {{"role", "assistant"}}, nullptr, {{"seq_id", c.seq}});
}

nlohmann::ordered_json content_chunk(const Completion& c, const std::string& text, int64_t step) {
    return chunk(c, {{"content", text}}, nullptr, {{"seq_id", c.seq}, {"step", step}});
}

nlohmann::ordered_json finish_chunk(const Completion& c, FinishReason reason, int64_t step) {
    return chunk(c, nlohmann::ordered_json::object(), reason_name(reason),
                 {{"seq_id", c.seq}, {"step", step}});
}

nlohmann::ordered_json completion_json(const Completion& c, const std::string& text,
                                       FinishReason reason, int64_t step, int prompt_tokens,
                                       int completion_tokens) {
    return {{"id", completion_id(c.seq)},
            {"object", "chat.completion"},
            {"created", c.created},
            {"model", c.model},
            {"choices",
             {{{"index", 0},
               {"message", {{"role", "assistant"}, {"content", text}}},
               {"finish_reason", reason_name(reason)}}}},
            {"usage",
             {{"prompt_tokens", prompt_tokens},
              {"completion_tokens", completion_tokens},
              {"total_tokens", prompt_tokens + completion_tokens}}},
            {"engine", {{"seq_id", c.seq}, {"step", step}}}};
}

nlohmann::ordered_json error_json(const std::string& message) {
    return {{"error", {{"message", message}, {"type", "invalid_request_error"}}}};
}

std::string sse_data(std::string_view data) {
    return "data: " + std::string(data) + "\n\n";
}

std::string sse_event(std::string_view event, std::string_view data) {
    return "event: " + std::string(event) + "\n" + sse_data(data);
}

}  // namespace engine::server
