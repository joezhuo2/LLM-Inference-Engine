#include <gtest/gtest.h>
#include <httplib.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "engine/runtime/generate.h"
#include "engine/runtime/model_runner.h"
#include "engine/server/gpu_backend.h"
#include "engine/server/http_server.h"
#include "engine/server/serving_loop.h"
#include "engine/tokenizer/chat.h"
#include "engine/tokenizer/text_stream.h"
#include "engine/tokenizer/tokenizer.h"
#include "support/model.h"
#include "support/paths.h"

using engine::ChatMessage;
using engine::Tokenizer;
using Json = nlohmann::json;

namespace {

constexpr int kMaxTokens = 64;
const std::vector<ChatMessage> kMessages = {{"user", "What is the capital of Canada?"}};

struct Answer {
    std::string text;
    std::string finish_reason;
};

Answer generate_answer(const Tokenizer& tok) {
    const auto& w = engine::test::tinyllama();
    const auto prompt = tok.encode(engine::chat_prompt(kMessages, true), false);
    engine::ModelRunner runner(w.config, w.weights, int64_t(prompt.size()),
                               int64_t(prompt.size()) + kMaxTokens, nullptr);
    engine::TextStream stream(tok);
    Answer a;
    const auto out = engine::generate(runner, prompt, kMaxTokens, w.config.eos_id, {},
                                      [&](int32_t token) { a.text += stream.push(token); });
    a.finish_reason = out.back() == w.config.eos_id ? "stop" : "length";
    return a;
}

Answer streamed_answer(int port) {
    httplib::Client client("127.0.0.1", port);
    client.set_read_timeout(60, 0);
    Json messages = Json::array();
    for (const auto& m : kMessages) messages.push_back({{"role", m.role}, {"content", m.content}});
    const Json body = {
        {"messages", messages}, {"max_tokens", kMaxTokens}, {"temperature", 0}, {"stream", true}};
    const auto res = client.Post("/v1/chat/completions", body.dump(), "application/json");
    Answer a;
    if (!res || res->status != 200) return a;
    size_t pos = 0;
    while ((pos = res->body.find("data: ", pos)) != std::string::npos) {
        const size_t end = res->body.find("\n\n", pos);
        const std::string data = res->body.substr(pos + 6, end - pos - 6);
        pos = end;
        if (data == "[DONE]") break;
        const Json choice = Json::parse(data)["choices"][0];
        if (choice["delta"].contains("content")) a.text += choice["delta"]["content"];
        if (!choice["finish_reason"].is_null()) a.finish_reason = choice["finish_reason"];
    }
    return a;
}

class ServerTest : public ::testing::Test {
protected:
    void SetUp() override {
        const auto path = engine::test::model_dir() / "tokenizer.model";
        if (!std::filesystem::exists(path)) GTEST_SKIP() << path << " not found";
        tok.emplace(path);
    }

    std::optional<Tokenizer> tok;
};

}  // namespace

TEST_F(ServerTest, StreamsTheGreedyAnswerOfGenerate) {
    const Answer expected = generate_answer(*tok);
    ASSERT_FALSE(expected.text.empty());

    const auto& w = engine::test::tinyllama();
    const auto backend = engine::server::make_gpu_backend(w, {.max_batch = 4, .graphs = true});
    engine::server::ServingLoop loop(
        *backend, [&](std::span<const int32_t> ids) { return tok->decode(ids); }, w.config.eos_id);
    engine::server::HttpServer http(
        loop, [&](const std::string& text) { return tok->encode(text, false); },
        {.max_context = w.config.max_positions});
    const int port = http.bind("127.0.0.1", 0);
    std::jthread engine([&](std::stop_token stop) { loop.run(stop); });
    std::jthread listener([&] { http.listen(); });

    std::vector<Answer> answers(4);
    {
        std::vector<std::jthread> clients;
        for (auto& a : answers) clients.emplace_back([&a, port] { a = streamed_answer(port); });
    }
    http.stop();
    listener.join();

    for (const Answer& a : answers) {
        EXPECT_EQ(a.text, expected.text);
        EXPECT_EQ(a.finish_reason, expected.finish_reason);
    }
    const auto m = loop.metrics();
    EXPECT_EQ(m.counters.finished_total, 4);
    EXPECT_EQ(m.counters.blocks_free, backend->num_blocks() - 1);
}
