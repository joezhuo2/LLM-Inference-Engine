#include <gtest/gtest.h>
#include <httplib.h>

#include <chrono>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <thread>
#include <vector>

#include "engine/server/http_server.h"
#include "engine/server/serving_loop.h"
#include "support/scripted_backend.h"

using engine::Request;
using engine::server::HttpOptions;
using engine::server::HttpServer;
using engine::server::ServingLoop;
using engine::test::kEos;
using engine::test::ScriptedBackend;
using engine::test::word_decode;
using Json = nlohmann::json;
using std::chrono::milliseconds;

namespace {

constexpr int kContext = 256;
const char* const kDevOrigin = "http://localhost:3000";

std::vector<int32_t> byte_encode(const std::string& text) {
    std::vector<int32_t> ids;
    for (const char c : text) ids.push_back(100 + int32_t(static_cast<unsigned char>(c) % 50));
    return ids;
}

std::vector<std::string> sse_data_lines(const std::string& body) {
    std::vector<std::string> out;
    size_t pos = 0;
    while ((pos = body.find("data: ", pos)) != std::string::npos) {
        const size_t end = body.find("\n\n", pos);
        out.push_back(body.substr(pos + 6, end - pos - 6));
        pos = end;
    }
    return out;
}

std::string chat_body(int max_tokens, bool stream) {
    return Json{{"model", "tinyllama"},
                {"messages", {{{"role", "user"}, {"content", "Hi"}}}},
                {"max_tokens", max_tokens},
                {"stream", stream}}
        .dump();
}

template <typename Done>
bool wait_for(Done done) {
    for (int i = 0; i < 500; ++i) {
        if (done()) return true;
        std::this_thread::sleep_for(milliseconds(10));
    }
    return false;
}

class HttpServerTest : public ::testing::Test {
protected:
    void start(ScriptedBackend::Next next, milliseconds delay = milliseconds(0)) {
        backend = std::make_unique<ScriptedBackend>(std::move(next), 64, 16, kContext, delay);
        loop = std::make_unique<ServingLoop>(*backend, word_decode, kEos);
        HttpOptions options{.cors_origins = {kDevOrigin},
                            .meta = {{"model", "scripted"}, {"max_batch", 8}},
                            .max_context = kContext};
        http = std::make_unique<HttpServer>(*loop, byte_encode, options);
        port = http->bind("127.0.0.1", 0);
        engine = std::jthread([this](std::stop_token stop) { loop->run(stop); });
        listener = std::jthread([this] { http->listen(); });
    }

    void TearDown() override {
        if (!http) return;
        http->stop();
        listener.join();
        engine.request_stop();
        engine.join();
    }

    httplib::Client client() const {
        httplib::Client c("127.0.0.1", port);
        c.set_read_timeout(10, 0);
        return c;
    }

    std::unique_ptr<ScriptedBackend> backend;
    std::unique_ptr<ServingLoop> loop;
    std::unique_ptr<HttpServer> http;
    std::jthread engine, listener;
    int port = 0;
};

int32_t always_11(const Request&) {
    return 11;
}

}  // namespace

TEST_F(HttpServerTest, HealthAnswersOk) {
    start(always_11);
    const auto res = client().Get("/health");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    EXPECT_EQ(Json::parse(res->body), Json({{"version", 1}, {"status", "ok"}}));
}

TEST_F(HttpServerTest, MetricsHaveTheContractFieldsAndNullLatencyAtFirst) {
    start(always_11);
    const auto res = client().Get("/metrics");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    const Json j = Json::parse(res->body);
    EXPECT_EQ(j["version"], 1);
    EXPECT_TRUE(j["time_s"].is_number());
    EXPECT_EQ(j["meta"]["model"], "scripted");
    EXPECT_EQ(j["requests"]["waiting"], 0);
    EXPECT_EQ(j["requests"]["finished_total"], 0);
    EXPECT_EQ(j["kv"]["blocks_total"], 64);
    EXPECT_EQ(j["kv"]["blocks_free"], 63);
    EXPECT_EQ(j["kv"]["block_size"], 16);
    EXPECT_EQ(j["tokens"]["output_total"], 0);
    EXPECT_TRUE(j["latency"]["ttft_s"].is_null());
    EXPECT_TRUE(j["latency"]["tpot_s"].is_null());
    EXPECT_EQ(j["steps_total"], 0);
}

TEST_F(HttpServerTest, PreflightFromTheDevServerAllowsThePrivateNetwork) {
    start(always_11);
    const auto res = client().Options("/v1/chat/completions",
                                      {{"Origin", kDevOrigin},
                                       {"Access-Control-Request-Method", "POST"},
                                       {"Access-Control-Request-Headers", "content-type"},
                                       {"Access-Control-Request-Private-Network", "true"}});
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 204);
    EXPECT_EQ(res->get_header_value("Access-Control-Allow-Origin"), kDevOrigin);
    EXPECT_EQ(res->get_header_value("Access-Control-Allow-Methods"), "GET, POST, OPTIONS");
    EXPECT_EQ(res->get_header_value("Access-Control-Allow-Headers"), "content-type");
    EXPECT_EQ(res->get_header_value("Access-Control-Allow-Private-Network"), "true");
}

TEST_F(HttpServerTest, PreflightFromAnotherOriginIsRefused) {
    start(always_11);
    const auto res = client().Options("/v1/chat/completions",
                                      {{"Origin", "https://evil.example"},
                                       {"Access-Control-Request-Method", "POST"},
                                       {"Access-Control-Request-Private-Network", "true"}});
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 403);
    EXPECT_FALSE(res->has_header("Access-Control-Allow-Origin"));
    EXPECT_FALSE(res->has_header("Access-Control-Allow-Private-Network"));
}

TEST_F(HttpServerTest, ResponsesCarryAnAllowedOrigin) {
    start(always_11);
    const auto allowed = client().Get("/metrics", httplib::Headers{{"Origin", kDevOrigin}});
    ASSERT_TRUE(allowed);
    EXPECT_EQ(allowed->get_header_value("Access-Control-Allow-Origin"), kDevOrigin);
}

TEST_F(HttpServerTest, RefusesRequestsFromAnotherOriginWithoutStartingWork) {
    start(always_11);
    const httplib::Headers evil = {{"Origin", "https://evil.example"}};
    const auto get = client().Get("/metrics", evil);
    ASSERT_TRUE(get);
    EXPECT_EQ(get->status, 403);
    EXPECT_FALSE(get->has_header("Access-Control-Allow-Origin"));
    const auto post =
        client().Post("/v1/chat/completions", evil, chat_body(4, false), "text/plain");
    ASSERT_TRUE(post);
    EXPECT_EQ(post->status, 403);
    EXPECT_EQ(Json::parse(post->body)["error"]["type"], "invalid_request_error");
    EXPECT_EQ(loop->metrics().counters.steps_total, 0);
}

TEST_F(HttpServerTest, AnswersAChatCompletionInOneObject) {
    start([](const Request& r) {
        return r.output.size() == 3 ? kEos : int32_t(20 + r.output.size());
    });
    const auto res =
        client().Post("/v1/chat/completions", chat_body(10, false), "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    const Json j = Json::parse(res->body);
    EXPECT_EQ(j["object"], "chat.completion");
    EXPECT_EQ(j["choices"][0]["message"]["content"], " w20 w21 w22");
    EXPECT_EQ(j["choices"][0]["finish_reason"], "stop");
    EXPECT_EQ(j["usage"]["completion_tokens"], 4);
    EXPECT_EQ(j["usage"]["prompt_tokens"], byte_encode("<|user|>\nHi</s>\n<|assistant|>\n").size());
    EXPECT_EQ(j["engine"]["seq_id"], 0);
}

TEST_F(HttpServerTest, StreamsAChatCompletionAsChunksThenDone) {
    start([](const Request& r) { return int32_t(20 + r.output.size()); });
    const auto res = client().Post("/v1/chat/completions", {{"Origin", kDevOrigin}},
                                   chat_body(3, true), "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    EXPECT_EQ(res->get_header_value("Content-Type"), "text/event-stream");
    EXPECT_EQ(res->get_header_value("Access-Control-Allow-Origin"), kDevOrigin);
    const auto lines = sse_data_lines(res->body);
    ASSERT_EQ(lines.size(), 6u);
    EXPECT_EQ(Json::parse(lines[0])["choices"][0]["delta"]["role"], "assistant");
    for (int i = 0; i < 3; ++i) {
        const Json chunk = Json::parse(lines[1 + i]);
        EXPECT_EQ(chunk["object"], "chat.completion.chunk");
        EXPECT_EQ(chunk["choices"][0]["delta"]["content"], " w" + std::to_string(20 + i));
        EXPECT_EQ(chunk["engine"]["step"], i);
    }
    const Json last = Json::parse(lines[4]);
    EXPECT_TRUE(last["choices"][0]["delta"].empty());
    EXPECT_EQ(last["choices"][0]["finish_reason"], "length");
    EXPECT_EQ(lines[5], "[DONE]");
}

TEST_F(HttpServerTest, RejectsABadRequestWith400) {
    start(always_11);
    const auto res =
        client().Post("/v1/chat/completions", R"({"messages":[]})", "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 400);
    EXPECT_EQ(Json::parse(res->body)["error"]["type"], "invalid_request_error");
    const auto too_long =
        client().Post("/v1/chat/completions", chat_body(kContext, false), "application/json");
    ASSERT_TRUE(too_long);
    EXPECT_EQ(too_long->status, 400);
}

TEST_F(HttpServerTest, StreamsEveryStepToASubscriber) {
    start(always_11, milliseconds(2));
    std::vector<Json> steps;
    std::jthread watcher([&] {
        std::string buffer;
        client().Get("/v1/steps", [&](const char* data, size_t size) {
            buffer.append(data, size);
            for (size_t end; (end = buffer.find("\n\n")) != std::string::npos;
                 buffer.erase(0, end + 2)) {
                const std::string event = buffer.substr(0, end);
                if (event.starts_with("event: step\ndata: "))
                    steps.push_back(Json::parse(event.substr(18)));
            }
            return steps.size() < 4;
        });
    });
    ASSERT_TRUE(wait_for([&] { return loop->steps().has_subscribers(); }));
    const auto res = client().Post("/v1/chat/completions", chat_body(6, false), "application/json");
    ASSERT_TRUE(res);
    watcher.join();
    ASSERT_EQ(steps.size(), 4u);
    for (int i = 0; i < 4; ++i) EXPECT_EQ(steps[i]["index"], i);
    EXPECT_EQ(steps[0]["kind"], "prefill");
    EXPECT_EQ(steps[1]["seqs"], Json::array({0}));
    EXPECT_EQ(steps[1]["tokens"][0]["text"], " w11");
    EXPECT_TRUE(wait_for([&] { return !loop->steps().has_subscribers(); }));
}

TEST_F(HttpServerTest, AClientThatDisconnectsMidStreamCancelsItsRequest) {
    start(always_11, milliseconds(3));
    int chunks = 0;
    client().Post("/v1/chat/completions", {}, chat_body(200, true), "application/json",
                  [&](const char*, size_t) { return ++chunks < 3; });
    ASSERT_TRUE(wait_for([&] {
        const Json m = Json::parse(client().Get("/metrics")->body);
        return m["requests"]["running"] == 0 && m["requests"]["waiting"] == 0;
    }));
    const Json m = Json::parse(client().Get("/metrics")->body);
    EXPECT_EQ(m["requests"]["finished_total"], 0);
    EXPECT_EQ(m["kv"]["blocks_free"], 63);
}
