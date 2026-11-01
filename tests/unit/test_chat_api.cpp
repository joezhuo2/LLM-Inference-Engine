#include <gtest/gtest.h>

#include <optional>
#include <string>
#include <vector>

#include "engine/server/chat_api.h"

using engine::server::ApiError;
using engine::server::ChatRequest;
using engine::server::Completion;
using engine::server::FinishReason;
using engine::server::parse_chat_request;
using engine::server::resolve_max_tokens;
using engine::server::SamplingDefaults;
using Json = nlohmann::json;

namespace {

const SamplingDefaults kDefaults{.temperature = 0.5f, .top_k = 40, .top_p = 0.9f, .max_tokens = 64};

ChatRequest parse(const std::string& body) {
    return parse_chat_request(body, kDefaults);
}

int status_of(const std::string& body) {
    try {
        parse(body);
    } catch (const ApiError& e) {
        return e.status();
    }
    return 200;
}

const Completion kCompletion{.seq = 341, .created = 1793000000, .model = "tinyllama"};

}  // namespace

TEST(ParseChatRequest, ReadsTheMessagesInOrder) {
    const auto r =
        parse(R"({"model":"x","messages":[{"role":"system","content":"Be brief."},)"
              R"({"role":"user","content":"Hi"},{"role":"assistant","content":"Hello"}]})");
    ASSERT_EQ(r.messages.size(), 3u);
    EXPECT_EQ(r.messages[0].role, "system");
    EXPECT_EQ(r.messages[0].content, "Be brief.");
    EXPECT_EQ(r.messages[2].role, "assistant");
    EXPECT_EQ(r.messages[2].content, "Hello");
}

TEST(ParseChatRequest, UsesTheServerDefaultsForMissingOrNullFields) {
    const auto r = parse(R"({"messages":[{"role":"user","content":"Hi"}],"temperature":null})");
    EXPECT_FLOAT_EQ(r.params.temperature, 0.5f);
    EXPECT_EQ(r.params.top_k, 40);
    EXPECT_FLOAT_EQ(r.params.top_p, 0.9f);
    EXPECT_FALSE(r.max_tokens.has_value());
    EXPECT_FALSE(r.seed.has_value());
    EXPECT_TRUE(r.stop.empty());
    EXPECT_FALSE(r.stream);
}

TEST(ParseChatRequest, ReadsTheSamplingFields) {
    const auto r =
        parse(R"({"messages":[{"role":"user","content":"Hi"}],"max_tokens":12,"temperature":0,)"
              R"("top_p":0.5,"top_k":3,"seed":7,"stream":true})");
    EXPECT_EQ(r.max_tokens, 12);
    EXPECT_FLOAT_EQ(r.params.temperature, 0.0f);
    EXPECT_FLOAT_EQ(r.params.top_p, 0.5f);
    EXPECT_EQ(r.params.top_k, 3);
    EXPECT_EQ(r.seed, 7u);
    EXPECT_TRUE(r.stream);
}

TEST(ParseChatRequest, TakesStopAsAStringOrAList) {
    const std::string messages = R"("messages":[{"role":"user","content":"Hi"}])";
    EXPECT_EQ(parse("{" + messages + R"(,"stop":"\n\n"})").stop,
              (std::vector<std::string>{"\n\n"}));
    EXPECT_EQ(parse("{" + messages + R"(,"stop":["a","","bc"]})").stop,
              (std::vector<std::string>{"a", "bc"}));
    EXPECT_TRUE(parse("{" + messages + R"(,"stop":""})").stop.empty());
    EXPECT_TRUE(parse("{" + messages + R"(,"stop":null})").stop.empty());
}

TEST(ParseChatRequest, RejectsMalformedBodiesWith400) {
    const std::string m = R"("messages":[{"role":"user","content":"Hi"}])";
    for (const std::string& body : {
             std::string("not json"),
             std::string("[1]"),
             std::string("{}"),
             std::string(R"({"messages":[]})"),
             std::string(R"({"messages":"Hi"})"),
             std::string(R"({"messages":[{"role":"tool","content":"x"}]})"),
             std::string(R"({"messages":[{"role":"user"}]})"),
             std::string(R"({"messages":[{"role":"user","content":[{"type":"text"}]}]})"),
             "{" + m + R"(,"max_tokens":0})",
             "{" + m + R"(,"max_tokens":1.5})",
             "{" + m + R"(,"temperature":-1})",
             "{" + m + R"(,"temperature":"hot"})",
             "{" + m + R"(,"top_p":0})",
             "{" + m + R"(,"top_p":1.5})",
             "{" + m + R"(,"top_k":-1})",
             "{" + m + R"(,"stop":3})",
             "{" + m + R"(,"stop":["a",3]})",
             "{" + m + R"(,"stream":"yes"})",
         })
        EXPECT_EQ(status_of(body), 400) << body;
}

TEST(ResolveMaxTokens, ClampsTheDefaultToTheContextLeft) {
    EXPECT_EQ(resolve_max_tokens(std::nullopt, 256, 100, 2048), 256);
    EXPECT_EQ(resolve_max_tokens(std::nullopt, 256, 2000, 2048), 49);
    EXPECT_EQ(resolve_max_tokens(std::nullopt, 256, 2048, 2048), 1);
}

TEST(ResolveMaxTokens, RejectsARequestThatDoesNotFit) {
    EXPECT_EQ(resolve_max_tokens(49, 256, 2000, 2048), 49);
    EXPECT_THROW(resolve_max_tokens(50, 256, 2000, 2048), ApiError);
    EXPECT_THROW(resolve_max_tokens(std::nullopt, 256, 2049, 2048), ApiError);
}

TEST(ChatChunks, MatchTheContractShape) {
    EXPECT_EQ(content_chunk(kCompletion, " the", 6118).dump(),
              R"({"id":"chatcmpl-341","object":"chat.completion.chunk","created":1793000000,)"
              R"("model":"tinyllama","choices":[{"index":0,"delta":{"content":" the"},)"
              R"("finish_reason":null}],"engine":{"seq_id":341,"step":6118}})");
}

TEST(ChatChunks, StartWithTheRoleAndEndWithAnEmptyDeltaAndAReason) {
    const auto role = role_chunk(kCompletion);
    EXPECT_EQ(role["choices"][0]["delta"], Json({{"role", "assistant"}}));
    EXPECT_TRUE(role["choices"][0]["finish_reason"].is_null());
    EXPECT_EQ(role["engine"]["seq_id"], 341);

    const auto stop = finish_chunk(kCompletion, FinishReason::Stop, 9);
    EXPECT_TRUE(stop["choices"][0]["delta"].is_object() && stop["choices"][0]["delta"].empty());
    EXPECT_EQ(stop["choices"][0]["finish_reason"], "stop");
    EXPECT_EQ(stop["engine"]["step"], 9);
    EXPECT_EQ(finish_chunk(kCompletion, FinishReason::Length, 9)["choices"][0]["finish_reason"],
              "length");
}

TEST(ChatCompletion, CarriesTheMessageAndUsage) {
    const auto j = completion_json(kCompletion, "Paris.", FinishReason::Length, 20, 30, 5);
    EXPECT_EQ(j["object"], "chat.completion");
    EXPECT_EQ(j["id"], "chatcmpl-341");
    EXPECT_EQ(j["choices"][0]["message"]["role"], "assistant");
    EXPECT_EQ(j["choices"][0]["message"]["content"], "Paris.");
    EXPECT_EQ(j["choices"][0]["finish_reason"], "length");
    EXPECT_EQ(j["usage"]["prompt_tokens"], 30);
    EXPECT_EQ(j["usage"]["completion_tokens"], 5);
    EXPECT_EQ(j["usage"]["total_tokens"], 35);
    EXPECT_EQ(j["engine"]["seq_id"], 341);
}

TEST(ErrorJson, FollowsTheOpenAiShape) {
    EXPECT_EQ(engine::server::error_json("bad").dump(),
              R"({"error":{"message":"bad","type":"invalid_request_error"}})");
}

TEST(Sse, FramesDataLinesAndEvents) {
    EXPECT_EQ(engine::server::sse_data("[DONE]"), "data: [DONE]\n\n");
    EXPECT_EQ(engine::server::sse_event("step", R"({"index":0})"),
              "event: step\ndata: {\"index\":0}\n\n");
}
