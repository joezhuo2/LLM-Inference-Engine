#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "engine/sched/request.h"

using engine::Request;
using engine::RequestState;
using engine::validate;

TEST(Request, StartsWaitingWithNoOutput) {
    const Request r{.id = 3, .prompt = {1, 2}};
    EXPECT_EQ(r.state, RequestState::Waiting);
    EXPECT_EQ(r.max_new_tokens, 1);
    EXPECT_TRUE(r.output.empty());
    EXPECT_EQ(r.preemptions, 0);
    EXPECT_EQ(r.params.temperature, 0.0f);
}

TEST(Request, TokensRunThroughThePromptThenTheOutput) {
    Request r{.id = 1, .prompt = {10, 11, 12}, .max_new_tokens = 4};
    EXPECT_EQ(r.length(), 3);
    r.output = {20, 21};
    EXPECT_EQ(r.length(), 5);
    std::vector<int32_t> tokens;
    for (int64_t pos = 0; pos < r.length(); ++pos) tokens.push_back(r.token(pos));
    EXPECT_EQ(tokens, (std::vector<int32_t>{10, 11, 12, 20, 21}));
}

TEST(Request, AcceptsRequestsThatFitTheContext) {
    EXPECT_NO_THROW(validate(Request{.prompt = {1}}, 1));
    EXPECT_NO_THROW(validate(Request{.prompt = {1, 2, 3}, .max_new_tokens = 6}, 8));
    EXPECT_NO_THROW(validate(Request{.prompt = std::vector<int32_t>(2048, 5)}, 2048));
    EXPECT_NO_THROW(validate(Request{.prompt = {1}, .max_new_tokens = 2048}, 2048));
}

TEST(Request, RejectsRequestsPastTheContext) {
    EXPECT_THROW(validate(Request{.prompt = {1, 2, 3}, .max_new_tokens = 7}, 8),
                 std::invalid_argument);
    EXPECT_THROW(validate(Request{.prompt = std::vector<int32_t>(2049, 5)}, 2048),
                 std::invalid_argument);
    EXPECT_THROW(validate(Request{.prompt = {1}, .max_new_tokens = 2049}, 2048),
                 std::invalid_argument);
}

TEST(Request, RejectsMalformedRequests) {
    EXPECT_THROW(validate(Request{.id = 4}, 16), std::invalid_argument);
    EXPECT_THROW(validate(Request{.prompt = {1}, .max_new_tokens = 0}, 16), std::invalid_argument);
    EXPECT_THROW(validate(Request{.prompt = {1}, .max_new_tokens = -1}, 16), std::invalid_argument);
    EXPECT_THROW(validate(Request{.prompt = {1}, .max_new_tokens = 2, .output = {7}}, 16),
                 std::invalid_argument);
    EXPECT_THROW(validate(Request{.prompt = {1}, .params = {-1.0f}}, 16), std::invalid_argument);
    EXPECT_THROW(validate(Request{.prompt = {1}, .params = {1.0f, 0, 0, 0.0f}}, 16),
                 std::invalid_argument);
}

TEST(Request, NamesTheRequestInErrors) {
    try {
        validate(Request{.id = 42, .prompt = {1}, .max_new_tokens = 0}, 16);
        FAIL() << "expected std::invalid_argument";
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find("request 42"), std::string::npos) << e.what();
    }
}
