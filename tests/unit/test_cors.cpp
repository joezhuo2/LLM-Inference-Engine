#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

#include "engine/server/cors.h"

using engine::server::CorsPolicy;
using engine::server::HeaderList;

namespace {

const CorsPolicy kPolicy({"http://localhost:3000", "https://example.vercel.app"});

std::string header(const HeaderList& headers, const std::string& name) {
    for (const auto& [key, value] : headers)
        if (key == name) return value;
    return "<missing>";
}

}  // namespace

TEST(CorsPolicy, AllowsOnlyTheListedOriginsExactly) {
    EXPECT_TRUE(kPolicy.allows("http://localhost:3000"));
    EXPECT_TRUE(kPolicy.allows("https://example.vercel.app"));
    EXPECT_FALSE(kPolicy.allows("http://localhost:3001"));
    EXPECT_FALSE(kPolicy.allows("https://localhost:3000"));
    EXPECT_FALSE(kPolicy.allows("https://example.vercel.app.evil.com"));
    EXPECT_FALSE(kPolicy.allows("null"));
    EXPECT_FALSE(kPolicy.allows(""));
}

TEST(CorsPolicy, EchoesAnAllowedOriginOnResponses) {
    const auto headers = kPolicy.response_headers("http://localhost:3000");
    EXPECT_EQ(header(headers, "Access-Control-Allow-Origin"), "http://localhost:3000");
    EXPECT_EQ(header(headers, "Vary"), "Origin");
}

TEST(CorsPolicy, SendsNoAllowOriginForOtherOrigins) {
    const auto headers = kPolicy.response_headers("https://evil.example");
    EXPECT_EQ(header(headers, "Access-Control-Allow-Origin"), "<missing>");
    EXPECT_EQ(header(headers, "Vary"), "Origin");
}

TEST(CorsPolicy, PreflightAllowsTheMethodsHeadersAndPrivateNetwork) {
    const auto headers =
        kPolicy.preflight_headers("http://localhost:3000", "content-type, x-custom", true);
    EXPECT_EQ(header(headers, "Access-Control-Allow-Origin"), "http://localhost:3000");
    EXPECT_EQ(header(headers, "Access-Control-Allow-Methods"), "GET, POST, OPTIONS");
    EXPECT_EQ(header(headers, "Access-Control-Allow-Headers"), "content-type, x-custom");
    EXPECT_EQ(header(headers, "Access-Control-Allow-Private-Network"), "true");
    EXPECT_EQ(header(headers, "Access-Control-Max-Age"), "600");
}

TEST(CorsPolicy, PreflightOmitsPrivateNetworkUnlessAskedAndDefaultsTheHeaders) {
    const auto headers = kPolicy.preflight_headers("https://example.vercel.app", "", false);
    EXPECT_EQ(header(headers, "Access-Control-Allow-Private-Network"), "<missing>");
    EXPECT_EQ(header(headers, "Access-Control-Allow-Headers"), "Content-Type");
}

TEST(CorsPolicy, PreflightFromAnotherOriginAllowsNothing) {
    const auto headers = kPolicy.preflight_headers("https://evil.example", "content-type", true);
    EXPECT_EQ(header(headers, "Access-Control-Allow-Origin"), "<missing>");
    EXPECT_EQ(header(headers, "Access-Control-Allow-Private-Network"), "<missing>");
    EXPECT_EQ(header(headers, "Access-Control-Allow-Methods"), "<missing>");
}

TEST(CorsPolicy, DefaultsToTheLocalDevServer) {
    const CorsPolicy policy(engine::server::kDefaultCorsOrigins);
    EXPECT_TRUE(policy.allows("http://localhost:3000"));
}

TEST(CorsPolicy, RejectsOriginsThatAreNotSchemeAndHost) {
    for (const std::string origin : {"*", "localhost:3000", "http://localhost:3000/",
                                     "https://x.app/path", "ftp://x.app", "http://", ""})
        EXPECT_THROW(CorsPolicy({origin}), std::invalid_argument) << origin;
}
