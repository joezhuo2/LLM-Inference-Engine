#include <gtest/gtest.h>

#include "engine/server/host_check.h"

using engine::server::host_allowed;
using engine::server::host_name;
using engine::server::kLoopbackHosts;

TEST(HostCheck, TakesTheNameOffThePort) {
    EXPECT_EQ(host_name("localhost:8000"), "localhost");
    EXPECT_EQ(host_name("localhost"), "localhost");
    EXPECT_EQ(host_name("127.0.0.1:8000"), "127.0.0.1");
    EXPECT_EQ(host_name("[::1]:8000"), "[::1]");
    EXPECT_EQ(host_name("[::1]"), "[::1]");
    EXPECT_EQ(host_name("LOCALHOST:8000"), "localhost");
}

TEST(HostCheck, MalformedHeadersHaveNoName) {
    EXPECT_EQ(host_name(""), "");
    EXPECT_EQ(host_name(":8000"), "");
    EXPECT_EQ(host_name("localhost:"), "");
    EXPECT_EQ(host_name("localhost:80a"), "");
    EXPECT_EQ(host_name("localhost:80:80"), "");
    EXPECT_EQ(host_name("[::1"), "");
    EXPECT_EQ(host_name("[::1]x"), "");
    EXPECT_EQ(host_name("[::1]:"), "");
}

TEST(HostCheck, AllowsLoopbackNamesOnAnyPort) {
    EXPECT_TRUE(host_allowed("localhost:8000", kLoopbackHosts));
    EXPECT_TRUE(host_allowed("127.0.0.1:1234", kLoopbackHosts));
    EXPECT_TRUE(host_allowed("[::1]:8000", kLoopbackHosts));
}

TEST(HostCheck, RefusesARebindingAttackersName) {
    EXPECT_FALSE(host_allowed("evil.example:8000", kLoopbackHosts));
    EXPECT_FALSE(host_allowed("localhost.evil.example:8000", kLoopbackHosts));
    EXPECT_FALSE(host_allowed("127.0.0.1.nip.io:8000", kLoopbackHosts));
    EXPECT_FALSE(host_allowed("", kLoopbackHosts));
}

TEST(HostCheck, AllowsANameAddedToTheList) {
    const std::vector<std::string> allowed = {"localhost", "gpu-box"};
    EXPECT_TRUE(host_allowed("gpu-box:8000", allowed));
    EXPECT_FALSE(host_allowed("127.0.0.1:8000", allowed));
}
