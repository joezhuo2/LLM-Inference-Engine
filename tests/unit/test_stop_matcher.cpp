#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "engine/server/stop_matcher.h"

using engine::server::StopMatcher;

namespace {

std::string run(StopMatcher& m, const std::vector<std::string>& pieces, bool* stopped) {
    std::string out;
    *stopped = false;
    for (const std::string& piece : pieces) {
        const auto r = m.push(piece);
        out += r.text;
        if (r.stopped) {
            *stopped = true;
            return out;
        }
    }
    return out + m.flush();
}

}  // namespace

TEST(StopMatcher, PassesTextThroughWithNoStops) {
    StopMatcher m({});
    EXPECT_EQ(m.push(" Hello").text, " Hello");
    EXPECT_EQ(m.push("\n\n").text, "\n\n");
    EXPECT_EQ(m.flush(), "");
}

TEST(StopMatcher, CutsTheTextBeforeTheStopString) {
    StopMatcher m({"END"});
    bool stopped = false;
    EXPECT_EQ(run(m, {"one", " two END three"}, &stopped), "one two ");
    EXPECT_TRUE(stopped);
}

TEST(StopMatcher, FindsAStopStringSplitAcrossTokens) {
    StopMatcher m({"\n\n"});
    EXPECT_EQ(m.push("Paris.\n").text, "Paris.");
    const auto r = m.push("\nNext");
    EXPECT_TRUE(r.stopped);
    EXPECT_EQ(r.text, "");
}

TEST(StopMatcher, ReleasesHeldTextWhenTheStopDoesNotFollow) {
    StopMatcher m({"</answer>"});
    EXPECT_EQ(m.push("a </an").text, "a ");
    EXPECT_EQ(m.push("d b").text, "</and b");
}

TEST(StopMatcher, FlushReturnsTheHeldTail) {
    StopMatcher m({"stop"});
    bool stopped = false;
    EXPECT_EQ(run(m, {"non", "st"}, &stopped), "nonst");
    EXPECT_FALSE(stopped);
}

TEST(StopMatcher, TheEarliestOfSeveralStopsWins) {
    StopMatcher m({"world", "lo"});
    bool stopped = false;
    EXPECT_EQ(run(m, {"hello world"}, &stopped), "hel");
    EXPECT_TRUE(stopped);
}

TEST(StopMatcher, HoldsTheLongestPrefixOfAnyStop) {
    StopMatcher m({"abc", "bcd"});
    EXPECT_EQ(m.push("xab").text, "x");
    EXPECT_EQ(m.push("cz").text, "");
}
