#include <gtest/gtest.h>

#include <chrono>
#include <string>
#include <thread>

#include "engine/server/broadcaster.h"
#include "engine/server/channel.h"

using engine::server::Channel;
using engine::server::StepBroadcaster;
using std::chrono::milliseconds;

TEST(Channel, PopsInOrderAndTimesOutWhenEmpty) {
    Channel<int> channel;
    channel.push(1);
    channel.push(2);
    EXPECT_EQ(channel.pop(milliseconds(0)), 1);
    EXPECT_EQ(channel.pop(milliseconds(0)), 2);
    EXPECT_EQ(channel.pop(milliseconds(10)), std::nullopt);
    EXPECT_FALSE(channel.closed());
}

TEST(Channel, DropsTheOldestItemWhenFull) {
    Channel<int> channel(2);
    for (int i = 0; i < 5; ++i) channel.push(i);
    EXPECT_EQ(channel.dropped(), 3u);
    EXPECT_EQ(channel.pop(milliseconds(0)), 3);
    EXPECT_EQ(channel.pop(milliseconds(0)), 4);
}

TEST(Channel, DrainsItsItemsAfterCloseThenReportsClosed) {
    Channel<int> channel;
    channel.push(1);
    channel.close();
    channel.push(2);
    EXPECT_FALSE(channel.closed());
    EXPECT_EQ(channel.pop(milliseconds(0)), 1);
    EXPECT_EQ(channel.pop(milliseconds(0)), std::nullopt);
    EXPECT_TRUE(channel.closed());
}

TEST(Channel, WakesAWaitingReader) {
    Channel<int> channel;
    std::thread writer([&] {
        std::this_thread::sleep_for(milliseconds(20));
        channel.push(7);
    });
    EXPECT_EQ(channel.pop(std::chrono::seconds(5)), 7);
    writer.join();
}

TEST(StepBroadcaster, DeliversEveryStepToEverySubscriber) {
    StepBroadcaster steps;
    const auto a = steps.subscribe();
    const auto b = steps.subscribe();
    steps.publish("s0");
    steps.publish("s1");
    for (const auto& s : {a, b}) {
        EXPECT_EQ(s->pop(milliseconds(0)), "s0");
        EXPECT_EQ(s->pop(milliseconds(0)), "s1");
    }
}

TEST(StepBroadcaster, ASlowSubscriberDropsOldStepsWithoutBlockingThePublisher) {
    StepBroadcaster steps(4);
    const auto slow = steps.subscribe();
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < 1000; ++i) steps.publish("s" + std::to_string(i));
    EXPECT_LT(std::chrono::steady_clock::now() - start, std::chrono::seconds(1));
    EXPECT_EQ(slow->dropped(), 996u);
    EXPECT_EQ(slow->pop(milliseconds(0)), "s996");
    steps.publish("s1000");
    EXPECT_EQ(slow->pop(milliseconds(0)), "s997");
}

TEST(StepBroadcaster, ForgetsSubscribersThatWentAway) {
    StepBroadcaster steps;
    EXPECT_FALSE(steps.has_subscribers());
    auto s = steps.subscribe();
    EXPECT_TRUE(steps.has_subscribers());
    s.reset();
    EXPECT_FALSE(steps.has_subscribers());
    steps.publish("s0");
}

TEST(StepBroadcaster, CloseEndsEverySubscription) {
    StepBroadcaster steps;
    const auto before = steps.subscribe();
    steps.close();
    const auto after = steps.subscribe();
    EXPECT_TRUE(before->closed());
    EXPECT_TRUE(after->closed());
    EXPECT_FALSE(steps.has_subscribers());
}
