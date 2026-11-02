#include "engine/server/broadcaster.h"

#include <algorithm>

namespace engine::server {

std::shared_ptr<Subscription> StepBroadcaster::subscribe() {
    auto subscription = std::make_shared<Subscription>(capacity_);
    const std::lock_guard lock(mutex_);
    if (closed_)
        subscription->close();
    else
        subscribers_.push_back(subscription);
    return subscription;
}

bool StepBroadcaster::has_subscribers() {
    const std::lock_guard lock(mutex_);
    prune();
    return !subscribers_.empty();
}

void StepBroadcaster::publish(const std::string& event) {
    const std::lock_guard lock(mutex_);
    prune();
    for (const auto& weak : subscribers_)
        if (const auto subscription = weak.lock()) subscription->push(event);
}

void StepBroadcaster::close() {
    const std::lock_guard lock(mutex_);
    closed_ = true;
    for (const auto& weak : subscribers_)
        if (const auto subscription = weak.lock()) subscription->close();
    subscribers_.clear();
}

void StepBroadcaster::prune() {
    std::erase_if(subscribers_, [](const auto& weak) { return weak.expired(); });
}

}  // namespace engine::server
