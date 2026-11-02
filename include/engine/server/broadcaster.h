#pragma once

#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "engine/server/channel.h"

namespace engine::server {

using Subscription = Channel<std::string>;

class StepBroadcaster {
public:
    explicit StepBroadcaster(size_t capacity = 256) : capacity_(capacity) {}

    std::shared_ptr<Subscription> subscribe();
    bool has_subscribers();
    void publish(const std::string& event);
    void close();

private:
    size_t capacity_;
    std::mutex mutex_;
    std::vector<std::weak_ptr<Subscription>> subscribers_;
    bool closed_ = false;

    void prune();
};

}  // namespace engine::server
