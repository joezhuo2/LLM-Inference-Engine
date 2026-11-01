#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <utility>

namespace engine::server {

template <typename T>
class Channel {
public:
    explicit Channel(size_t capacity = 0) : capacity_(capacity) {}

    void push(T item) {
        {
            const std::lock_guard lock(mutex_);
            if (closed_) return;
            if (capacity_ > 0 && items_.size() == capacity_) {
                items_.pop_front();
                ++dropped_;
            }
            items_.push_back(std::move(item));
        }
        ready_.notify_one();
    }

    std::optional<T> pop(std::chrono::milliseconds wait) {
        std::unique_lock lock(mutex_);
        ready_.wait_for(lock, wait, [&] { return !items_.empty() || closed_; });
        if (items_.empty()) return std::nullopt;
        T item = std::move(items_.front());
        items_.pop_front();
        return item;
    }

    void close() {
        {
            const std::lock_guard lock(mutex_);
            closed_ = true;
        }
        ready_.notify_all();
    }

    bool closed() const {
        const std::lock_guard lock(mutex_);
        return closed_ && items_.empty();
    }

    uint64_t dropped() const {
        const std::lock_guard lock(mutex_);
        return dropped_;
    }

private:
    size_t capacity_;
    mutable std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<T> items_;
    bool closed_ = false;
    uint64_t dropped_ = 0;
};

}  // namespace engine::server
