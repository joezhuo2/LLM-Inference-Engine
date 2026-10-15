#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <vector>

#include "engine/core/tensor.h"
#include "engine/runtime/device_buffer.h"
#include "support/device.h"

namespace engine::test {

class GuardedBf16 {
public:
    static constexpr size_t kGuard = 64;
    static constexpr uint16_t kSentinel = 0xA5A5;

    GuardedBf16(const std::vector<uint16_t>& values, size_t shift, bool fill_sentinel)
        : n_(values.size()), begin_(kGuard + shift) {
        std::vector<uint16_t> host(begin_ + n_ + kGuard, kSentinel);
        if (!fill_sentinel)
            for (size_t i = 0; i < n_; ++i) host[begin_ + i] = values[i];
        buffer_ = upload(host);
    }

    void* data() { return static_cast<uint16_t*>(buffer_.data()) + begin_; }

    Tensor tensor(std::initializer_list<int64_t> shape) {
        return Tensor(data(), DType::BF16, shape);
    }

    std::vector<uint16_t> values() const {
        const auto all = download<uint16_t>(buffer_);
        return {all.begin() + std::ptrdiff_t(begin_), all.begin() + std::ptrdiff_t(begin_ + n_)};
    }

    bool guards_intact() const {
        const auto all = download<uint16_t>(buffer_);
        for (size_t i = 0; i < begin_; ++i)
            if (all[i] != kSentinel) return false;
        for (size_t i = begin_ + n_; i < all.size(); ++i)
            if (all[i] != kSentinel) return false;
        return true;
    }

private:
    size_t n_, begin_;
    DeviceBuffer buffer_;
};

}  // namespace engine::test
