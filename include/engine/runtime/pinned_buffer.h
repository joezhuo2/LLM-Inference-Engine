#pragma once

#include <cstddef>

namespace engine {

class PinnedBuffer {
public:
    PinnedBuffer() = default;
    explicit PinnedBuffer(size_t bytes);
    ~PinnedBuffer();

    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    PinnedBuffer(PinnedBuffer&& other) noexcept;
    PinnedBuffer& operator=(PinnedBuffer&& other) noexcept;

    void* data() { return ptr_; }
    const void* data() const { return ptr_; }
    size_t size() const { return size_; }

private:
    void* ptr_ = nullptr;
    size_t size_ = 0;
};

}  // namespace engine
