#pragma once

#include <cstddef>

namespace engine {

class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t bytes);
    ~DeviceBuffer();

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;

    void* data() { return ptr_; }
    const void* data() const { return ptr_; }
    size_t size() const { return size_; }

private:
    void* ptr_ = nullptr;
    size_t size_ = 0;
};

}  // namespace engine
