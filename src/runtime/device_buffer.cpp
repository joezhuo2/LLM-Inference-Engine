#include "engine/runtime/device_buffer.h"

#include <utility>

#include "kernels/cuda_check.h"

namespace engine {

DeviceBuffer::DeviceBuffer(size_t bytes) : size_(bytes) {
    if (bytes == 0) return;
    CUDA_CHECK(cudaMalloc(&ptr_, bytes));
#ifndef NDEBUG
    CUDA_CHECK(cudaMemset(ptr_, 0xFF, bytes));
    CUDA_CHECK(cudaDeviceSynchronize());
#endif
}

DeviceBuffer::~DeviceBuffer() {
    CUDA_CHECK(cudaFree(ptr_));
}

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
    : ptr_(std::exchange(other.ptr_, nullptr)), size_(std::exchange(other.size_, 0)) {}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
        CUDA_CHECK(cudaFree(ptr_));
        ptr_ = std::exchange(other.ptr_, nullptr);
        size_ = std::exchange(other.size_, 0);
    }
    return *this;
}

}  // namespace engine
