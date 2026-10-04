#include "engine/runtime/pinned_buffer.h"

#include <utility>

#include "kernels/cuda_check.h"

namespace engine {

PinnedBuffer::PinnedBuffer(size_t bytes) : size_(bytes) {
    if (bytes > 0) CUDA_CHECK(cudaMallocHost(&ptr_, bytes));
}

PinnedBuffer::~PinnedBuffer() {
    CUDA_CHECK(cudaFreeHost(ptr_));
}

PinnedBuffer::PinnedBuffer(PinnedBuffer&& other) noexcept
    : ptr_(std::exchange(other.ptr_, nullptr)), size_(std::exchange(other.size_, 0)) {}

PinnedBuffer& PinnedBuffer::operator=(PinnedBuffer&& other) noexcept {
    if (this != &other) {
        CUDA_CHECK(cudaFreeHost(ptr_));
        ptr_ = std::exchange(other.ptr_, nullptr);
        size_ = std::exchange(other.size_, 0);
    }
    return *this;
}

}  // namespace engine
