#pragma once

#include <cstdlib>
#include <new>
#include <utility>

class Buffer {
    private:
        void* memory_address{nullptr};
        std::size_t requested_size{0};
    
    public: 
        Buffer() = default;
        
        explicit Buffer(std::size_t size) : requested_size(size) {
            if (requested_size > 0) {
                memory_address = std::malloc(size);
                if (memory_address == nullptr) {
                    throw std::bad_alloc();
                }
            }
        }

        ~Buffer() {
            std::free(memory_address);
        }

        Buffer(const Buffer&) = delete;
        Buffer& operator = (const Buffer&) = delete;

        Buffer(Buffer&& other) noexcept :
            memory_address(std::exchange(other.memory_address, nullptr)),
            requested_size(std::exchange(other.requested_size, 0)) {}

        Buffer& operator = (Buffer&& other) noexcept {
            if (this != &other) {
                std::free(memory_address);

                memory_address = std::exchange(other.memory_address, nullptr);
                requested_size = std::exchange(other.requested_size, 0);
            }
            return *this;
        }

        void* data() {
            return memory_address;
        }

        const void* data() const {
            return memory_address;
        }

        std::size_t size() const {
            return requested_size;
        }

        bool empty() const {
            return requested_size == 0 || memory_address == nullptr;
        }
};
