#pragma once

#include <cerrno>
#include <string>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

class SafetensorReader {
    public:
        static inline std::string read_json_header(const std::string& file_path) {
            int fd = ::open(file_path.c_str(), O_RDONLY);
            if (fd == -1) {
                throw std::runtime_error("Failed to open file at " + file_path + " (" + std::strerror(errno) + ")");
            }

            struct FDCloser {
                int file_descriptor;
                ~FDCloser() {
                    if (file_descriptor != -1) {
                        ::close(file_descriptor);
                    }
                }
            } 
            
            fdCloser{fd};

            struct stat sb;
            if (::fstat(fd, &sb) == -1) {
                throw std::runtime_error("Failed to get file size for: " + file_path);
            }

            size_t file_size = static_cast<size_t>(sb.st_size);

            if (file_size < sizeof(uint64_t)) {
                throw std::runtime_error("Invalid safetensors file: File size is smaller than 8 bytes.");
            }

            void* mapped_ptr = ::mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0);
            if (mapped_ptr == MAP_FAILED) {
                throw std::runtime_error("Failed to mmap file: " + file_path + " (" + std::strerror(errno) + ")");
            }

            struct MemoryUnmapper {
                void* ptr;
                size_t size;
                ~MemoryUnmapper() {
                    if (ptr != MAP_FAILED && ptr != nullptr) {
                        ::munmap(ptr, size);
                    }
                }
            }

            unmapper{mapped_ptr, file_size};

            const uint8_t* byte_buffer = static_cast<const uint8_t*>(mapped_ptr);
            uint64_t header_size = 0;

            std::memcpy(&header_size, byte_buffer, sizeof(uint64_t));

            if (header_size > file_size - sizeof(uint64_t)) {
                throw std::runtime_error("Corrupted safetensors file: header size exceeds actual file length.");
            }

            const char* jsonStart = reinterpret_cast<const char*>(byte_buffer + sizeof(uint64_t));
            return std::string(jsonStart, static_cast<size_t>(header_size));
        }
};
