#pragma once

#include <cstddef>
#include <filesystem>
#include <span>

namespace engine {

class MappedFile {
public:
    explicit MappedFile(const std::filesystem::path& path);
    ~MappedFile();

    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    MappedFile(MappedFile&& other) noexcept;
    MappedFile& operator=(MappedFile&& other) noexcept;

    std::span<const std::byte> bytes() const { return {data_, size_}; }

private:
    const std::byte* data_ = nullptr;
    size_t size_ = 0;
};

}  // namespace engine
