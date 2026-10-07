#pragma once

#include <unistd.h>

#include <atomic>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>

namespace engine::test {

class TempFile {
public:
    explicit TempFile(std::string_view contents) {
        static std::atomic<int> counter{0};
        path_ = std::filesystem::temp_directory_path() /
                ("engine_test_" + std::to_string(::getpid()) + "_" + std::to_string(counter++));
        std::ofstream(path_, std::ios::binary)
            .write(contents.data(), std::streamsize(contents.size()));
    }
    ~TempFile() { std::filesystem::remove(path_); }

    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;

    const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

}  // namespace engine::test
