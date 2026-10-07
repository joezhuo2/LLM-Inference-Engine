#include "engine/loader/mapped_file.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <string>
#include <system_error>
#include <utility>

namespace engine {

namespace {

[[noreturn]] void fail(const char* op, const std::filesystem::path& path) {
    throw std::system_error(errno, std::generic_category(), std::string(op) + " " + path.string());
}

struct Fd {
    int fd;
    ~Fd() { ::close(fd); }
};

}  // namespace

MappedFile::MappedFile(const std::filesystem::path& path) {
    const Fd file{::open(path.c_str(), O_RDONLY | O_CLOEXEC)};
    if (file.fd == -1) fail("open", path);
    struct stat st;
    if (::fstat(file.fd, &st) == -1) fail("fstat", path);
    size_ = size_t(st.st_size);
    if (size_ == 0) return;
    void* p = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, file.fd, 0);
    if (p == MAP_FAILED) fail("mmap", path);
    data_ = static_cast<const std::byte*>(p);
}

MappedFile::~MappedFile() {
    if (data_) ::munmap(const_cast<std::byte*>(data_), size_);
}

MappedFile::MappedFile(MappedFile&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)) {}

MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
    if (this != &other) {
        if (data_) ::munmap(const_cast<std::byte*>(data_), size_);
        data_ = std::exchange(other.data_, nullptr);
        size_ = std::exchange(other.size_, 0);
    }
    return *this;
}

}  // namespace engine
