#pragma once

#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "engine/loader/mapped_file.h"
#include "engine/loader/safetensors.h"

namespace engine::test {

class SafetensorsFile {
public:
    explicit SafetensorsFile(const std::filesystem::path& path)
        : file_(path), header_(parse_safetensors_header(file_.bytes())) {}

    const TensorEntry& entry(const std::string& name) const { return header_.tensors.at(name); }

    template <typename T>
    std::vector<T> read(const std::string& name) const {
        const TensorEntry& e = entry(name);
        std::vector<T> out(e.bytes / sizeof(T));
        std::memcpy(out.data(), file_.bytes().data() + e.offset, e.bytes);
        return out;
    }

private:
    MappedFile file_;
    SafetensorsHeader header_;
};

}  // namespace engine::test
