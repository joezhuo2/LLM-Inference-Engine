#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace engine::test {

inline std::string safetensors_file(uint64_t header_length, std::string_view header,
                                    std::string_view data) {
    std::string out(8, '\0');
    for (int i = 0; i < 8; ++i) out[i] = char((header_length >> (8 * i)) & 0xFF);
    out += header;
    out += data;
    return out;
}

inline std::string safetensors_file(std::string_view header, std::string_view data = {}) {
    return safetensors_file(header.size(), header, data);
}

inline std::span<const std::byte> as_bytes(const std::string& s) {
    return std::as_bytes(std::span(s.data(), s.size()));
}

}  // namespace engine::test
