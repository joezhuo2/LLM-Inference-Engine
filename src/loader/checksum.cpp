#include "engine/loader/checksum.h"

#include <bit>
#include <cmath>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <utility>

namespace engine {

namespace {

DType dtype_from_name(const std::string& name) {
    for (DType d : {DType::BF16, DType::F16, DType::F32, DType::I32, DType::I64})
        if (dtype_name(d) == name) return d;
    throw std::runtime_error("checksum: unknown dtype " + name);
}

}  // namespace

TensorChecksum checksum_bf16(std::string name, std::vector<int64_t> shape,
                             std::span<const std::byte> data) {
    uint64_t numel = 1;
    for (int64_t d : shape) numel *= uint64_t(d);
    if (data.size() != numel * dtype_size(DType::BF16))
        throw std::invalid_argument("checksum: " + name + " has the wrong byte size");

    TensorChecksum c{std::move(name), DType::BF16, std::move(shape)};
    for (size_t i = 0; i < data.size(); i += 2) {
        const uint32_t bits = uint32_t(data[i]) | uint32_t(data[i + 1]) << 8;
        const double v = std::bit_cast<float>(bits << 16);
        c.sum += v;
        c.abs_sum += std::fabs(v);
    }
    return c;
}

std::vector<TensorChecksum> checksum_file(std::span<const std::byte> file,
                                          const SafetensorsHeader& header) {
    std::vector<TensorChecksum> out;
    for (const auto& [name, t] : header.tensors) {
        if (t.dtype != DType::BF16) throw std::runtime_error("checksum: " + name + " is not BF16");
        out.push_back(checksum_bf16(name, t.shape, file.subspan(t.offset, t.bytes)));
    }
    return out;
}

nlohmann::json checksums_to_json(const std::vector<TensorChecksum>& checksums) {
    nlohmann::json tensors = nlohmann::json::object();
    for (const TensorChecksum& c : checksums) {
        tensors[c.name] = {{"dtype", std::string(dtype_name(c.dtype))},
                           {"shape", c.shape},
                           {"sum", c.sum},
                           {"abs_sum", c.abs_sum}};
    }
    return {{"tensors", tensors}};
}

std::vector<TensorChecksum> checksums_from_json(const nlohmann::json& json) {
    std::vector<TensorChecksum> out;
    for (const auto& [name, t] : json.at("tensors").items()) {
        out.push_back({name, dtype_from_name(t.at("dtype").get<std::string>()),
                       t.at("shape").get<std::vector<int64_t>>(), t.at("sum").get<double>(),
                       t.at("abs_sum").get<double>()});
    }
    return out;
}

}  // namespace engine
