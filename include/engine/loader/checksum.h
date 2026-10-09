#pragma once

#include <cstddef>
#include <cstdint>
#include <nlohmann/json_fwd.hpp>
#include <span>
#include <string>
#include <vector>

#include "engine/core/tensor.h"
#include "engine/loader/safetensors.h"

namespace engine {

struct TensorChecksum {
    std::string name;
    DType dtype = DType::BF16;
    std::vector<int64_t> shape;
    double sum = 0.0;
    double abs_sum = 0.0;
};

TensorChecksum checksum_bf16(std::string name, std::vector<int64_t> shape,
                             std::span<const std::byte> data);
std::vector<TensorChecksum> checksum_file(std::span<const std::byte> file,
                                          const SafetensorsHeader& header);

nlohmann::json checksums_to_json(const std::vector<TensorChecksum>& checksums);
std::vector<TensorChecksum> checksums_from_json(const nlohmann::json& json);

}  // namespace engine
