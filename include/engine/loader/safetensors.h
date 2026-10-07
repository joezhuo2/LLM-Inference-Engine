#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <vector>

#include "engine/core/tensor.h"

namespace engine {

struct TensorEntry {
    DType dtype = DType::BF16;
    std::vector<int64_t> shape;
    uint64_t offset = 0;
    uint64_t bytes = 0;
};

struct SafetensorsHeader {
    uint64_t data_offset = 0;
    std::map<std::string, TensorEntry> tensors;
};

SafetensorsHeader parse_safetensors_header(std::span<const std::byte> file);

}  // namespace engine
