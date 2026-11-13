#include "engine/loader/tensor_parallel.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace engine {

namespace {

constexpr uint64_t align_up(uint64_t byte_os, uint64_t alignment) {
    return (byte_os + alignment - 1) & ~(alignment - 1);
}

const TensorEntry& require_bf16_tensor(const SafetensorsHeader& header, const std::string& name,
                                       const std::vector<int64_t>& shape,
                                       std::unordered_set<std::string>& seen) {
    seen.insert(name);

    auto it = header.tensors.find(name);

    if (it == header.tensors.end()) {
        throw std::runtime_error("missing required tensor " + name);
    }

    const TensorEntry& info = it->second;

    if (info.dtype != DType::BF16) {
        throw std::runtime_error("tensor " + name + " must have dtype BF16, got " +
                                 std::string(dtype_name(info.dtype)));
    }

    if (info.shape != shape) {
        throw std::runtime_error("unexpected shape for tensor: " + name);
    }

    uint64_t bytes = dtype_size(DType::BF16);

    for (int64_t dim_size : shape) {
        bytes *= static_cast<uint64_t>(dim_size);
    }

    if (info.bytes != bytes) {
        throw std::runtime_error("byte size mismatch for tensor: " + name);
    }

    return info;
}

Region allocate_region(uint64_t& buffer_cursor, const std::vector<int64_t>& region_shape) {
    buffer_cursor = align_up(buffer_cursor, kWeightAlignment);
    uint64_t region_bytes = dtype_size(DType::BF16);

    for (int64_t dim_size : region_shape) {
        region_bytes *= static_cast<uint64_t>(dim_size);
    }

    Region region{buffer_cursor, region_bytes, region_shape};
    buffer_cursor += region_bytes;

    return region;
}

}  // namespace

WeightLayout plan_shard_layout(const ModelConfig&, const SafetensorsHeader&, int, int) {
    throw std::logic_error("unimplemented: plan_shard_layout");
}

}  // namespace engine
