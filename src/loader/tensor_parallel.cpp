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

void push_whole_copy(std::vector<CopyOp>& copy_ops, uint64_t data_os, const std::string& name,
                     const TensorEntry& info, const Region& dst_region) {
    copy_ops.push_back(CopyOp{.name = name,
                              .src = data_os + info.offset,
                              .dst = dst_region.offset,
                              .bytes = info.bytes,
                              .shape = info.shape});
}

void push_row_slice_copy(std::vector<CopyOp>& copy_ops, uint64_t data_os, const std::string& name,
                         const TensorEntry& info, uint64_t dst_os, int64_t start, int64_t count) {
    constexpr uint64_t bf16_elem_bytes = dtype_size(DType::BF16);
    const int64_t cols = info.shape[1];
    const uint64_t row_bytes = static_cast<uint64_t>(cols) * bf16_elem_bytes;

    copy_ops.push_back(
        CopyOp{.name = name,
               .src = data_os + info.offset + static_cast<uint64_t>(start) * row_bytes,
               .dst = dst_os,
               .bytes = static_cast<uint64_t>(count) * row_bytes,
               .shape = {count, cols}});
}

}  // namespace

WeightLayout plan_shard_layout(const ModelConfig&, const SafetensorsHeader&, int, int) {
    throw std::logic_error("unimplemented: plan_shard_layout");
}

}  // namespace engine
