#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>

#include "engine/core/stream.h"
#include "engine/loader/weight_layout.h"
#include "engine/model/config.h"
#include "engine/model/weights.h"
#include "engine/runtime/device_buffer.h"

namespace engine {

inline constexpr size_t kUploadChunkBytes = size_t(64) << 20;

struct UploadStats {
    uint64_t bytes = 0;
    double seconds = 0.0;

    double gb_per_s() const { return seconds > 0.0 ? double(bytes) / seconds / 1e9 : 0.0; }
};

UploadStats upload_weights(std::span<const std::byte> file, const WeightLayout& layout,
                           void* device, Stream stream, size_t chunk_bytes = kUploadChunkBytes);

struct DeviceWeights {
    ModelConfig config;
    WeightLayout layout;
    DeviceBuffer buffer;
    ModelWeights weights;
    UploadStats stats;
};

DeviceWeights load_weights(const std::filesystem::path& model_dir, Stream stream);

}  // namespace engine
