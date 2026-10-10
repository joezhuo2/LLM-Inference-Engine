#pragma once

#include <cstddef>
#include <cstdint>

#include "engine/model/config.h"

namespace engine {

constexpr int kKvBlockSize = 16;
constexpr double kKvCacheFraction = 0.9;

int64_t kv_block_bytes(const ModelConfig& config, int block_size);
int kv_cache_blocks(const ModelConfig& config, int block_size, size_t free_bytes,
                    double fraction = kKvCacheFraction);

}  // namespace engine
