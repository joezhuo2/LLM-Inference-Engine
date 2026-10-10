#include "engine/kv/cache_size.h"

#include <algorithm>
#include <climits>
#include <stdexcept>
#include <string>

namespace engine {

int64_t kv_block_bytes(const ModelConfig& config, int block_size) {
    if (block_size < 1) throw std::invalid_argument("KV cache: block_size must be positive");
    return 2 * int64_t(config.layers) * config.kv_dim() * block_size * 2;
}

int kv_cache_blocks(const ModelConfig& config, int block_size, size_t free_bytes, double fraction) {
    if (!(fraction > 0.0 && fraction <= 1.0))
        throw std::invalid_argument("KV cache: fraction must be in (0, 1]");
    const int64_t block = kv_block_bytes(config, block_size);
    const auto budget = int64_t(double(free_bytes) * fraction);
    const int64_t blocks = budget / block;
    if (blocks < 2)
        throw std::runtime_error("KV cache: " + std::to_string(budget) + " bytes fit " +
                                 std::to_string(blocks) + " blocks of " + std::to_string(block) +
                                 " bytes, at least 2 are needed");
    return int(std::min<int64_t>(blocks, INT_MAX));
}

}  // namespace engine
