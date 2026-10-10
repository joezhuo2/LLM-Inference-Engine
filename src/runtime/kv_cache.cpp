#include "engine/runtime/kv_cache.h"

#include <cstddef>
#include <stdexcept>

#include "kernels/cuda_check.h"

namespace engine {

size_t device_free_bytes() {
    size_t free = 0, total = 0;
    CUDA_CHECK(cudaMemGetInfo(&free, &total));
    return free;
}

KvCache::KvCache(const ModelConfig& config, int num_blocks, int block_size)
    : layers_(config.layers),
      kv_heads_(config.kv_heads),
      head_dim_(config.head_dim),
      num_blocks_(num_blocks),
      block_size_(block_size) {
    if (num_blocks < 2)
        throw std::invalid_argument(
            "KvCache: num_blocks must be at least 2 (the null block and one more)");
    buffer_ = DeviceBuffer(size_t(num_blocks) * size_t(kv_block_bytes(config, block_size)));
}

Tensor KvCache::k(int layer) {
    return at(layer, 0);
}

Tensor KvCache::v(int layer) {
    return at(layer, 1);
}

Tensor KvCache::at(int layer, int which) {
    if (layer < 0 || layer >= layers_) throw std::invalid_argument("KvCache: layer out of range");
    const size_t plane = bytes() / (2 * size_t(layers_));
    auto* base = static_cast<std::byte*>(buffer_.data()) + (2 * size_t(layer) + which) * plane;
    return Tensor(base, DType::BF16, {num_blocks_, kv_heads_, block_size_, head_dim_});
}

}  // namespace engine
