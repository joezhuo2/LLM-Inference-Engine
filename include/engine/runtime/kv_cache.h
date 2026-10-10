#pragma once

#include <cstddef>

#include "engine/core/tensor.h"
#include "engine/kv/cache_size.h"
#include "engine/model/config.h"
#include "engine/runtime/device_buffer.h"

namespace engine {

size_t device_free_bytes();

class KvCache {
public:
    KvCache(const ModelConfig& config, int num_blocks, int block_size = kKvBlockSize);

    Tensor k(int layer);
    Tensor v(int layer);

    int layers() const { return layers_; }
    int num_blocks() const { return num_blocks_; }
    int block_size() const { return block_size_; }
    size_t bytes() const { return buffer_.size(); }

private:
    Tensor at(int layer, int which);

    int layers_;
    int kv_heads_;
    int head_dim_;
    int num_blocks_;
    int block_size_;
    DeviceBuffer buffer_;
};

}  // namespace engine
