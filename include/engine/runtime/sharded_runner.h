#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "engine/core/stream.h"
#include "engine/core/tensor.h"
#include "engine/kernels/gemm.h"
#include "engine/model/config.h"
#include "engine/model/weights.h"
#include "engine/runtime/device_buffer.h"
#include "engine/runtime/model_runner.h"

namespace engine {

struct RankLayer {
    LayerWeights weights;
    Tensor qkv, attn, partial, gate_up, act, k_cache, v_cache;
};

void sharded_layer(const Blas& blas, const ModelConfig& shard, std::span<const RankLayer> ranks,
                   const Tensor& x, const Tensor& norm, const Tensor& positions, int64_t start);

class ShardedRunner {
public:
    ShardedRunner(const ModelConfig& config, std::vector<ModelWeights> ranks, int64_t max_tokens,
                  int64_t max_context, Stream stream);

    Tensor forward(std::span<const int32_t> ids, bool all_logits = false,
                   const LayerHook& hook = {});
    void reset() { length_ = 0; }

    int num_ranks() const { return int(ranks_.size()); }
    int64_t length() const { return length_; }
    int64_t max_context() const { return max_context_; }
    Stream stream() const { return stream_; }

private:
    struct Rank {
        ModelWeights weights;
        DeviceBuffer qkv, attn, partial, gate_up, act, kv;
    };

    std::vector<RankLayer> layer_views(int layer, int64_t tokens);

    ModelConfig config_;
    ModelConfig shard_;
    std::vector<Rank> ranks_;
    int64_t max_tokens_;
    int64_t max_context_;
    int64_t length_ = 0;
    Stream stream_;
    DeviceBuffer workspace_;
    Blas blas_;
    DeviceBuffer tokens_, x_, norm_, logits_;
};

}  // namespace engine
