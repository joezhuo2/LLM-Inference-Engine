#pragma once

#include <cstdint>
#include <functional>
#include <span>

#include "engine/core/stream.h"
#include "engine/core/tensor.h"
#include "engine/kernels/gemm.h"
#include "engine/model/config.h"
#include "engine/model/weights.h"
#include "engine/runtime/device_buffer.h"

namespace engine {

using LayerHook = std::function<void(int layer, const Tensor& hidden)>;

class ModelRunner {
public:
    ModelRunner(const ModelConfig& config, const ModelWeights& weights, int64_t max_tokens,
                int64_t max_context, Stream stream);

    Tensor forward(std::span<const int32_t> ids, bool all_logits = false,
                   const LayerHook& hook = {});
    void reset() { length_ = 0; }

    int64_t length() const { return length_; }
    int64_t max_context() const { return max_context_; }
    Stream stream() const { return stream_; }

private:
    ModelConfig config_;
    ModelWeights weights_;
    int64_t max_tokens_;
    int64_t max_context_;
    int64_t length_ = 0;
    Stream stream_;
    DeviceBuffer workspace_;
    Blas blas_;
    DeviceBuffer tokens_, x_, norm_, qkv_, attn_, proj_, gate_up_, act_, logits_, kv_;
};

}  // namespace engine
