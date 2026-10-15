#pragma once

#include <cstdint>

#include "engine/core/stream.h"
#include "engine/core/tensor.h"
#include "engine/kernels/gemm.h"
#include "engine/model/config.h"
#include "engine/model/weights.h"
#include "engine/runtime/device_buffer.h"
#include "engine/runtime/kv_cache.h"
#include "engine/sched/paged_batch.h"

namespace engine {

class PagedRunner {
public:
    PagedRunner(const ModelConfig& config, const ModelWeights& weights, KvCache& cache,
                int64_t max_tokens, int64_t max_seqs, int64_t max_context, Stream stream);

    Tensor forward(const PagedBatch& batch);

    int64_t max_tokens() const { return max_tokens_; }
    int64_t max_seqs() const { return max_seqs_; }
    int64_t max_blocks() const { return max_blocks_; }
    Stream stream() const { return stream_; }

private:
    ModelConfig config_;
    ModelWeights weights_;
    KvCache* cache_;
    int64_t max_tokens_;
    int64_t max_seqs_;
    int64_t max_blocks_;
    Stream stream_;
    DeviceBuffer workspace_;
    Blas blas_;
    DeviceBuffer ints_, x_, norm_, qkv_, attn_, proj_, gate_up_, act_, logits_;
};

}  // namespace engine
