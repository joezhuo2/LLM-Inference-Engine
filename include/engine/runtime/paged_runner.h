#pragma once

#include <cstdint>
#include <span>

#include "engine/core/stream.h"
#include "engine/core/tensor.h"
#include "engine/kernels/gemm.h"
#include "engine/model/config.h"
#include "engine/model/weights.h"
#include "engine/runtime/device_buffer.h"
#include "engine/runtime/kv_cache.h"
#include "engine/runtime/pinned_buffer.h"
#include "engine/sched/paged_batch.h"

struct CUevent_st;

namespace engine {

class PagedRunner {
public:
    PagedRunner(const ModelConfig& config, const ModelWeights& weights, KvCache& cache,
                int64_t max_tokens, int64_t max_seqs, int64_t max_context, Stream stream);

    ~PagedRunner();
    PagedRunner(const PagedRunner&) = delete;
    PagedRunner& operator=(const PagedRunner&) = delete;

    Tensor forward(const PagedBatch& batch);

    void stage(const PagedBatch& batch);
    Tensor run_staged(int64_t tokens, int64_t seqs, std::span<const int32_t> logit_rows = {});

    Tensor logits(int64_t rows);

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
    int64_t longest_context_ = 0;
    Stream stream_;
    DeviceBuffer workspace_;
    Blas blas_;
    PinnedBuffer staging_;
    CUevent_st* staged_ = nullptr;
    DeviceBuffer ints_, x_, norm_, qkv_, attn_, proj_, gate_up_, act_, logits_, scratch_;
};

}  // namespace engine
