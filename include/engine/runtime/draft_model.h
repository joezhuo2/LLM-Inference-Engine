#pragma once

#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

#include "engine/kv/block_manager.h"
#include "engine/runtime/device_buffer.h"
#include "engine/runtime/kv_cache.h"
#include "engine/runtime/paged_runner.h"
#include "engine/runtime/sampler.h"
#include "engine/runtime/weights.h"

namespace engine {

int draft_model_blocks(int max_context, int max_seqs, int block_size);

class DraftModel {
public:
    DraftModel(const DeviceWeights& weights, int max_context, int max_seqs, Stream stream);

    std::vector<int32_t> propose(SeqId id, std::span<const int32_t> context, int want);

    int max_context() const { return max_context_; }
    int64_t forwards() const { return forwards_; }

private:
    bool reserve(SeqId id, int count);
    int32_t forward(SeqId id, int first, int count);

    int max_context_;
    BlockManager blocks_;
    KvCache cache_;
    PagedRunner runner_;
    Sampler sampler_;
    DeviceBuffer next_;
    std::unordered_map<SeqId, std::vector<int32_t>> fed_;
    int64_t forwards_ = 0;
};

}  // namespace engine
