#include "engine/runtime/draft_model.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>

#include "engine/kv/cache_size.h"
#include "engine/sampling/sample.h"
#include "engine/sched/paged_batch.h"
#include "kernels/cuda_check.h"

namespace engine {

int draft_model_blocks(int max_context, int max_seqs, int block_size) {
    if (max_context < 1 || max_seqs < 1 || block_size < 1)
        throw std::invalid_argument("[draft_model_blocks] every argument must be positive");
    return max_seqs * ((max_context + block_size - 1) / block_size) + 1;
}

DraftModel::DraftModel(const DeviceWeights& weights, int max_context, int max_seqs, Stream stream)
    : max_context_(std::min(max_context, weights.config.max_positions)),
      blocks_(draft_model_blocks(max_context_, max_seqs, kKvBlockSize), kKvBlockSize),
      cache_(weights.config, draft_model_blocks(max_context_, max_seqs, kKvBlockSize),
             kKvBlockSize),
      runner_(weights.config, weights.weights, cache_, max_context_, 1, max_context_, stream),
      sampler_(1),
      next_(sizeof(int32_t)) {}

bool DraftModel::reserve(SeqId id, int count) {
    const auto fits = [&] {
        if (!fed_.count(id)) return blocks_.can_allocate(count);
        const int have = int(blocks_.table(id).size()) * kKvBlockSize - blocks_.length(id);
        return count <= have + blocks_.num_free() * kKvBlockSize;
    };
    if (!fits()) {
        for (auto it = fed_.begin(); it != fed_.end();) {
            if (it->first == id) {
                ++it;
                continue;
            }
            blocks_.free(it->first);
            it = fed_.erase(it);
        }
        if (!fits()) return false;
    }
    if (!fed_.count(id)) {
        blocks_.allocate(id, count);
        fed_[id];
    } else {
        for (int i = 0; i < count; ++i) blocks_.append_slot(id);
    }
    return true;
}

int32_t DraftModel::forward(SeqId id, int first, int count) {
    const std::vector<int32_t>& tokens = fed_.at(id);
    const std::vector<int>& table = blocks_.table(id);
    PagedBatch batch;
    batch.prefill = count > 1;
    batch.num_seqs = 1;
    batch.max_blocks = runner_.max_blocks();
    for (int pos = first; pos < first + count; ++pos) {
        batch.token_ids.push_back(tokens[size_t(pos)]);
        batch.positions.push_back(pos);
        batch.slot_mapping.push_back(blocks_.slot(id, pos));
        batch.context_lens.push_back(pos + 1);
        const size_t row = batch.block_tables.size();
        batch.block_tables.resize(row + size_t(batch.max_blocks), 0);
        std::copy(table.begin(), table.end(), batch.block_tables.begin() + int64_t(row));
    }
    batch.logit_rows.push_back(count - 1);
    if (batch.prefill) {
        batch.chunk_starts = {0, count};
        batch.cached_lens = {first};
    }

    const Tensor logits = runner_.forward(batch);
    const SampleRow row{};
    const Tensor ids(next_.data(), DType::I32, {1});
    sampler_(ids, logits, std::span<const SampleRow>(&row, 1), runner_.stream());
    int32_t token = 0;
    CUDA_CHECK(cudaMemcpyAsync(&token, next_.data(), sizeof token, cudaMemcpyDeviceToHost,
                               runner_.stream()));
    CUDA_CHECK(cudaStreamSynchronize(runner_.stream()));
    ++forwards_;
    return token;
}

std::vector<int32_t> DraftModel::propose(SeqId id, std::span<const int32_t> context, int want) {
    if (want < 1 || context.empty() || int(context.size()) + want > max_context_) return {};

    size_t common = 0;
    const auto known = fed_.find(id);
    if (known != fed_.end()) {
        std::vector<int32_t>& fed = known->second;
        const size_t limit = std::min(fed.size(), context.size() - 1);
        while (common < limit && fed[common] == context[common]) ++common;
        if (common == 0) {
            blocks_.free(id);
            fed_.erase(known);
        } else if (common < fed.size()) {
            blocks_.truncate(id, int(common));
            fed.resize(common);
        }
    }

    const int fresh = int(context.size() - common);
    if (!reserve(id, fresh + want - 1)) return {};
    std::vector<int32_t>& fed = fed_.at(id);
    fed.insert(fed.end(), context.begin() + int64_t(common), context.end());

    std::vector<int32_t> draft{forward(id, int(common), fresh)};
    while (int(draft.size()) < want) {
        fed.push_back(draft.back());
        draft.push_back(forward(id, int(fed.size()) - 1, 1));
    }
    return draft;
}

}  // namespace engine
