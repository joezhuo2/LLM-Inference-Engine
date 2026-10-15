#include "engine/runtime/paged_runner.h"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine/kernels/add.h"
#include "engine/kernels/embedding.h"
#include "engine/kernels/paged_attention.h"
#include "engine/kernels/rmsnorm.h"
#include "engine/kernels/rope.h"
#include "engine/kernels/swiglu.h"
#include "engine/kernels/write_kv.h"
#include "kernels/cuda_check.h"

namespace engine {

namespace {

constexpr size_t kWorkspaceBytes = size_t(32) << 20;

void require(bool ok, const char* what) {
    if (!ok) throw std::invalid_argument(std::string("PagedRunner: ") + what);
}

DeviceBuffer buffer(int64_t rows, int64_t cols, DType dtype) {
    return DeviceBuffer(size_t(rows * cols) * dtype_size(dtype));
}

Tensor view(DeviceBuffer& b, int64_t rows, int64_t cols) {
    return Tensor(b.data(), DType::BF16, {rows, cols});
}

}  // namespace

PagedRunner::PagedRunner(const ModelConfig& config, const ModelWeights& weights, KvCache& cache,
                         int64_t max_tokens, int64_t max_seqs, int64_t max_context, Stream stream)
    : config_(config),
      weights_(weights),
      cache_(&cache),
      max_tokens_(max_tokens),
      max_seqs_(max_seqs),
      stream_(stream),
      workspace_(kWorkspaceBytes),
      blas_(stream, workspace_.data(), workspace_.size()) {
    require(cache.layers() == config.layers, "the cache must have one K and V plane per layer");
    require(max_seqs > 0 && max_seqs <= max_tokens,
            "max_seqs must be positive and at most max_tokens");
    require(max_context > 0 && max_context <= config.max_positions,
            "max_context must be positive and at most max_positions");
    max_blocks_ = (max_context + cache.block_size() - 1) / cache.block_size();
    const int64_t h = config.hidden;
    ints_ = buffer(1, 4 * max_tokens + max_tokens * max_blocks_, DType::I32);
    x_ = buffer(max_tokens, h, DType::BF16);
    norm_ = buffer(max_tokens, h, DType::BF16);
    qkv_ = buffer(max_tokens, config.qkv_dim(), DType::BF16);
    attn_ = buffer(max_tokens, h, DType::BF16);
    proj_ = buffer(max_tokens, h, DType::BF16);
    gate_up_ = buffer(max_tokens, 2 * int64_t(config.intermediate), DType::BF16);
    act_ = buffer(max_tokens, config.intermediate, DType::BF16);
    logits_ = buffer(max_seqs, config.vocab, DType::BF16);
}

Tensor PagedRunner::forward(const PagedBatch& batch) {
    const int64_t t = batch.num_tokens();
    const int64_t s = batch.num_seqs;
    require(t > 0 && t <= max_tokens_, "the number of tokens must be between 1 and max_tokens");
    require(s > 0 && s <= max_seqs_, "the number of sequences must be between 1 and max_seqs");
    require(batch.max_blocks == max_blocks_, "the block table width must equal max_blocks");
    require(int64_t(batch.positions.size()) == t && int64_t(batch.slot_mapping.size()) == t &&
                int64_t(batch.context_lens.size()) == t,
            "positions, slot_mapping and context_lens must have one entry per token");
    require(int64_t(batch.logit_rows.size()) == s, "logit_rows must have one entry per sequence");
    require(int64_t(batch.block_tables.size()) == t * max_blocks_,
            "block_tables must have max_blocks entries per token");
    for (const int32_t row : batch.logit_rows)
        require(row >= 0 && row < t, "a logit row is outside the batch");

    std::vector<int32_t> host;
    host.reserve(size_t(4 * t + t * max_blocks_));
    host.insert(host.end(), batch.token_ids.begin(), batch.token_ids.end());
    host.insert(host.end(), batch.positions.begin(), batch.positions.end());
    host.insert(host.end(), batch.slot_mapping.begin(), batch.slot_mapping.end());
    host.insert(host.end(), batch.context_lens.begin(), batch.context_lens.end());
    host.insert(host.end(), batch.block_tables.begin(), batch.block_tables.end());
    CUDA_CHECK(cudaMemcpyAsync(ints_.data(), host.data(), host.size() * sizeof(int32_t),
                               cudaMemcpyHostToDevice, stream_));
    auto* ints = static_cast<int32_t*>(ints_.data());
    const Tensor input_ids(ints, DType::I32, {t});
    const Tensor positions(ints + t, DType::I32, {t});
    const Tensor slots(ints + 2 * t, DType::I32, {t});
    const Tensor context_lens(ints + 3 * t, DType::I32, {t});
    const Tensor block_tables(ints + 4 * t, DType::I32, {t, max_blocks_});

    const ModelConfig& c = config_;
    const auto eps = float(c.rms_eps);
    const Tensor x = view(x_, t, c.hidden);
    const Tensor norm = view(norm_, t, c.hidden);
    const Tensor qkv = view(qkv_, t, c.qkv_dim());
    const Tensor attn = view(attn_, t, c.hidden);
    const Tensor proj = view(proj_, t, c.hidden);
    const Tensor gate_up = view(gate_up_, t, 2 * int64_t(c.intermediate));
    const Tensor act = view(act_, t, c.intermediate);

    naive::embedding(x, input_ids, weights_.embed, stream_);
    for (int i = 0; i < c.layers; ++i) {
        const LayerWeights& w = weights_.layers[size_t(i)];
        naive::rmsnorm(norm, x, w.ln1, eps, stream_);
        gemm(blas_, qkv, norm, w.wqkv);
        naive::rope(qkv, positions, c.heads, c.kv_heads, float(c.rope_theta), stream_);
        write_kv(cache_->k(i), cache_->v(i), qkv, slots, stream_);
        paged_attention_decode(attn, qkv, cache_->k(i), cache_->v(i), block_tables, context_lens,
                               stream_);
        gemm(blas_, proj, attn, w.wo);
        naive::add(x, x, proj, stream_);
        naive::rmsnorm(norm, x, w.ln2, eps, stream_);
        gemm(blas_, gate_up, norm, w.wgu);
        naive::swiglu(act, gate_up, stream_);
        gemm(blas_, proj, act, w.wdown);
        naive::add(x, x, proj, stream_);
    }

    const size_t row_bytes = size_t(c.hidden) * dtype_size(DType::BF16);
    for (int64_t i = 0; i < s; ++i) {
        CUDA_CHECK(cudaMemcpyAsync(
            static_cast<std::byte*>(attn_.data()) + size_t(i) * row_bytes,
            static_cast<std::byte*>(x_.data()) + size_t(batch.logit_rows[size_t(i)]) * row_bytes,
            row_bytes, cudaMemcpyDeviceToDevice, stream_));
    }
    const Tensor last(attn_.data(), DType::BF16, {s, c.hidden});
    const Tensor last_norm(norm_.data(), DType::BF16, {s, c.hidden});
    naive::rmsnorm(last_norm, last, weights_.final_norm, eps, stream_);
    const Tensor logits = view(logits_, s, c.vocab);
    gemm(blas_, logits, last_norm, weights_.lm_head);
    return logits;
}

}  // namespace engine
