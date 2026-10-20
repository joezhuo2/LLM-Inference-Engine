#include "engine/runtime/paged_runner.h"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine/kernels/add.h"
#include "engine/kernels/add_rmsnorm.h"
#include "engine/kernels/embedding.h"
#include "engine/kernels/paged_attention_v2.h"
#include "engine/kernels/rmsnorm.h"
#include "engine/kernels/rope.h"
#include "engine/kernels/swiglu.h"
#include "engine/kernels/write_kv.h"
#include "engine/runtime/nvtx.h"
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
    CUDA_CHECK(cudaEventCreateWithFlags(&staged_, cudaEventDisableTiming));
    require(cache.layers() == config.layers, "the cache must have one K and V plane per layer");
    require(max_seqs > 0 && max_seqs <= max_tokens,
            "max_seqs must be positive and at most max_tokens");
    require(max_context > 0 && max_context <= config.max_positions,
            "max_context must be positive and at most max_positions");
    max_blocks_ = (max_context + cache.block_size() - 1) / cache.block_size();
    const int64_t h = config.hidden;
    ints_ = buffer(1, 4 * max_tokens + max_tokens * max_blocks_, DType::I32);
    staging_ = PinnedBuffer(ints_.size());
    x_ = buffer(max_tokens, h, DType::BF16);
    norm_ = buffer(max_tokens, h, DType::BF16);
    qkv_ = buffer(max_tokens, config.qkv_dim(), DType::BF16);
    attn_ = buffer(max_tokens, h, DType::BF16);
    proj_ = buffer(max_tokens, h, DType::BF16);
    gate_up_ = buffer(max_tokens, 2 * int64_t(config.intermediate), DType::BF16);
    act_ = buffer(max_tokens, config.intermediate, DType::BF16);
    logits_ = buffer(max_seqs, config.vocab, DType::BF16);
}

Tensor PagedRunner::logits(int64_t rows) {
    require(rows > 0 && rows <= max_seqs_, "rows must be between 1 and max_seqs");
    return view(logits_, rows, config_.vocab);
}

PagedRunner::~PagedRunner() {
    if (staged_ != nullptr) cudaEventDestroy(staged_);
}

Tensor PagedRunner::forward(const PagedBatch& batch) {
    stage(batch);
    return run_staged(batch.num_tokens(), batch.num_seqs, batch.logit_rows);
}

void PagedRunner::stage(const PagedBatch& batch) {
    const NvtxRange range("stage");
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

    CUDA_CHECK(cudaEventSynchronize(staged_));
    auto* host = static_cast<int32_t*>(staging_.data());
    size_t at = 0;
    for (const std::vector<int32_t>* part :
         {&batch.token_ids, &batch.positions, &batch.slot_mapping, &batch.context_lens,
          &batch.block_tables}) {
        std::copy(part->begin(), part->end(), host + at);
        at += part->size();
    }
    CUDA_CHECK(
        cudaMemcpyAsync(ints_.data(), host, at * sizeof(int32_t), cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaEventRecord(staged_, stream_));
}

Tensor PagedRunner::run_staged(int64_t t, int64_t s, std::span<const int32_t> logit_rows) {
    require(t > 0 && t <= max_tokens_, "the number of tokens must be between 1 and max_tokens");
    require(s > 0 && s <= max_seqs_, "the number of sequences must be between 1 and max_seqs");
    const bool identity = logit_rows.empty();
    require(!identity || t == s, "without logit_rows every token must be its own sequence");
    require(identity || int64_t(logit_rows.size()) == s,
            "logit_rows must have one entry per sequence");
    for (const int32_t row : logit_rows)
        require(row >= 0 && row < t, "a logit row is outside the batch");

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

    ranged("embedding", [&] { naive::embedding(x, input_ids, weights_.embed, stream_); });
    for (int i = 0; i < c.layers; ++i) {
        char name[24];
        std::snprintf(name, sizeof name, "layer %d", i);
        const NvtxRange layer_range(name);
        const LayerWeights& w = weights_.layers[size_t(i)];
        if (i == 0) ranged("norm1", [&] { naive::rmsnorm(norm, x, w.ln1, eps, stream_); });
        ranged("qkv", [&] { gemm(blas_, qkv, norm, w.wqkv); });
        ranged("rope", [&] {
            fast::rope(qkv, positions, c.heads, c.kv_heads, float(c.rope_theta), stream_);
        });
        ranged("write_kv", [&] { write_kv(cache_->k(i), cache_->v(i), qkv, slots, stream_); });
        ranged("attention", [&] {
            paged_attention_decode_v2(attn, qkv, cache_->k(i), cache_->v(i), block_tables,
                                      context_lens, stream_);
        });
        ranged("o_proj", [&] { gemm(blas_, proj, attn, w.wo); });
        ranged("add1_norm2", [&] { fast::add_rmsnorm(x, norm, x, proj, w.ln2, eps, stream_); });
        ranged("gate_up", [&] { gemm(blas_, gate_up, norm, w.wgu); });
        ranged("swiglu", [&] { fast::swiglu(act, gate_up, stream_); });
        ranged("down", [&] { gemm(blas_, proj, act, w.wdown); });
        if (i + 1 < c.layers) {
            const Tensor& next_ln1 = weights_.layers[size_t(i) + 1].ln1;
            ranged("add2_norm1",
                   [&] { fast::add_rmsnorm(x, norm, x, proj, next_ln1, eps, stream_); });
        } else {
            ranged("add2", [&] { naive::add(x, x, proj, stream_); });
        }
    }

    Tensor last(x_.data(), DType::BF16, {s, c.hidden});
    if (!identity) {
        const size_t row_bytes = size_t(c.hidden) * dtype_size(DType::BF16);
        for (int64_t i = 0; i < s; ++i) {
            CUDA_CHECK(cudaMemcpyAsync(
                static_cast<std::byte*>(attn_.data()) + size_t(i) * row_bytes,
                static_cast<std::byte*>(x_.data()) + size_t(logit_rows[size_t(i)]) * row_bytes,
                row_bytes, cudaMemcpyDeviceToDevice, stream_));
        }
        last = Tensor(attn_.data(), DType::BF16, {s, c.hidden});
    }
    const Tensor last_norm(norm_.data(), DType::BF16, {s, c.hidden});
    ranged("final_norm",
           [&] { naive::rmsnorm(last_norm, last, weights_.final_norm, eps, stream_); });
    const Tensor out = logits(s);
    ranged("lm_head", [&] { gemm(blas_, out, last_norm, weights_.lm_head); });
    return out;
}

}  // namespace engine
