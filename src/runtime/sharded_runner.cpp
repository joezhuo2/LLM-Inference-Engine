#include "engine/runtime/sharded_runner.h"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

#include "engine/kernels/embedding.h"
#include "engine/kernels/rmsnorm.h"
#include "engine/loader/tensor_parallel.h"
#include "kernels/cuda_check.h"

namespace engine {

namespace {

constexpr size_t kWorkspaceBytes = size_t(32) << 20;

void require(bool ok, const char* what) {
    if (!ok) throw std::invalid_argument(std::string("ShardedRunner: ") + what);
}

DeviceBuffer buffer(int64_t rows, int64_t cols, DType dtype) {
    return DeviceBuffer(size_t(rows * cols) * dtype_size(dtype));
}

Tensor view(DeviceBuffer& b, int64_t rows, int64_t cols) {
    return Tensor(b.data(), DType::BF16, {rows, cols});
}

Tensor rows(const Tensor& t, int64_t begin, int64_t count) {
    const size_t row_bytes = size_t(t.shape[1]) * dtype_size(t.dtype);
    return Tensor(static_cast<std::byte*>(t.data) + size_t(begin) * row_bytes, t.dtype,
                  {count, t.shape[1]});
}

bool has_shape(const Tensor& t, int64_t rows, int64_t cols) {
    return t.ndim == 2 && t.shape[0] == rows && t.shape[1] == cols;
}

}  // namespace

ShardedRunner::ShardedRunner(const ModelConfig& config, std::vector<ModelWeights> ranks,
                             int64_t max_tokens, int64_t max_context, Stream stream)
    : config_(config),
      shard_(shard_config(config, int(ranks.size()))),
      max_tokens_(max_tokens),
      max_context_(max_context),
      stream_(stream),
      workspace_(kWorkspaceBytes),
      blas_(stream, workspace_.data(), workspace_.size()) {
    require(max_tokens > 0 && max_tokens <= max_context,
            "max_tokens must be positive and at most max_context");
    require(max_context <= config.max_positions, "max_context must be at most max_positions");
    const int64_t h = config.hidden;
    const ModelConfig& s = shard_;
    for (ModelWeights& w : ranks) {
        require(int(w.layers.size()) == config.layers, "a rank must have every layer");
        for (const LayerWeights& l : w.layers)
            require(has_shape(l.wqkv, s.qkv_dim(), h) && has_shape(l.wo, h, s.q_dim()) &&
                        has_shape(l.wgu, 2 * int64_t(s.intermediate), h) &&
                        has_shape(l.wdown, h, s.intermediate),
                    "the weights of a rank must have the shapes of one shard of this many ranks");
        Rank& r = ranks_.emplace_back();
        r.weights = std::move(w);
        r.qkv = buffer(max_tokens, s.qkv_dim(), DType::BF16);
        r.attn = buffer(max_tokens, s.q_dim(), DType::BF16);
        r.partial = buffer(max_tokens, h, DType::BF16);
        r.gate_up = buffer(max_tokens, 2 * int64_t(s.intermediate), DType::BF16);
        r.act = buffer(max_tokens, s.intermediate, DType::BF16);
        r.kv = buffer(2 * int64_t(s.layers) * max_context, s.kv_dim(), DType::BF16);
    }
    tokens_ = buffer(2, max_tokens, DType::I32);
    x_ = buffer(max_tokens, h, DType::BF16);
    norm_ = buffer(max_tokens, h, DType::BF16);
    logits_ = buffer(max_tokens, config.vocab, DType::BF16);
}

std::vector<RankLayer> ShardedRunner::layer_views(int layer, int64_t tokens) {
    const ModelConfig& s = shard_;
    std::vector<RankLayer> views;
    for (Rank& r : ranks_) {
        const Tensor kv = view(r.kv, 2 * int64_t(s.layers) * max_context_, s.kv_dim());
        views.push_back({.weights = r.weights.layers[size_t(layer)],
                         .qkv = view(r.qkv, tokens, s.qkv_dim()),
                         .attn = view(r.attn, tokens, s.q_dim()),
                         .partial = view(r.partial, tokens, s.hidden),
                         .gate_up = view(r.gate_up, tokens, 2 * int64_t(s.intermediate)),
                         .act = view(r.act, tokens, s.intermediate),
                         .k_cache = rows(kv, 2 * layer * max_context_, max_context_),
                         .v_cache = rows(kv, (2 * layer + 1) * max_context_, max_context_)});
    }
    return views;
}

Tensor ShardedRunner::forward(std::span<const int32_t> ids, bool all_logits,
                              const LayerHook& hook) {
    const auto t = int64_t(ids.size());
    require(t > 0 && t <= max_tokens_, "the number of tokens must be between 1 and max_tokens");
    require(length_ + t <= max_context_, "the sequence would exceed max_context");
    const int64_t start = length_;

    std::vector<int32_t> host(ids.begin(), ids.end());
    for (int64_t i = 0; i < t; ++i) host.push_back(int32_t(start + i));
    CUDA_CHECK(cudaMemcpyAsync(tokens_.data(), host.data(), host.size() * sizeof(int32_t),
                               cudaMemcpyHostToDevice, stream_));
    auto* token_data = static_cast<int32_t*>(tokens_.data());
    const Tensor input_ids(token_data, DType::I32, {t});
    const Tensor positions(token_data + t, DType::I32, {t});

    const ModelConfig& c = config_;
    const ModelWeights& whole = ranks_.front().weights;
    const Tensor x = view(x_, t, c.hidden);
    const Tensor norm = view(norm_, t, c.hidden);

    naive::embedding(x, input_ids, whole.embed, stream_);
    for (int i = 0; i < c.layers; ++i) {
        const std::vector<RankLayer> ranks = layer_views(i, t);
        sharded_layer(blas_, shard_, ranks, x, norm, positions, start);
        if (hook) hook(i, x);
    }
    naive::rmsnorm(norm, x, whole.final_norm, float(c.rms_eps), stream_);

    const int64_t n = all_logits ? t : 1;
    const Tensor logits = view(logits_, n, c.vocab);
    gemm(blas_, logits, rows(norm, t - n, n), whole.lm_head);
    length_ += t;
    return logits;
}

}  // namespace engine
