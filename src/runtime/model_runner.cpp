#include "engine/runtime/model_runner.h"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine/kernels/add.h"
#include "engine/kernels/attention.h"
#include "engine/kernels/embedding.h"
#include "engine/kernels/rmsnorm.h"
#include "engine/kernels/rope.h"
#include "engine/kernels/swiglu.h"
#include "kernels/cuda_check.h"

namespace engine {

namespace {

constexpr size_t kWorkspaceBytes = size_t(32) << 20;

void require(bool ok, const char* what) {
    if (!ok) throw std::invalid_argument(std::string("ModelRunner: ") + what);
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

}  // namespace

ModelRunner::ModelRunner(const ModelConfig& config, const ModelWeights& weights, int64_t max_tokens,
                         int64_t max_context, Stream stream)
    : config_(config),
      weights_(weights),
      max_tokens_(max_tokens),
      max_context_(max_context),
      stream_(stream),
      workspace_(kWorkspaceBytes),
      blas_(stream, workspace_.data(), workspace_.size()) {
    require(max_tokens > 0 && max_tokens <= max_context,
            "max_tokens must be positive and at most max_context");
    require(max_context <= config.max_positions, "max_context must be at most max_positions");
    const int64_t h = config.hidden;
    tokens_ = buffer(2, max_tokens, DType::I32);
    x_ = buffer(max_tokens, h, DType::BF16);
    norm_ = buffer(max_tokens, h, DType::BF16);
    qkv_ = buffer(max_tokens, config.qkv_dim(), DType::BF16);
    attn_ = buffer(max_tokens, h, DType::BF16);
    proj_ = buffer(max_tokens, h, DType::BF16);
    gate_up_ = buffer(max_tokens, 2 * int64_t(config.intermediate), DType::BF16);
    act_ = buffer(max_tokens, config.intermediate, DType::BF16);
    logits_ = buffer(max_tokens, config.vocab, DType::BF16);
    kv_ = buffer(2 * int64_t(config.layers) * max_context, config.kv_dim(), DType::BF16);
}

Tensor ModelRunner::forward(std::span<const int32_t> ids, bool all_logits, const LayerHook& hook) {
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
    const auto eps = float(c.rms_eps);
    const Tensor x = view(x_, t, c.hidden);
    const Tensor norm = view(norm_, t, c.hidden);
    const Tensor qkv = view(qkv_, t, c.qkv_dim());
    const Tensor attn = view(attn_, t, c.hidden);
    const Tensor proj = view(proj_, t, c.hidden);
    const Tensor gate_up = view(gate_up_, t, 2 * int64_t(c.intermediate));
    const Tensor act = view(act_, t, c.intermediate);
    const Tensor kv = view(kv_, 2 * int64_t(c.layers) * max_context_, c.kv_dim());

    naive::embedding(x, input_ids, weights_.embed, stream_);
    for (int i = 0; i < c.layers; ++i) {
        const LayerWeights& w = weights_.layers[size_t(i)];
        const Tensor k_cache = rows(kv, 2 * i * max_context_, max_context_);
        const Tensor v_cache = rows(kv, (2 * i + 1) * max_context_, max_context_);
        naive::rmsnorm(norm, x, w.ln1, eps, stream_);
        gemm(blas_, qkv, norm, w.wqkv);
        naive::rope(qkv, positions, c.heads, c.kv_heads, float(c.rope_theta), stream_);
        naive::store_kv(k_cache, v_cache, qkv, start, stream_);
        naive::attention(attn, qkv, k_cache, v_cache, start, c.heads, c.kv_heads, stream_);
        gemm(blas_, proj, attn, w.wo);
        naive::add(x, x, proj, stream_);
        naive::rmsnorm(norm, x, w.ln2, eps, stream_);
        gemm(blas_, gate_up, norm, w.wgu);
        naive::swiglu(act, gate_up, stream_);
        gemm(blas_, proj, act, w.wdown);
        naive::add(x, x, proj, stream_);
        if (hook) hook(i, x);
    }
    naive::rmsnorm(norm, x, weights_.final_norm, eps, stream_);

    const int64_t n = all_logits ? t : 1;
    const Tensor logits = view(logits_, n, c.vocab);
    gemm(blas_, logits, rows(norm, t - n, n), weights_.lm_head);
    length_ += t;
    return logits;
}

}  // namespace engine
