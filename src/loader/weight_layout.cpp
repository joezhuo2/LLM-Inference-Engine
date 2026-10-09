#include "engine/loader/weight_layout.h"

#include <algorithm>
#include <set>
#include <stdexcept>

namespace engine {

namespace {

[[noreturn]] void fail(const std::string& what) {
    throw std::runtime_error("weights: " + what);
}

std::string shape_string(const std::vector<int64_t>& shape) {
    std::string s = "[";
    for (size_t i = 0; i < shape.size(); ++i) s += (i ? ", " : "") + std::to_string(shape[i]);
    return s + "]";
}

uint64_t bf16_bytes(const std::vector<int64_t>& shape) {
    uint64_t n = dtype_size(DType::BF16);
    for (int64_t d : shape) n *= uint64_t(d);
    return n;
}

class Planner {
public:
    explicit Planner(const SafetensorsHeader& header) : header_(header) {}

    Region region(std::vector<int64_t> shape) {
        end_ = (end_ + kWeightAlignment - 1) / kWeightAlignment * kWeightAlignment;
        Region r{end_, bf16_bytes(shape), std::move(shape)};
        end_ += r.bytes;
        return r;
    }

    void copy(const std::string& name, const std::vector<int64_t>& shape, uint64_t dst) {
        const auto it = header_.tensors.find(name);
        if (it == header_.tensors.end()) fail("missing tensor " + name);
        const TensorEntry& t = it->second;
        if (t.dtype != DType::BF16) fail(name + " must be BF16");
        if (t.shape != shape)
            fail(name + " has shape " + shape_string(t.shape) + ", expected " +
                 shape_string(shape));
        copies_.push_back({name, t.offset, dst, t.bytes, shape});
        used_.insert(name);
    }

    std::vector<CopyOp> finish() {
        for (const auto& [name, t] : header_.tensors)
            if (!used_.contains(name)) fail("unexpected tensor " + name);
        std::sort(copies_.begin(), copies_.end(),
                  [](const CopyOp& a, const CopyOp& b) { return a.src < b.src; });
        return std::move(copies_);
    }

    uint64_t end() const { return end_; }

private:
    const SafetensorsHeader& header_;
    std::vector<CopyOp> copies_;
    std::set<std::string> used_;
    uint64_t end_ = 0;
};

}  // namespace

WeightLayout plan_weight_layout(const ModelConfig& c, const SafetensorsHeader& header) {
    if (c.tie_embeddings) fail("tied embeddings are not supported");
    const int64_t d = c.hidden, kv = c.kv_dim(), ff = c.intermediate, vocab = c.vocab;
    const uint64_t row = uint64_t(d) * dtype_size(DType::BF16);

    Planner p(header);
    WeightLayout w;
    w.embed = p.region({vocab, d});
    p.copy("model.embed_tokens.weight", {vocab, d}, w.embed.offset);

    for (int i = 0; i < c.layers; ++i) {
        const std::string pre = "model.layers." + std::to_string(i) + ".";
        LayerLayout& l = w.layers.emplace_back();

        l.ln1 = p.region({d});
        p.copy(pre + "input_layernorm.weight", {d}, l.ln1.offset);

        l.wqkv = p.region({c.qkv_dim(), d});
        p.copy(pre + "self_attn.q_proj.weight", {d, d}, l.wqkv.offset);
        p.copy(pre + "self_attn.k_proj.weight", {kv, d}, l.wqkv.offset + uint64_t(d) * row);
        p.copy(pre + "self_attn.v_proj.weight", {kv, d}, l.wqkv.offset + uint64_t(d + kv) * row);

        l.wo = p.region({d, d});
        p.copy(pre + "self_attn.o_proj.weight", {d, d}, l.wo.offset);

        l.ln2 = p.region({d});
        p.copy(pre + "post_attention_layernorm.weight", {d}, l.ln2.offset);

        l.wgu = p.region({2 * ff, d});
        p.copy(pre + "mlp.gate_proj.weight", {ff, d}, l.wgu.offset);
        p.copy(pre + "mlp.up_proj.weight", {ff, d}, l.wgu.offset + uint64_t(ff) * row);

        l.wdown = p.region({d, ff});
        p.copy(pre + "mlp.down_proj.weight", {d, ff}, l.wdown.offset);
    }

    w.final_norm = p.region({d});
    p.copy("model.norm.weight", {d}, w.final_norm.offset);
    w.lm_head = p.region({vocab, d});
    p.copy("lm_head.weight", {vocab, d}, w.lm_head.offset);

    w.copies = p.finish();
    w.total_bytes = p.end();
    return w;
}

}  // namespace engine
