#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "engine/core/tensor.h"
#include "engine/loader/safetensors.h"
#include "engine/model/config.h"

namespace engine::test {

inline ModelConfig tiny_config() {
    ModelConfig c;
    c.hidden = 8;
    c.layers = 2;
    c.heads = 2;
    c.kv_heads = 1;
    c.head_dim = 4;
    c.intermediate = 12;
    c.vocab = 16;
    c.max_positions = 32;
    c.rope_theta = 10000.0;
    c.rms_eps = 1e-5;
    c.bos_id = 1;
    c.eos_id = 2;
    return c;
}

inline ModelConfig tinyllama_config() {
    ModelConfig c = tiny_config();
    c.hidden = 2048;
    c.layers = 22;
    c.heads = 32;
    c.kv_heads = 4;
    c.head_dim = 64;
    c.intermediate = 5632;
    c.vocab = 32000;
    c.max_positions = 2048;
    return c;
}

inline std::vector<std::pair<std::string, std::vector<int64_t>>> llama_tensors(
    const ModelConfig& c) {
    const int64_t d = c.hidden, kv = c.kv_dim(), ff = c.intermediate, v = c.vocab;
    std::vector<std::pair<std::string, std::vector<int64_t>>> t = {
        {"model.embed_tokens.weight", {v, d}},
        {"model.norm.weight", {d}},
        {"lm_head.weight", {v, d}}};
    for (int i = 0; i < c.layers; ++i) {
        const std::string p = "model.layers." + std::to_string(i) + ".";
        t.push_back({p + "input_layernorm.weight", {d}});
        t.push_back({p + "self_attn.q_proj.weight", {d, d}});
        t.push_back({p + "self_attn.k_proj.weight", {kv, d}});
        t.push_back({p + "self_attn.v_proj.weight", {kv, d}});
        t.push_back({p + "self_attn.o_proj.weight", {d, d}});
        t.push_back({p + "post_attention_layernorm.weight", {d}});
        t.push_back({p + "mlp.gate_proj.weight", {ff, d}});
        t.push_back({p + "mlp.up_proj.weight", {ff, d}});
        t.push_back({p + "mlp.down_proj.weight", {d, ff}});
    }
    return t;
}

inline SafetensorsHeader fake_header(const ModelConfig& c, uint64_t data_offset = 0) {
    SafetensorsHeader h;
    h.data_offset = data_offset;
    for (auto& [name, shape] : llama_tensors(c)) h.tensors[name] = {DType::BF16, shape, 0, 0};
    uint64_t at = data_offset;
    for (auto& [name, t] : h.tensors) {
        t.bytes = dtype_size(t.dtype);
        for (int64_t d : t.shape) t.bytes *= uint64_t(d);
        t.offset = at;
        at += t.bytes;
    }
    return h;
}

}  // namespace engine::test
