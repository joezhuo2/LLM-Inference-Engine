#include "engine/loader/tensor_parallel.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace engine {

namespace {

constexpr uint64_t align_up(uint64_t byte_os, uint64_t alignment) {
    return (byte_os + alignment - 1) & ~(alignment - 1);
}

std::string shape_string(const std::vector<int64_t>& shape) {
    std::string text = "[";

    for (size_t i = 0; i < shape.size(); ++i) {
        text += (i ? ", " : "") + std::to_string(shape[i]);
    }

    return text + "]";
}

const TensorEntry& require_bf16_tensor(const SafetensorsHeader& header, const std::string& name,
                                       const std::vector<int64_t>& shape,
                                       std::unordered_set<std::string>& seen) {
    seen.insert(name);

    auto it = header.tensors.find(name);

    if (it == header.tensors.end()) {
        throw std::runtime_error("weights: missing tensor " + name);
    }

    const TensorEntry& info = it->second;

    if (info.dtype != DType::BF16) {
        throw std::runtime_error("weights: " + name + " must be BF16");
    }

    if (info.shape != shape) {
        throw std::runtime_error("weights: " + name + " has shape " + shape_string(info.shape) +
                                 ", expected " + shape_string(shape));
    }

    uint64_t bytes = dtype_size(DType::BF16);

    for (int64_t dim_size : shape) {
        bytes *= static_cast<uint64_t>(dim_size);
    }

    if (info.bytes != bytes) {
        throw std::runtime_error("weights: " + name +
                                 " has a byte size that does not match its shape");
    }

    return info;
}

Region allocate_region(uint64_t& buffer_cursor, const std::vector<int64_t>& region_shape) {
    buffer_cursor = align_up(buffer_cursor, kWeightAlignment);
    uint64_t region_bytes = dtype_size(DType::BF16);

    for (int64_t dim_size : region_shape) {
        region_bytes *= static_cast<uint64_t>(dim_size);
    }

    Region region{buffer_cursor, region_bytes, region_shape};
    buffer_cursor += region_bytes;

    return region;
}

void push_whole_copy(std::vector<CopyOp>& copy_ops, const std::string& name,
                     const TensorEntry& info, const Region& dst_region) {
    copy_ops.push_back(CopyOp{.name = name,
                              .src = info.offset,
                              .dst = dst_region.offset,
                              .bytes = info.bytes,
                              .shape = info.shape});
}

void push_row_slice_copy(std::vector<CopyOp>& copy_ops, const std::string& name,
                         const TensorEntry& info, uint64_t dst_os, int64_t start, int64_t count) {
    constexpr uint64_t bf16_elem_bytes = dtype_size(DType::BF16);
    const int64_t cols = info.shape[1];
    const uint64_t row_bytes = static_cast<uint64_t>(cols) * bf16_elem_bytes;

    copy_ops.push_back(CopyOp{.name = name,
                              .src = info.offset + static_cast<uint64_t>(start) * row_bytes,
                              .dst = dst_os,
                              .bytes = static_cast<uint64_t>(count) * row_bytes,
                              .shape = {count, cols}});
}

void push_col_slice_copy(std::vector<CopyOp>& copy_ops, const std::string& name,
                         const TensorEntry& info, const Region& dst_region, int64_t start,
                         int64_t count) {
    constexpr uint64_t bf16_elem_bytes = dtype_size(DType::BF16);

    const int64_t rows = info.shape[0];
    const int64_t full_cols = info.shape[1];
    const uint64_t full_row_bytes = static_cast<uint64_t>(full_cols) * bf16_elem_bytes;
    const uint64_t slice_row_bytes = static_cast<uint64_t>(count) * bf16_elem_bytes;
    const uint64_t col_byte_os = static_cast<uint64_t>(start) * bf16_elem_bytes;

    for (int64_t r = 0; r < rows; ++r) {
        copy_ops.push_back(
            CopyOp{.name = name,
                   .src = info.offset + static_cast<uint64_t>(r) * full_row_bytes + col_byte_os,
                   .dst = dst_region.offset + static_cast<uint64_t>(r) * slice_row_bytes,
                   .bytes = slice_row_bytes,
                   .shape = {1, count}});
    }
}

}  // namespace

WeightLayout plan_shard_layout(const ModelConfig& config, const SafetensorsHeader& header, int rank,
                               int num_ranks) {
    if (num_ranks < 1) {
        throw std::invalid_argument("[plan_shard_layout] num_ranks must be >= 1");
    }

    if (rank < 0 || rank >= num_ranks) {
        throw std::invalid_argument("[plan_shard_layout] rank must be in [0, num_ranks)");
    }

    if (config.kv_heads <= 0 || (config.kv_heads % num_ranks) != 0) {
        throw std::invalid_argument(
            "[plan_shard_layout] config.kv_heads must be a multiple of num_ranks");
    }

    if (config.intermediate <= 0 || (config.intermediate % num_ranks) != 0) {
        throw std::invalid_argument(
            "[plan_shard_layout] config.intermediate must be a multiple of num_ranks");
    }

    if (config.heads <= 0 || (config.heads % num_ranks) != 0 ||
        (config.heads % config.kv_heads) != 0) {
        throw std::invalid_argument(
            "[plan_shard_layout] config.heads must be divisible by num_ranks and kv_heads");
    }

    if (config.tie_embeddings) {
        throw std::runtime_error("weights: tied embeddings are not supported");
    }

    std::unordered_set<std::string> seen;

    const int64_t vocab_size = config.vocab;
    const int64_t hidden_size = config.hidden;
    const int64_t head_dim = config.head_dim;

    const int64_t full_q_rows = static_cast<int64_t>(config.heads) * head_dim;
    const int64_t full_kv_rows = static_cast<int64_t>(config.kv_heads) * head_dim;
    const int64_t full_intermediate_rows = config.intermediate;

    const TensorEntry& embed =
        require_bf16_tensor(header, "model.embed_tokens.weight", {vocab_size, hidden_size}, seen);
    const TensorEntry& final_norm =
        require_bf16_tensor(header, "model.norm.weight", {hidden_size}, seen);
    const TensorEntry& lm_head =
        require_bf16_tensor(header, "lm_head.weight", {vocab_size, hidden_size}, seen);

    struct LayerTensorEntries {
        const TensorEntry* ln1 = nullptr;
        const TensorEntry* q_proj = nullptr;
        const TensorEntry* k_proj = nullptr;
        const TensorEntry* v_proj = nullptr;
        const TensorEntry* o_proj = nullptr;
        const TensorEntry* ln2 = nullptr;
        const TensorEntry* gate_proj = nullptr;
        const TensorEntry* up_proj = nullptr;
        const TensorEntry* down_proj = nullptr;
    };

    std::vector<LayerTensorEntries> all_layer_entries;
    all_layer_entries.reserve(config.layers);

    for (int layer = 0; layer < config.layers; ++layer) {
        const std::string prefix = "model.layers." + std::to_string(layer) + ".";

        LayerTensorEntries layer_entry{};

        layer_entry.ln1 =
            &require_bf16_tensor(header, prefix + "input_layernorm.weight", {hidden_size}, seen);

        layer_entry.q_proj = &require_bf16_tensor(header, prefix + "self_attn.q_proj.weight",
                                                  {full_q_rows, hidden_size}, seen);
        layer_entry.k_proj = &require_bf16_tensor(header, prefix + "self_attn.k_proj.weight",
                                                  {full_kv_rows, hidden_size}, seen);
        layer_entry.v_proj = &require_bf16_tensor(header, prefix + "self_attn.v_proj.weight",
                                                  {full_kv_rows, hidden_size}, seen);
        layer_entry.o_proj = &require_bf16_tensor(header, prefix + "self_attn.o_proj.weight",
                                                  {hidden_size, full_q_rows}, seen);

        layer_entry.ln2 = &require_bf16_tensor(header, prefix + "post_attention_layernorm.weight",
                                               {hidden_size}, seen);

        layer_entry.gate_proj = &require_bf16_tensor(header, prefix + "mlp.gate_proj.weight",
                                                     {full_intermediate_rows, hidden_size}, seen);
        layer_entry.up_proj = &require_bf16_tensor(header, prefix + "mlp.up_proj.weight",
                                                   {full_intermediate_rows, hidden_size}, seen);
        layer_entry.down_proj = &require_bf16_tensor(header, prefix + "mlp.down_proj.weight",
                                                     {hidden_size, full_intermediate_rows}, seen);

        all_layer_entries.push_back(layer_entry);
    }

    for (const auto& [name, _] : header.tensors) {
        if (!seen.contains(name)) {
            throw std::runtime_error("weights: unexpected tensor " + name);
        }
    }

    const int64_t shard_q_heads = config.heads / num_ranks;
    const int64_t shard_kv_heads = config.kv_heads / num_ranks;
    const int64_t shard_intermediate_units = config.intermediate / num_ranks;

    constexpr uint64_t bf16_elem_bytes = dtype_size(DType::BF16);

    const int64_t shard_q_rows = shard_q_heads * head_dim;
    const int64_t shard_kv_rows = shard_kv_heads * head_dim;
    const int64_t shard_wqkv_rows = (shard_q_heads + (2 * shard_kv_heads)) * head_dim;

    WeightLayout shard_layout;
    shard_layout.layers.resize(config.layers);
    uint64_t buffer_cursor = 0;

    shard_layout.embed = allocate_region(buffer_cursor, {vocab_size, hidden_size});
    push_whole_copy(shard_layout.copies, "model.embed_tokens.weight", embed, shard_layout.embed);

    for (int layer = 0; layer < config.layers; ++layer) {
        const std::string prefix = "model.layers." + std::to_string(layer) + ".";
        const LayerTensorEntries& entry = all_layer_entries[layer];
        LayerLayout& dst_layer = shard_layout.layers[layer];

        dst_layer.ln1 = allocate_region(buffer_cursor, {hidden_size});

        push_whole_copy(shard_layout.copies, prefix + "input_layernorm.weight", *entry.ln1,
                        dst_layer.ln1);

        dst_layer.wqkv = allocate_region(buffer_cursor, {shard_wqkv_rows, hidden_size});

        const uint64_t q_bytes =
            static_cast<uint64_t>(shard_q_rows * hidden_size) * bf16_elem_bytes;
        const uint64_t kv_bytes =
            static_cast<uint64_t>(shard_kv_rows * hidden_size) * bf16_elem_bytes;

        push_row_slice_copy(shard_layout.copies, prefix + "self_attn.q_proj.weight", *entry.q_proj,
                            dst_layer.wqkv.offset, rank * shard_q_rows, shard_q_rows);
        push_row_slice_copy(shard_layout.copies, prefix + "self_attn.k_proj.weight", *entry.k_proj,
                            dst_layer.wqkv.offset + q_bytes, rank * shard_kv_rows, shard_kv_rows);
        push_row_slice_copy(shard_layout.copies, prefix + "self_attn.v_proj.weight", *entry.v_proj,
                            dst_layer.wqkv.offset + q_bytes + kv_bytes, rank * shard_kv_rows,
                            shard_kv_rows);

        dst_layer.wo = allocate_region(buffer_cursor, {hidden_size, shard_q_rows});
        push_col_slice_copy(shard_layout.copies, prefix + "self_attn.o_proj.weight", *entry.o_proj,
                            dst_layer.wo, rank * shard_q_rows, shard_q_rows);

        dst_layer.ln2 = allocate_region(buffer_cursor, {hidden_size});
        push_whole_copy(shard_layout.copies, prefix + "post_attention_layernorm.weight", *entry.ln2,
                        dst_layer.ln2);

        dst_layer.wgu = allocate_region(buffer_cursor, {2 * shard_intermediate_units, hidden_size});
        const uint64_t gate_bytes =
            static_cast<uint64_t>(shard_intermediate_units * hidden_size) * bf16_elem_bytes;

        push_row_slice_copy(shard_layout.copies, prefix + "mlp.gate_proj.weight", *entry.gate_proj,
                            dst_layer.wgu.offset, rank * shard_intermediate_units,
                            shard_intermediate_units);
        push_row_slice_copy(shard_layout.copies, prefix + "mlp.up_proj.weight", *entry.up_proj,
                            dst_layer.wgu.offset + gate_bytes, rank * shard_intermediate_units,
                            shard_intermediate_units);

        dst_layer.wdown = allocate_region(buffer_cursor, {hidden_size, shard_intermediate_units});
        push_col_slice_copy(shard_layout.copies, prefix + "mlp.down_proj.weight", *entry.down_proj,
                            dst_layer.wdown, rank * shard_intermediate_units,
                            shard_intermediate_units);
    }

    shard_layout.final_norm = allocate_region(buffer_cursor, {hidden_size});
    push_whole_copy(shard_layout.copies, "model.norm.weight", final_norm, shard_layout.final_norm);

    shard_layout.lm_head = allocate_region(buffer_cursor, {vocab_size, hidden_size});
    push_whole_copy(shard_layout.copies, "lm_head.weight", lm_head, shard_layout.lm_head);

    std::sort(shard_layout.copies.begin(), shard_layout.copies.end(),
              [](const CopyOp& a, const CopyOp& b) { return a.src < b.src; });

    shard_layout.total_bytes = buffer_cursor;

    return shard_layout;
}

}  // namespace engine
