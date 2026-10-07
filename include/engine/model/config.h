#pragma once

#include <filesystem>
#include <string_view>

namespace engine {

struct ModelConfig {
    int hidden = 0;
    int layers = 0;
    int heads = 0;
    int kv_heads = 0;
    int head_dim = 0;
    int intermediate = 0;
    int vocab = 0;
    int max_positions = 0;
    double rope_theta = 0.0;
    double rms_eps = 0.0;
    int bos_id = 0;
    int eos_id = 0;
    bool tie_embeddings = false;

    int kv_dim() const { return kv_heads * head_dim; }
    int qkv_dim() const { return hidden + 2 * kv_dim(); }
    int group_size() const { return heads / kv_heads; }
};

ModelConfig parse_config(std::string_view json);
ModelConfig load_config(const std::filesystem::path& path);

}  // namespace engine
