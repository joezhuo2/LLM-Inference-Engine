#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "engine/loader/safetensors.h"
#include "engine/model/config.h"

namespace engine {

inline constexpr uint64_t kWeightAlignment = 256;

struct Region {
    uint64_t offset = 0;
    uint64_t bytes = 0;
    std::vector<int64_t> shape;
};

struct LayerLayout {
    Region ln1, wqkv, wo, ln2, wgu, wdown;
};

struct CopyOp {
    std::string name;
    uint64_t src = 0;
    uint64_t dst = 0;
    uint64_t bytes = 0;
    std::vector<int64_t> shape;
};

struct WeightLayout {
    Region embed, final_norm, lm_head;
    std::vector<LayerLayout> layers;
    std::vector<CopyOp> copies;
    uint64_t total_bytes = 0;
};

WeightLayout plan_weight_layout(const ModelConfig& config, const SafetensorsHeader& header);

}  // namespace engine
