#pragma once

#include <vector>

#include "engine/core/tensor.h"
#include "engine/loader/weight_layout.h"

namespace engine {

struct LayerWeights {
    Tensor ln1, wqkv, wo, ln2, wgu, wdown;
};

struct ModelWeights {
    Tensor embed, final_norm, lm_head;
    std::vector<LayerWeights> layers;
};

ModelWeights bind_weights(const WeightLayout& layout, void* base);

}  // namespace engine
