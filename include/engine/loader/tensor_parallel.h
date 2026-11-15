#pragma once

#include "engine/loader/safetensors.h"
#include "engine/loader/weight_layout.h"
#include "engine/model/config.h"

namespace engine {

ModelConfig shard_config(const ModelConfig& config, int num_ranks);

WeightLayout plan_shard_layout(const ModelConfig& config, const SafetensorsHeader& header, int rank,
                               int num_ranks);

}  // namespace engine
