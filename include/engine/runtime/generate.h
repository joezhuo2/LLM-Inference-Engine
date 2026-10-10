#pragma once

#include <cstdint>
#include <functional>
#include <span>
#include <vector>

#include "engine/runtime/model_runner.h"

namespace engine {

using TokenCallback = std::function<void(int32_t token)>;

struct SamplingParams {
    float temperature = 0.0f;
    uint64_t seed = 0;
};

std::vector<int32_t> generate(ModelRunner& runner, std::span<const int32_t> prompt,
                              int max_new_tokens, int32_t eos_id,
                              const SamplingParams& sampling = {},
                              const TokenCallback& on_token = {});

}  // namespace engine
