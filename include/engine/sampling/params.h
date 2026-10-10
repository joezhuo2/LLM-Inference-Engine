#pragma once

#include <cstdint>

namespace engine {

struct SamplingParams {
    float temperature = 0.0f;
    uint64_t seed = 0;
    int32_t top_k = 0;
    float top_p = 1.0f;
};

void validate(const SamplingParams& params);

}  // namespace engine
