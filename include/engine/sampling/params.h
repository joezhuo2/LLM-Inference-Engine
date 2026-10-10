#pragma once

#include <cstdint>

namespace engine {

struct SamplingParams {
    float temperature = 0.0f;
    uint64_t seed = 0;
};

void validate(const SamplingParams& params);

}  // namespace engine
