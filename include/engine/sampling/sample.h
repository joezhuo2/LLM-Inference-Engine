#pragma once

#include <cstdint>

#include "engine/core/stream.h"
#include "engine/core/tensor.h"

namespace engine {

void sample(const Tensor& ids, const Tensor& logits, float temperature, uint64_t seed,
            uint64_t step, Stream stream);

}  // namespace engine
