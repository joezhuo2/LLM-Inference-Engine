#pragma once

#include <cstdint>

#include "engine/core/stream.h"
#include "engine/core/tensor.h"
#include "engine/sampling/params.h"

namespace engine {

struct SampleRow {
    SamplingParams params;
    uint64_t step = 0;
};

void sample(const Tensor& ids, const Tensor& logits, const SampleRow* rows, Stream stream);

}  // namespace engine
