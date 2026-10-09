#pragma once

#include "engine/core/stream.h"
#include "engine/core/tensor.h"

namespace engine {

void greedy(const Tensor& ids, const Tensor& logits, Stream stream);

}  // namespace engine
