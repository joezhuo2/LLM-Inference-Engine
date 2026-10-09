#pragma once

#include "engine/core/stream.h"
#include "engine/core/tensor.h"

namespace engine::naive {

void swiglu(const Tensor& out, const Tensor& gate_up, Stream stream);

}  // namespace engine::naive
