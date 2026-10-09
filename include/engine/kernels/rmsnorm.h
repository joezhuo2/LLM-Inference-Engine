#pragma once

#include "engine/core/stream.h"
#include "engine/core/tensor.h"

namespace engine::naive {

void rmsnorm(const Tensor& out, const Tensor& x, const Tensor& weight, float eps, Stream stream);

}  // namespace engine::naive
