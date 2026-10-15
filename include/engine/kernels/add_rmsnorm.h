#pragma once

#include "engine/core/stream.h"
#include "engine/core/tensor.h"

namespace engine::fast {

void add_rmsnorm(const Tensor& sum, const Tensor& norm, const Tensor& x, const Tensor& residual,
                 const Tensor& weight, float eps, Stream stream);

}  // namespace engine::fast
