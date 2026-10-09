#pragma once

#include "engine/core/stream.h"
#include "engine/core/tensor.h"

namespace engine::naive {

void add(const Tensor& out, const Tensor& a, const Tensor& b, Stream stream);

}  // namespace engine::naive
