#pragma once

#include "engine/core/stream.h"
#include "engine/core/tensor.h"

namespace engine::naive {

void embedding(const Tensor& out, const Tensor& ids, const Tensor& table, Stream stream);

}  // namespace engine::naive
