#pragma once

#include "engine/core/stream.h"
#include "engine/core/tensor.h"

namespace engine::naive {

void rope(const Tensor& qkv, const Tensor& positions, int heads, int kv_heads, float theta,
          Stream stream);

}  // namespace engine::naive
