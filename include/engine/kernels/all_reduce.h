#pragma once

#include <span>

#include "engine/core/stream.h"
#include "engine/core/tensor.h"

namespace engine {

inline constexpr int kMaxRanks = 8;

void all_reduce_add(const Tensor& x, std::span<const Tensor> partials, Stream stream);

}  // namespace engine
