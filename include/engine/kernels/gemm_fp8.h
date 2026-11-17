#pragma once

#include "engine/core/stream.h"
#include "engine/core/tensor.h"

namespace engine {

void gemm_fp8(const Tensor& y, const Tensor& x, const Tensor& w, const Tensor& row_scale,
              Stream stream);

}  // namespace engine
