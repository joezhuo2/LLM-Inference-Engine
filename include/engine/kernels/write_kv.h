#pragma once

#include "engine/core/stream.h"
#include "engine/core/tensor.h"

namespace engine {

void write_kv(const Tensor& k_cache, const Tensor& v_cache, const Tensor& qkv,
              const Tensor& slot_mapping, Stream stream);

}  // namespace engine
