#pragma once

#include <cstdint>

#include "engine/core/stream.h"
#include "engine/core/tensor.h"

namespace engine::naive {

void store_kv(const Tensor& k_cache, const Tensor& v_cache, const Tensor& qkv, int64_t start,
              Stream stream);

void attention(const Tensor& out, const Tensor& qkv, const Tensor& k_cache, const Tensor& v_cache,
               int64_t start, int heads, int kv_heads, Stream stream);

}  // namespace engine::naive
