#pragma once

#include "engine/core/stream.h"
#include "engine/core/tensor.h"

namespace engine {

void flash_attention_prefill(const Tensor& out, const Tensor& qkv, const Tensor& k_cache,
                             const Tensor& v_cache, const Tensor& block_tables,
                             const Tensor& chunk_starts, const Tensor& cached_lens, Stream stream);

}  // namespace engine
