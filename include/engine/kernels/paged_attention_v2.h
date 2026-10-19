#pragma once

#include "engine/core/stream.h"
#include "engine/core/tensor.h"

namespace engine {

void paged_attention_decode_v2(const Tensor& out, const Tensor& qkv, const Tensor& k_cache,
                               const Tensor& v_cache, const Tensor& block_tables,
                               const Tensor& context_lens, Stream stream);

}  // namespace engine
