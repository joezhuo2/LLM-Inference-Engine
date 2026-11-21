#pragma once

#include "engine/core/stream.h"
#include "engine/core/tensor.h"

namespace engine {

void write_kv_fp8(const Tensor& k_cache, const Tensor& v_cache, const Tensor& qkv,
                  const Tensor& slot_mapping, float k_scale, float v_scale, Stream stream);

void paged_attention_decode_fp8(const Tensor& out, const Tensor& qkv, const Tensor& k_cache,
                                const Tensor& v_cache, const Tensor& block_tables,
                                const Tensor& context_lens, float k_scale, float v_scale,
                                Stream stream);

}  // namespace engine
