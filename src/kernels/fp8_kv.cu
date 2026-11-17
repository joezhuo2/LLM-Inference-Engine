#include <stdexcept>

#include "engine/kernels/fp8_kv.h"

namespace engine {

/*
TODO(JOE): an FP8 KV cache (write_kv_fp8, paged_attention_decode_fp8)

Purpose. Issue #126. The paged cache stores every key and value as BF16, 2 bytes per element. Stored as FP8 E4M3, 1 byte per element, the same VRAM holds twice the blocks, so twice the cached tokens. It will not make attention faster yet (attention is far from its bandwidth roof at batch 16, see #94), so this is about capacity first. These two functions are the write and the read of that cache: write_kv_fp8 rounds the new keys and values of a step into the cache, and paged_attention_decode_fp8 is decode attention that reads them back. Nothing else in the engine changes until the wiring branch.

Words. E4M3 is one sign bit, four exponent bits and three mantissa bits. It has 8 values per power of two from 2^-6 to 2^8, a few smaller ones down to 2^-9, the largest finite value 448, signed zeros, one NaN pattern per sign (0x7F and 0xFF) and no infinities. A scale s is a positive FP32 number that says what the byte means: the stored byte is e4m3(x / s) and the value read back is float(byte) * s. One scale for all keys of the call and one for all values. With s = 1 the cache holds values up to 448, far above what TinyLlama's keys and values reach, and the wiring branch decides which scales to pass.

Interface. Declared in include/engine/kernels/fp8_kv.h. Both functions take the same tensors as write_kv and paged_attention_decode with two differences: k_cache and v_cache are DType::F8E4M3 [num_blocks, kv_heads, block_size, head_dim], one byte per element, and k_scale and v_scale follow the tensors. qkv is still BF16 [T, (heads + 2 * kv_heads) * head_dim] with the query, key and value columns in that order, slot_mapping is I32 [T] with -1 for a row that writes nothing, out is BF16 [B, heads * head_dim], block_tables is I32 [B, W] and context_lens is I32 [B] with 0 for a padding row. Everything runs on stream and neither function waits for it.

Semantics of the write. For every token t with slot s = slot_mapping[t] >= 0, every KV head h and every d: k_cache[s / block_size][h][s % block_size][d] = e4m3(float(K[t][h][d]) / k_scale), and the same for V with v_scale. The division is one FP32 division, then one rounding to E4M3: round to nearest, ties to the even mantissa, and anything whose magnitude rounds above 448 becomes 448 with its sign (so do the infinities). NaN stays NaN and negative zero stays negative zero. That is exactly __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3) from cuda_fp8.h, which also runs on the device. No other byte of either cache may change.

Semantics of the read. Key j of row r, KV head h, is K_j[d] = float(k_cache byte) * k_scale, and V_j likewise with v_scale, both FP32 (cuda_fp8.h converts a byte to float exactly). From there on it is paged_attention_decode: for query head q of row r, s_j = bf16(bf16(Q_q . K_j) * 1 / sqrt(head_dim)) over the context_lens[r] tokens that block_tables[r] reaches, softmax over j, and out = bf16(sum_j p_j * V_j), with the dot product, the maximum, the sum and the output accumulated in FP32. The order of the additions is yours. A row with context_lens[r] = 0 writes nothing. Position j of row r is in block block_tables[r][j / block_size] at offset j % block_size; table entries past the last used block can hold anything, including -1.

Numerics. The tests round nothing differently from you only when the scales are powers of two: then float(byte) * scale is exact and is also a BF16 value, so the result must match naive::attention on those values within the tolerance the other paged kernels are held to, 1/64 of (sum_j p_j |V_j[d]| + |naive output|). For scales that are not powers of two the tests compare against a double precision softmax on float(byte) * scale with the same tolerance. A context of one token has p = 1 and must return bf16(V) exactly.

Determinism. The output of a row depends only on its own query, its context length, the table entries it uses and the cache bytes they reach. It is bit-identical whether the row runs alone or in a batch, and two identical calls give identical bytes. NaN bytes, saturated bytes or garbage in cells no row reaches, and NaN in the K and V columns of qkv (the read ignores them: the step's own key and value are already in the cache), must not reach the output.

Errors and preconditions. Before launching anything, throw std::invalid_argument with a message that names the function when: k_cache or v_cache is not F8E4M3 (a BF16 cache must be rejected, not reinterpreted) or not 4-dimensional, or their shapes differ; qkv or out is not BF16 or not 2-dimensional; the width of qkv is not above 2 * kv_heads * head_dim or the rest is not a multiple of head_dim, or the number of heads is not a multiple of kv_heads; the rows of out, qkv, block_tables and context_lens differ, or slot_mapping does not have one entry per row of qkv; slot_mapping, block_tables or context_lens is not I32 or has the wrong number of dimensions; block_tables has no columns; k_scale or v_scale is not finite or not above 0. Slots, table entries and context lengths on the device are the caller's promise and are not checked.

Edge cases. T = 0 and B = 0 are valid and launch nothing (their tensors may have null data). Rows of -1 slots only. block_size 1 and head_dim that is not a multiple of 16. Values far above 448 * scale (they saturate, they do not wrap or turn into NaN). Contexts of 1 token, around block boundaries, 2,048 and 8,192 tokens. A batch of 16 rows of unequal length.

What to compare against. For the write, the bytes the host computes with the same cuda_fp8.h call (tests/support/fp8.h), compared byte for byte over the whole cache with guard bytes around it, including a test that sends all 65,536 BF16 bit patterns through. For the read, naive::attention and the double precision reference above. How much accuracy FP8 costs against Hugging Face is not a property of these two functions and is measured by the wiring branch.

What is not here. The wiring is a separate Part B branch once the tests pass: KvCache gets a dtype, kv_cache_blocks counts 1 byte per element, PagedRunner picks the FP8 pair behind a flag on engine generate, serve and bench, the scales are chosen (start with 1 and look at the largest |K| and |V| a real run produces), and results.md gets blocks that fit, attention time and the logit error against the reference dump with and without FP8. The prefill path (flash_attention_prefill reads the cache too) and the split and grouped decode kernels need their own FP8 readers after that. A scale per layer or per head, and E5M2, are not part of this stub.

Tests that prove it. tests/todo/gpu/test_fp8_kv.cpp (target todo_gpu_tests, labels todo_joe and gpu), WriteKvFp8.* and PagedAttentionDecodeFp8.*, checked against a throwaway implementation and mutants of it. Run them with ctest --preset debug -L todo_joe -R Fp8. When all pass, move the file to tests/gpu/ and add it to gpu_tests.
*/
void write_kv_fp8(const Tensor&, const Tensor&, const Tensor&, const Tensor&, float, float,
                  Stream) {
    throw std::logic_error("unimplemented: write_kv_fp8");
}

void paged_attention_decode_fp8(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                const Tensor&, const Tensor&, float, float, Stream) {
    throw std::logic_error("unimplemented: paged_attention_decode_fp8");
}

}  // namespace engine
