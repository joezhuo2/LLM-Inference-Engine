#include <stdexcept>

#include "engine/kernels/paged_attention_grouped.h"

namespace engine {

/*
TODO(JOE): decode attention with one block per sequence and KV head (paged_attention_decode_grouped)

Purpose. Issue #94. At batch 16 attention is about half the step and runs at 7% of its bandwidth roof (results.md, Batch size past 16). paged_attention_decode_v2 launches one block per row and query head, 32 per row for TinyLlama. The 8 query heads of a group share one KV head, so 8 blocks each load the same keys and values from DRAM, the key loads are scalar 2-byte reads, and every 256-token chunk runs two tree reductions through shared memory with 8 __syncthreads each. This kernel launches one block per (row, KV head). The block loads each key and value of its head once and serves all the query heads of the group from that one copy. It is a decode kernel: one query position per row, like v2, and it must give the same answers as v2 on every input both accept.

Before you write it. Nobody has measured how much each of the three costs above is worth. Profile v2 at batch 16 with Nsight Compute first (profiling.md has the recipe and scripts/paged_attention_bench.cu is the benchmark) and write down: DRAM bytes per launch against the bytes of K and V the rows reach once, the share of time in the reductions, and occupancy. Then you know which of the ideas below pays, and the profile of your kernel has something to be compared with.

Interface. Declared in include/engine/kernels/paged_attention_grouped.h. The arguments are those of paged_attention_decode_v2, with the same meaning: out BF16 [B, heads * head_dim], qkv BF16 [B, (heads + 2 * kv_heads) * head_dim] (query, key and value columns in that order; only the query columns are read, the step's own key and value are already in the cache), k_cache and v_cache BF16 [num_blocks, kv_heads, block_size, head_dim], block_tables I32 [B, W], context_lens I32 [B] with 0 for a padding row. It runs on stream and does not wait for it. No scratch tensor: if you need one, change the signature and tell Claude, the tests take the kernel as a std::function.

Semantics. For row r and query head h with KV head g = h / (heads / kv_heads): s_j = bf16(bf16(Q_h . K_j) * 1 / sqrt(head_dim)) for the context_lens[r] tokens j that block_tables[r] reaches, p = softmax(s), out[r][h] = bf16(sum_j p_j * V_j). Position j is in block block_tables[r][j / block_size] at offset j % block_size. A row with context_lens[r] = 0 writes nothing. Table entries past the last used block can hold anything, including -1.

Ideas from the issue, none of them required by the tests. (1) Q K^T through mma.sync: the group's queries are the rows of one operand ([8, 64] for TinyLlama, padded to 16 rows) and a tile of 8 or 16 keys is the other, with FP32 accumulators, the way flash_prefill.cu does it. (2) 16-byte loads of K and V, which the cache layout allows because head_dim is a multiple of 16 here and a token's head is contiguous. (3) The running maximum and sum per query head kept in registers and combined with warp shuffles in place of shared-memory trees. (4) The value sum for all query heads of the group from one load of each V tile, which is a second matrix product, P V. CI compiles for sm_120, sm_89 and sm_75, and BF16 mma needs sm_80: keep a path that compiles on sm_75, like the ENGINE_FLASH_EMULATE_MMA switch in flash_prefill.cu.

Numerics. Round each score exactly as the naive kernel, v1 and v2 do: the dot product accumulated in FP32 (mma with FP32 accumulators does that; the order inside it is the hardware's), rounded to BF16, multiplied by the scale in FP32, rounded to BF16 again. These two roundings are what makes the engine match Hugging Face, which computes in BF16, and they are why a tensor-core version is acceptable at all: the tests hold it to the oracle. Keep the running maximum, the running sum and the output accumulator in FP32 and round the output once. The tests allow 1/64 of (sum_j p_j |V_j[d]| + |naive output|) per element against naive::attention, and against Hugging Face's attn output of every traced layer the bound the other paged kernels meet (outside_value_scale in the harness).

Determinism. The output of a row depends only on its own query, its context length, the table entries it uses and the cache cells they reach. It is bit-identical whether the row runs alone or in a batch of 16 and two identical calls give identical bytes, so the order of the additions may depend on the context length but not on the rest of the batch or on timing. NaN or huge values in cells no row reaches, and in the K and V columns of qkv, must not reach the output: a tile that runs past the context must mask the extra keys before the maximum, not multiply them by zero afterwards.

Errors and preconditions. Before launching anything, throw std::invalid_argument with a message starting with [paged_attention_decode_grouped] for everything v2 rejects (dtypes, numbers of dimensions, cache shapes that differ, a qkv width that does not match heads and kv_heads, heads not a multiple of kv_heads, rows that differ between out, qkv, block_tables and context_lens, a block table with no columns), and also when head_dim is not a multiple of 16 or is above 128. Table entries and context lengths on the device are the caller's promise.

Edge cases. B = 0 launches nothing (its tensors may have null data). Contexts of 1 token, around block boundaries, around your tile size (the tests use 7 to 9, 16, 17, 64, 127 to 129), 2,048 and 8,192. A batch where every row is a padding row. Groups of 1 to 12 query heads per KV head, head_dim 16, 32, 48, 64, 96 and 128, block_size 1 to 32. Scores large enough that exp of the unshifted score would overflow. One long row beside 15 short ones: a block per (row, KV head) means 4 blocks do all the work of the long row, so decide whether the split kernel still takes that case (see What is not here).

What to compare against. naive::attention on a contiguous copy of each row's context, and the reference dump. For speed: v2 at batch 16 in the same Nsight Compute session, time per launch and DRAM bytes, with contexts of 128, 512 and 2,048, and the batch sweep of results.md (Batch size past 16) rerun once the kernel is wired in.

What is not here. Wiring is a separate Part B branch once the tests pass: PagedRunner picks this kernel from the batch size and the longest context the way it picks between v2 and split today, the graphs are captured again, and profiling.md and results.md get the numbers. Splitting a long context across blocks (what paged_attention_decode_split does for batch 1) is not part of this stub. An FP8 cache reader (#126) comes after both.

Tests that prove it. tests/todo/gpu/test_paged_attention_grouped.cpp (target todo_gpu_tests, labels todo_joe and gpu), PagedAttentionDecodeGrouped.* and LayoutsGrouped: the tests v2 passes, with layouts whose head_dim is a multiple of 16, the two head_dim rejections, and 16 rows of one length around tile edges. Run them with ctest --preset debug -L todo_joe -R Grouped. When all pass, move the file to tests/gpu/ and add it to gpu_tests.
*/
void paged_attention_decode_grouped(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                    const Tensor&, const Tensor&, Stream) {
    throw std::logic_error("unimplemented: paged_attention_decode_grouped");
}

}  // namespace engine
