#include <cstdio>
#include <cstdlib>

#include "engine/kernels/write_kv.h"

namespace engine {

/*
TODO(JOE): write_kv

Purpose. write_kv stores the keys and values of the tokens of one forward step into the paged KV cache of one layer, so the paged attention kernels of this and every later step can read them. It replaces naive::store_kv, which appends the rows of a single sequence to a contiguous cache: in a batched step the T tokens belong to different sequences, each sequence's tokens live in blocks scattered over the cache, and some rows of the batch are padding. Where each token goes is decided on the CPU by the block manager and arrives as slot_mapping; write_kv only moves data. The model runner calls it once per layer, after RoPE and before attention, with that layer's K and V views of the KvCache.

Interface. k_cache and v_cache are BF16 tensors of the same shape [num_blocks, kv_heads, block_size, head_dim], contiguous and row-major (the views KvCache::k(layer) and KvCache::v(layer) return). qkv is the BF16 [T, width] output of the fused QKV GEMM after RoPE; with kv_width = kv_heads * head_dim, the K values of a token are the columns width - 2 * kv_width to width - kv_width - 1 and the V values the last kv_width columns, so the Q columns in front of them are never read and no head count other than the cache's is needed (the same convention as naive::store_kv). Within the K slice, column h * head_dim + d is head h, dimension d, and likewise for V. slot_mapping is an I32 tensor [T] in device memory. stream is the stream to run on. A slot s names token position s % block_size of physical block s / block_size. For every token t with s = slot_mapping[t] >= 0, and every kv head h and dimension d, write_kv sets k_cache[s / block_size][h][s % block_size][d] to the K value of token t at head h, dimension d, and v_cache[s / block_size][h][s % block_size][d] to the V value. A slot of -1 marks a padding row: nothing is written for it. Nothing else in either cache changes, and qkv and slot_mapping are not modified.

Numerics. This is a copy: every written value is bit-identical to the BF16 value in qkv, with no conversion or rounding.

Errors and preconditions. Before launching anything, throw std::invalid_argument with a message naming write_kv when: k_cache or v_cache is not BF16 or not 4-dimensional, or their shapes differ; qkv is not BF16 or not 2-dimensional; qkv's width is not larger than 2 * kv_width (there must be at least one query column in front of K and V); slot_mapping is not I32, not 1-dimensional, or its length is not T. The caller guarantees, without checks (they would need a copy back to the host), that every slot is -1 or between 0 and num_blocks * block_size - 1, and that the non-negative slots of one call are pairwise distinct; the result is undefined otherwise. The launch is asynchronous on stream, with the usual KERNEL_CHECK after it.

Edge cases and boundaries. T = 0 is a valid call that does nothing and must not launch a kernel with an empty grid. Slots at the first and last position of a block (s % block_size equal to 0 and to block_size - 1), slot 0, the last slot of the cache and slots in block 0 (the null block is just memory to this kernel) are all written like any other. It must be correct for any positive block_size, kv_heads and head_dim, not only TinyLlama's 16, 4 and 64, and for T up to at least 2,048 (a full-length prefill). Indices into the cache must not overflow 32 bits for the cache sizes KvCache produces from free memory on an 8 GB card.

What to compare against. A host reference in the test that applies the definition above element by element to a copy of the caches.

Tests that prove it. tests/todo/gpu/test_write_kv.cpp: a sequence of 37 tokens through a shuffled block table on TinyLlama's layout, padding rows, the boundary slots, block sizes 1 and 5 with an odd head layout, a 2,048-token prefill, a 16-sequence decode batch, T = 0, guard bytes around both caches proving nothing outside them is written, qkv and slot_mapping unchanged, and every rejected argument. The caches are filled with random values first and compared in full, so a write to a wrong place fails even when the right place is also written. They run in the todo_gpu_tests target with the todo_joe label (ctest --preset debug -L todo_joe). When they all pass, move the file to tests/gpu/, add it to gpu_tests and delete todo_gpu_tests if it is empty; also run compute-sanitizer --tool memcheck on them.
*/
void write_kv(const Tensor&, const Tensor&, const Tensor&, const Tensor&, Stream) {
    std::fprintf(stderr, "unimplemented: write_kv\n");
    std::abort();
}

}  // namespace engine
