#include <cstdio>
#include <cstdlib>

#include "engine/kernels/rope.h"

namespace engine::fast {

/*
TODO(JOE): rope (fast)

Purpose. fast::rope is the optimized replacement of naive::rope (src/kernels/rope.cu) in the decode and prefill path of PagedRunner. RoPE rotates the Q and K slices of every row of the fused QKV GEMM output in place, by an angle that depends on the token's position; it is pure elementwise work with a transcendental per element, so on a memory-bound decode step its cost is the launch plus the uncoalesced 2-byte accesses of the naive loop (one thread per pair, two scattered BF16 loads and stores) and the per-pair cosf and sinf. The fast kernel must do the same rotation with 128-bit accesses and without recomputing what can be shared, and give exactly the same bits.

Interface. Same as naive::rope. qkv is BF16 [T, (heads + 2 * kv_heads) * head_dim] and is modified in place: within each row, the first heads * head_dim columns are the query heads and the next kv_heads * head_dim the key heads; the last kv_heads * head_dim columns (V) are not touched. positions is I32 [T] in device memory, one position per row. head_dim is derived as width / (heads + 2 * kv_heads) and must be even. theta is the RoPE base (10000 for TinyLlama). stream is the stream to run on. The rotation is HF rotate_half style: for head h and index i in [0, head_dim / 2), with angle = position * theta^(-2i / head_dim), x[i] and x[i + head_dim / 2] of the head become x[i] * cos - x[i + half] * sin and x[i + half] * cos + x[i] * sin. Everything outside the Q and K columns of qkv, and positions, is unchanged.

Numerics. Bit-identical to naive::rope for every input, which is the only reference: cos and sin are computed with the accurate cosf and sinf in FP32 from the angle position * inv_freq[i] (inv_freq[i] = 1 / pow(theta, 2i / head_dim) as the naive launcher computes it on the host), each rounded to BF16; every product (lo * cos, hi * cos, lo * sin, -hi * sin) is rounded to BF16 on its own and the two terms of each output are added in FP32 and rounded once more. The fast-math intrinsics (__sinf, __cosf, --use_fast_math) change the bits and are not allowed; if you use sincosf, the tests must still show bit identity. This rounding sequence is what the HF reference does, so any shortcut that fuses a multiply and an add (fmaf) is a mismatch.

Errors and preconditions. Before launching anything, throw std::invalid_argument with a message naming rope when: heads or kv_heads is not positive; qkv is not BF16 or not 2-dimensional; positions is not I32, not 1-dimensional or its length is not T; the width is not a multiple of heads + 2 * kv_heads, or the head_dim it gives is odd or larger than 256. Positions are the caller's responsibility (non-negative and below the model's max positions). The launch is asynchronous on stream with the usual KERNEL_CHECK after it.

Edge cases and boundaries. T = 0 is a valid call that does nothing (no launch of an empty grid; the data pointers may be null). head_dim is any even value from 2 to 256, not only 64: 128-bit accesses need head_dim / 2 to be a multiple of 8, so the other cases (2, 6, 10, 50 and so on) need a scalar path, and the base pointer of qkv and the row stride (width * 2 bytes) are not always multiples of 16 (the tests offset the pointer by one element). Rows are independent, so the result of a row must not depend on T or on the row's place in the batch. Positions up to 2047 and an arbitrary order of positions within a batch (a decode batch is 16 unrelated sequences) must work, and the same position in two rows must give identical rotations. When the number of query heads differs from the key heads (GQA, 32 and 4 in TinyLlama) both are rotated with the same angles.

What to compare against. naive::rope on the same input (it is itself checked against HF on all 22 layers), and an untouched-V check in the test.

Tests that prove it. tests/todo/gpu/test_rope_fast.cpp: bit identity with naive::rope on TinyLlama's layout (32, 4, 64) at T = 1, 16 and 2048 with consecutive and random positions, other head layouts (heads, kv_heads, head_dim) including head_dim 2, 6, 50, 128 and 256 and one kv head, theta 10000 and 500000, an unaligned base pointer, guard bytes around qkv, V columns and positions unchanged, determinism, T = 0 and every rejected argument. They run in the todo_gpu_tests target with the todo_joe label (ctest --preset debug -L todo_joe). When they all pass, move the file to tests/gpu/, add it to gpu_tests and delete todo_gpu_tests if it is empty; also run compute-sanitizer --tool memcheck on them and ncu before and after.
*/
void rope(const Tensor&, const Tensor&, int, int, float, Stream) {
    std::fprintf(stderr, "unimplemented: fast::rope\n");
    std::abort();
}

}  // namespace engine::fast
