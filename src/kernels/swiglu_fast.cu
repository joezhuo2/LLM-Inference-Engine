#include <cstdio>
#include <cstdlib>

#include "engine/kernels/swiglu.h"

namespace engine::fast {

/*
TODO(JOE): swiglu (fast)

Purpose. fast::swiglu is the optimized replacement of naive::swiglu (src/kernels/swiglu.cu) between the gate/up GEMM and the down projection in every MLP of PagedRunner. The activation of one token is silu(gate) * up over ff = 5632 values, read from the two halves of one row of the fused gate_up GEMM output. At decode the arithmetic is negligible and the whole kernel is bandwidth and launch bound: the naive version issues one thread per output element with two 2-byte loads and a 2-byte store each, which wastes most of every memory transaction. The fast kernel must move the same bytes with 128-bit accesses, and compute the same bits.

Interface. Same as naive::swiglu. gate_up is BF16 [T, 2 * ff], row-major: columns 0 to ff - 1 of a row are the gate values and columns ff to 2 * ff - 1 the up values (the fused wgu layout of the weights). out is BF16 [T, ff]. stream is the stream to run on. For every row t and column j, out[t][j] = silu(gate_up[t][j]) * gate_up[t][ff + j] with the rounding described below. gate_up is not modified, and out must not overlap gate_up.

Numerics. Bit-identical to naive::swiglu for every input, including large magnitudes and special values: g and u are the BF16 inputs converted to float, silu is g / (1 + expf(-g)) with the accurate expf (not __expf, not --use_fast_math, no reciprocal approximation) rounded to BF16 on its own, and the result is float(silu_bf16) * u rounded to BF16 once. This is the sequence HF performs (silu on a BF16 tensor, then a BF16 multiply), which is why the naive kernel matches HF on all 22 layers with zero mismatches. For g very negative expf(-g) overflows to infinity and silu becomes -0 (not NaN); for g = +inf silu is +inf; a NaN input gives a NaN output (only NaN-ness is compared, not the payload). The tests check all of this.

Errors and preconditions. Before launching anything, throw std::invalid_argument with a message naming swiglu when: gate_up is not BF16 or not 2-dimensional, or its width is odd; out is not BF16, not 2-dimensional, does not have T rows or its width is not half of gate_up's. The launch is asynchronous on stream with the usual KERNEL_CHECK after it.

Edge cases and boundaries. T = 0 is a valid call that does nothing (the data pointers may be null, no launch of an empty grid). ff is any positive integer, 1 included, not only a multiple of 8: with 128-bit accesses you need a scalar path for ff values that are not multiples of 8, and note that the up half of a row starts at byte offset ff * 2 within a row that starts at 2 * 2 * ff * t, so the alignment of the up values and of the next row depends on ff. The base pointers of gate_up and out are not always 16-byte aligned either (the tests offset them by one element). Index arithmetic must not overflow 32 bits (T * 2 * ff up to at least 2,048 * 2 * 5632 is within int32 but a large batch times a large ff is not; use 64-bit indices). Every row is independent and the result must not depend on T or the row position, and runs must be bit-identical.

What to compare against. naive::swiglu on the same input, a host reference for the special values, and the HF activations through the naive kernel.

Tests that prove it. tests/todo/gpu/test_swiglu_fast.cpp: bit identity with naive::swiglu on TinyLlama's ff = 5632 at T = 1, 16 and 64, on odd and small ff values (1, 3, 7, 8, 9, 100, 5631), on an unaligned base pointer for each buffer, on inputs with magnitudes up to 100 and on special values (negative overflow, infinities, NaN, signed zeros), guard bytes around out, gate_up unchanged, determinism, T = 0 and every rejected argument. They run in the todo_gpu_tests target with the todo_joe label (ctest --preset debug -L todo_joe). When they all pass, move the file to tests/gpu/, add it to gpu_tests and delete todo_gpu_tests if it is empty; also run compute-sanitizer --tool memcheck on them and ncu before and after.
*/
void swiglu(const Tensor&, const Tensor&, Stream) {
    std::fprintf(stderr, "unimplemented: fast::swiglu\n");
    std::abort();
}

}  // namespace engine::fast
