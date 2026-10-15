#include <cstdio>
#include <cstdlib>

#include "engine/kernels/add_rmsnorm.h"

namespace engine::fast {

/*
TODO(JOE): add_rmsnorm

Purpose. add_rmsnorm fuses the residual add and the RMSNorm that follows it in every decoder layer. Today PagedRunner runs naive::add (x = x + proj) and then naive::rmsnorm (norm = rmsnorm(x)) as two kernels, so the new hidden state is written to global memory by the first kernel and read back by the second. A decode step at batch 1 is memory bound (it streams the 2.2 GB of weights, and the activations are 4 KB per row), but the two extra launches and the round trip sit on the critical path of 22 layers times 2 norms, and they are what the nsys timeline shows as gaps once CUDA Graphs remove the launch latency. The fused kernel reads x and residual once, writes the sum and the normalized row once, and is one launch instead of two. The model runner will call it twice per layer (after the attention output projection with ln2, after the MLP down projection with the next layer's ln1 or the final norm), so its output must be usable as the next residual stream.

Interface. All tensors are BF16 and 2-dimensional [T, hidden] except weight, which is BF16 [hidden]. sum, norm, x and residual have the same shape. x is the residual stream, residual is the branch output being added to it (proj in the runner), weight is the RMSNorm scale of the norm that follows, eps its epsilon. stream is the stream to run on. For every row t: sum[t] = x[t] + residual[t] elementwise, and norm[t] = rmsnorm(sum[t]) with the definition of naive::rmsnorm (see src/kernels/rmsnorm.cu): rstd = rsqrt(mean(sum[t]^2) + eps) with the mean over the BF16-rounded sum values accumulated in FP32, and norm[t][i] = bf16(weight[i] * bf16(sum[t][i] * rstd)). Both outputs are fully written; x, residual and weight are not modified, except that sum may be the very same tensor (same data pointer) as x or as residual, which is how the runner uses it (sum = x). norm must not overlap sum, x or residual.

Numerics. sum is bit-identical to naive::add: float(x) + float(residual) rounded once to BF16. The norm is computed from the rounded sum, not from the unrounded FP32 sum, exactly as the unfused pair does, so fusing changes no rounding step. The only permitted difference from naive::rmsnorm is the summation order of the FP32 sum of squares (at most 2 BF16 ulps on at most 0.1% of the elements, the close_up_to_reduction_order criterion the naive kernel itself is tested with). A given row must produce bit-identical output whatever T is and wherever the row sits in the batch (batch invariance: a sequence decoded alone and in a batch of 16 must not differ because of this kernel), and two runs on the same input must be bit-identical (no atomics in the reduction).

Errors and preconditions. Before launching anything, throw std::invalid_argument with a message naming add_rmsnorm when: any of sum, norm, x, residual or weight is not BF16; x is not 2-dimensional; residual, sum or norm does not have exactly x's shape; weight is not 1-dimensional of length hidden. Overlap other than the exact aliasing of sum with x or with residual is the caller's bug and is not checked. The launch is asynchronous on stream with the usual KERNEL_CHECK after it.

Edge cases and boundaries. T = 0 is a valid call that does nothing, with no launch of an empty grid, and the tensors' data pointers may be null. hidden is any positive integer up to at least 8192 (1, 2, 100, 2047 and 2049 included, not only multiples of 8): if you load 128 bits at a time you need a scalar path for the tail and for rows whose address is not 16-byte aligned, and the tests pass pointers offset by one element so the base pointer itself is not aligned. Rows of different T must not influence one another. An all-zero row gives sum 0 and norm 0 (no NaN from the normalization). The output must not depend on the contents of memory outside the tensors.

What to compare against. A host reference in the test (round the sum to BF16, then the double-accumulated RMSNorm of the naive test), and the unfused naive::add followed by naive::rmsnorm on the GPU.

Tests that prove it. tests/todo/gpu/test_add_rmsnorm.cpp: TinyLlama's hidden size 2048 at T = 1, 16 and 64, 5632 and 8192, the odd hidden sizes above, an unaligned base pointer for each, sum aliasing x and aliasing residual, guard bytes around every tensor proving nothing outside them is written and the inputs are unchanged, row results independent of T and of the row's position, determinism, T = 0, an all-zero row and every rejected argument. They run in the todo_gpu_tests target with the todo_joe label (ctest --preset debug -L todo_joe). When they all pass, move the file to tests/gpu/, add it to gpu_tests and delete todo_gpu_tests if it is empty; also run compute-sanitizer --tool memcheck on them, then ncu the kernel at T = 1 and T = 16 against the naive pair and record the before and after in docs.
*/
void add_rmsnorm(const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, float,
                 Stream) {
    std::fprintf(stderr, "unimplemented: add_rmsnorm\n");
    std::abort();
}

}  // namespace engine::fast
