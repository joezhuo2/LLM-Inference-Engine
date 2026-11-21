#include <stdexcept>

#include "engine/kernels/gemm_fp8.h"

namespace engine {

/*
TODO(JOE): a matrix product that reads FP8 weights (gemm_fp8)

Purpose. Issue #104. A decoded token reads every weight once, 2.07 GB of BF16, and at batch 1 the step runs close to the rate memory delivers those bytes: the floor is bytes divided by bandwidth, about 6.4 ms. The arithmetic is not the limit, the bytes are. With the weights stored as FP8 E4M3, one byte per element, a step reads about 1.04 GB and the floor falls to about 3.2 ms. gemm_fp8 is the product that reads such a matrix: the activations stay BF16, only the weights are 8 bits wide. Attention, the norms and the KV cache do not shrink, so the real gain is smaller than half, and what it does to the logits has to be measured (both are for the wiring branch).

Words. E4M3 is described in the spec of write_kv_fp8 (src/kernels/fp8_kv.cu). A weight matrix has one scale per output row: the stored byte is e4m3(w[o][i] / row_scale[o]) and the weight it stands for is float(byte) * row_scale[o]. One scale per row costs out floats for the whole matrix and lets every row use the full range of the format whatever its largest weight is.

Interface. Declared in include/engine/kernels/gemm_fp8.h. y is BF16 [B, out], x is BF16 [B, in], w is F8E4M3 [out, in] row-major (the layout gemm takes, with one byte per element), row_scale is F32 [out]. It runs on stream and does not wait for it. There is no Blas argument: this is your kernel, not a cuBLAS call (see What is not here for the library route).

Semantics. y[b][o] = bf16(row_scale[o] * sum_i float(x[b][i]) * float(w[o][i])). The sum is accumulated in FP32 in any order you like, the multiplication by the scale happens once per output element after the sum, and the result is rounded to BF16 once, to nearest even. Converting a byte to float is exact (cuda_fp8.h does it on the device). x, w and row_scale are not modified and nothing outside y is written.

Numerics. The tests compare against a double precision sum of the same products and allow 2^-7 of the result (one BF16 rounding) plus 1e-4 of the sum of the absolute products plus 1e-6, the tolerance gemm is held to. Products of small integers are exact in FP32, so a hand example must come out exactly. A NaN byte in a row of w makes that output NaN; nothing special is needed for it.

Speed. Correctness is what the tests check. The reason to exist is the byte count, so when it is correct measure it the way profiling.md measures gemm: bytes read per second against the peak, at B = 1 for [2560, 2048] (QKV), [2048, 2048], [11264, 2048] and [2048, 5632] (the MLP) and [32000, 2048] (the LM head), and at B = 16. At B = 1 every weight byte is used once, so the loads decide everything: 16-byte loads, one output row per warp or a few rows per block, x kept in shared memory or registers as FP32. At B = 16 the same bytes are used 16 times and the arithmetic starts to matter.

Errors and preconditions. Before launching anything, throw std::invalid_argument with a message starting with gemm_fp8: when x, y or w is not 2-dimensional or row_scale is not 1-dimensional; x or y is not BF16, w is not F8E4M3 (a BF16 matrix must be rejected, not reinterpreted) or row_scale is not F32; x.shape[1] differs from w.shape[1]; y is not [x.shape[0], w.shape[0]]; row_scale does not have w.shape[0] elements. The values on the device are not checked.

Edge cases. B = 0 is valid and launches nothing (x and y may have null data). in = 1, out = 1, B = 1. in that is not a multiple of 16 or 4 (the tail of a row must not read into the next row's bytes as if they were its own, and must not read past the end of w for the last row). Rows whose scale is 0 give 0. Saturated bytes (448 times the scale).

What to compare against. A double precision loop on the host over float(byte) * row_scale (tests/support/fp8.h decodes the byte with cuda_fp8.h). Not gemm on the original BF16 weights: how far FP8 weights move the result is a property of the quantizer, not of this kernel.

What is not here. (1) The quantizer: for each row, row_scale = largest |w| / 448 and byte = e4m3(w / row_scale), written once at load or into a second safetensors file; Part A, with the upload and a flag, once the tests pass. (2) Wiring into the runners and the measurement the issue asks for: step time, bytes per step and perplexity or logit error against the reference dump, before and after, in results.md. (3) Tensor cores. The PTX manual lists mma.sync with .e4m3 operands from sm_89 and more shapes on sm_120, and cublasLtMatmul takes CUDA_R_8F_E4M3 with a scale for the whole tensor, but both want the two inputs in FP8, so the activations would be rounded to 3 mantissa bits as well. That is a different accuracy question and a different function; check the manual of the CUDA version in use before relying on a shape. At batch 1 it would not help anyway, the step is bound by memory. (4) INT8 weights with mma on s8 operands: same remark. (5) The embedding table, which is read one row per token and gains nothing.

Tests that prove it. tests/todo/gpu/test_gemm_fp8.cpp (target todo_gpu_tests, labels todo_joe and gpu), GemmFp8.* and the shapes in ShapesFp8, checked against a throwaway implementation and mutants of it. Run them with ctest --preset debug -L todo_joe -R GemmFp8. When all pass, move the file to tests/gpu/ and add it to gpu_tests.
*/
void gemm_fp8(const Tensor&, const Tensor&, const Tensor&, const Tensor&, Stream) {
    throw std::logic_error("unimplemented: gemm_fp8");
}

}  // namespace engine
