import { type KernelStory, measured } from "./types";

const fused = "fused-add-rmsnorm-branch-30";
const wired = "fast-kernels-in-the-forward-pass-branch-36";
const tuned = "tuning-add-rmsnorm-branch-39";

const oneToken = { tokens: "1" };
const bench =
  "nvcc -std=c++20 -arch=sm_120 -Iinclude -Isrc scripts/add_rmsnorm_bench.cu build/release/src/libengine_kernels.a build/release/src/libengine_core.a -o /tmp/ar_bench";

export const addRmsnorm: KernelStory = {
  slug: "add-rmsnorm",
  index: 2,
  title: "fused add + RMSNorm",
  short: "Two kernels and a round trip to memory become one, then 128-bit loads and the row kept in registers.",
  formula: "sum = x + r,  norm = w · sum / √(mean(sum²) + ε)",
  headline: {
    value: `${(measured(tuned, oneToken, "naive pair (us)") / measured(tuned, oneToken, "tuned fused (us)")).toFixed(1)}x`,
    label: "faster than the naive add + RMSNorm pair, one token",
  },
  sources: [
    { path: "src/kernels/add_rmsnorm.cu", label: "first fused and tuned kernels" },
    { path: "src/kernels/add.cu", label: "v0 add" },
    { path: "src/kernels/rmsnorm.cu", label: "v0 RMSNorm" },
    { path: "src/runtime/paged_runner.cpp", label: "wiring" },
    { path: "scripts/add_rmsnorm_bench.cu", label: "Nsight Compute driver" },
  ],
  tests: [
    { path: "tests/gpu/test_add_rmsnorm.cpp", label: "fused kernel" },
    { path: "tests/gpu/test_rmsnorm.cpp", label: "v0 RMSNorm" },
  ],
  testing:
    "28 GPU tests run the fused kernel against the naive add and RMSNorm pair: 21 shapes, hidden sizes from 1 to 16,384 including odd ones, 6 of them with pointers one element off alignment, so both the tuned and the scalar kernel run. sum must be bit-identical to the naive add; norm may differ only by the order of the sum of squares. The rest cover sum aliasing x or the residual, every row independent of its batch, the same bits on every run, an all-zero row, zero tokens and every rejected argument, with guard bytes around the outputs proving nothing else is written. compute-sanitizer memcheck reports 0 errors. In the forward pass, the paged runner still matches Hugging Face's argmax on 99.57% of teacher-forced tokens.",
  excerpts: [
    {
      file: "src/kernels/add_rmsnorm.cu",
      from: "__shared__ float s_rstd;",
      to: "norm_row[i] = __float2bfloat16(w_val",
      caption: "The first fused kernel: thread 0 computes rstd, a barrier, then the sums are read back from memory",
    },
    {
      file: "src/kernels/add_rmsnorm.cu",
      from: "uint4 xv = x_row[v];",
      to: "sum_row[v] = xv;",
      caption: "The tuned kernel: 8 values per 16-byte load, the rounded sums kept in registers",
    },
    {
      file: "src/kernels/add_rmsnorm.cu",
      from: "__device__ __forceinline__ float blockAllReduceSum(float val) {",
      to: "return val;",
      caption: "xor shuffles leave the total in every lane, so one barrier is enough",
    },
  ],
  measurements: [
    {
      section: fused,
      table: 0,
      caption: "the naive pair against the first fused kernel, GPU time and DRAM bytes per launch",
      command: `${bench} && ncu --kernel-name regex:"add_kernel|rmsnorm_kernel|add_rmsnorm_kernel" --launch-skip 6 --launch-count 3 --metrics gpu__time_duration.sum,dram__bytes.sum /tmp/ar_bench 64`,
    },
    {
      section: tuned,
      table: 0,
      caption: "the tuned kernel against both, µs per launch at hidden 2048",
      command: `${bench} && ncu --kernel-name regex:"add_kernel|rmsnorm_kernel|add_rmsnorm" --launch-skip 6 --launch-count 3 --metrics gpu__time_duration.sum /tmp/ar_bench 64`,
    },
    {
      section: wired,
      table: 0,
      caption: "a batch-1 decode step before and after the fast kernels were wired in (fused add + RMSNorm, RoPE and SwiGLU together)",
      command: "scripts/reproduce.sh --skip-tests",
    },
  ],
  timeline: [
    {
      date: "Oct 9",
      prs: [20],
      branch: "kernel/naive-oracles",
      version: "v0",
      author: "claude",
      title: "Two naive kernels, the oracles",
      text: "add runs one thread per element. rmsnorm runs one block per row: a sum of squares through a shared-memory tree, then a second pass that scales the row. Two launches, and the sum goes out to memory between them.",
    },
    {
      date: "Oct 14",
      prs: [35],
      branch: "kernel/m4-stubs",
      version: "spec",
      author: "claude",
      title: "A spec and 25 tests",
      text: "sum bit-identical to the naive add, norm computed from the rounded sum and equal to the naive RMSNorm up to the order of the sum of squares, any hidden size, any pointer alignment, sum allowed to overwrite x or the residual. Checked against a throwaway kernel that called the naive pair.",
    },
    {
      date: "Oct 15",
      prs: [39],
      branch: "kernel/add-rmsnorm-impl",
      version: "fused",
      author: "joe",
      title: "One kernel instead of two",
      text: "256 threads per row add x and the residual and write the sum; a warp shuffle and a shared-memory pass reduce the squares; thread 0 computes rstd; a second pass reads the sums back and writes the norm. Fixes along the way: square the rounded sum, not the float one; round twice like the naive kernel; rsqrtf, not the double rsqrt. Two thirds of the DRAM bytes, but no faster: about 11 µs at any row count.",
      metric: {
        kind: "bars",
        label: "DRAM bytes per launch at 64 tokens",
        unit: "KB",
        digits: 1,
        bars: [
          { label: "pair", value: measured(fused, { tokens: "64" }, "naive DRAM (KB)") },
          { label: "fused", value: measured(fused, { tokens: "64" }, "fast DRAM (KB)"), highlight: true },
        ],
      },
    },
    {
      date: "Oct 19",
      prs: [46],
      branch: "runtime/wire-fast-kernels",
      version: "fused",
      author: "claude",
      title: "Fused into every layer",
      text: "Each residual add now runs with the norm after it: one per layer after attention, and one into the next layer's input norm. With fast RoPE and SwiGLU, 43 launches fewer per step. The new summation order moves logits by rounding, and one request of the 16-request test now parts from its solo run at an exact BF16 tie, so the serve tests accept exact ties and nothing else. Inside the step the fused kernel was still slower than the pair it replaced, 4.8 against 4.1 µs.",
      metric: {
        kind: "bars",
        label: "kernels per batch-1 decode step",
        unit: "",
        digits: 0,
        bars: [
          { label: "naive", value: measured(wired, { "decode step at batch 1": "kernels per step" }, "naive kernels, eager") },
          { label: "fast", value: measured(wired, { "decode step at batch 1": "kernels per step" }, "fast kernels, eager"), highlight: true },
        ],
      },
    },
    {
      date: "Oct 20",
      prs: [49],
      branch: "kernel/add-rmsnorm-tune",
      version: "tuned",
      author: "claude",
      title: "128-bit loads, the row in registers",
      text: "Drafted at Joe's request. x and the residual come in 8 values per 16-byte load, the rounded sums stay in registers for the second pass, and xor shuffles leave the total in every thread after one barrier. Other shapes keep the first kernel. The two fused ops in a decode step went from 177 to 73 µs.",
      metric: {
        kind: "bars",
        label: "one token at hidden 2048, µs per launch",
        unit: "µs",
        digits: 1,
        bars: [
          { label: "pair", value: measured(tuned, oneToken, "naive pair (us)") },
          { label: "first", value: measured(tuned, oneToken, "first fused (us)") },
          { label: "tuned", value: measured(tuned, oneToken, "tuned fused (us)"), highlight: true },
        ],
      },
    },
  ],
};
