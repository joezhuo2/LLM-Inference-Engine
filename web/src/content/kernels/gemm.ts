import { type KernelStory, measured } from "./types";

const roof = "roofline";

export const gemmOps = ["qkv", "o_proj", "gate_up", "down", "lm_head"] as const;

export const roofline = (op: string, rows: string, column: string) => measured(roof, { op, rows }, column, 0, "results");

const b1Roof = gemmOps.map((op) => roofline(op, "1", "% of roof"));
const mlp = (rows: string) => roofline("gate_up", rows, "us/step") + roofline("down", rows, "us/step");

export const gemm: KernelStory = {
  slug: "gemm",
  index: 5,
  title: "GEMM and GEMV",
  short: "Not hand-written: cuBLAS with the row-major trick, and how close batch 1 runs to the memory bandwidth roof.",
  formula: "y[B, out] = x[B, in] · Wᵀ,  W is [out, in]",
  headline: {
    value: `${Math.min(...b1Roof)}–${Math.max(...b1Roof)}%`,
    label: "of the bandwidth roof for the five GEMVs at batch 1",
  },
  sources: [
    { path: "src/kernels/gemm.cpp", label: "the cublasGemmEx call" },
    { path: "include/engine/kernels/gemm.h", label: "Blas handle and gemm" },
    { path: "scripts/gemm_peak.cu", label: "compute ceiling" },
    { path: "scripts/roofline.py", label: "roofline" },
    { path: "src/runtime/paged_runner.cpp", label: "the five calls per layer" },
  ],
  tests: [
    { path: "tests/gpu/test_gemm.cpp", label: "against a CPU reference" },
    { path: "tests/gpu/test_model_runner.cpp", label: "the forward pass against Hugging Face" },
  ],
  testing:
    "cuBLAS is not reimplemented, so the tests check the wrapper around it: that the handle is bound to the given stream, a 3 × 4 by 5 × 4 hand example that must come out exact, rejected shapes and dtypes, and eight shapes against a CPU reference summed in FP64 (1 × 1 × 1, odd sizes, and TinyLlama's fused QKV, down and LM head shapes at 1, 16 and 64 rows), each output within 2⁻⁷ of the reference plus 10⁻⁴ of the sum of the absolute products. Bit identity is not on the table here: cuBLAS picks a different kernel, and so a different summation order, for one row than for 16, and PyTorch runs K and V as separate GEMMs through cuBLASLt, so the engine's K, V and down outputs differ from Hugging Face's in the last bits. The forward-pass tests hold that to at most 0.35% relative error after layer 0 and a 99% argmax match on the logits.",
  excerpts: [
    {
      file: "src/kernels/gemm.cpp",
      from: "const int b = int(x.shape[0]);",
      to: "out, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));",
      caption: "The whole kernel: one cublasGemmEx asking for Yᵀ = W · Xᵀ in column-major terms, which is Y = X · Wᵀ in row-major memory",
    },
    {
      file: "src/kernels/gemm.cpp",
      from: "Blas::Blas(Stream stream, void* workspace, size_t workspace_bytes) : stream_(stream) {",
      to: "CUBLAS_CHECK(cublasSetWorkspace(handle_, workspace, workspace_bytes));",
      caption: "The handle gets its workspace from the caller, so cuBLAS never allocates memory while a CUDA Graph is being captured",
    },
    {
      file: "src/runtime/decode_graphs.cpp",
      from: "cudaGraphExec_t capture(PagedRunner& runner, const PagedBatch& padded) {",
      to: "CUDA_CHECK(cudaStreamEndCapture(runner.stream(), &graph));",
      caption: "One eager step before every capture: the first call of a shape is when cuBLAS picks its kernel and loads it, and neither can happen inside a graph",
    },
  ],
  measurements: [
    {
      doc: "results",
      section: roof,
      table: 0,
      caption: "each GEMM and attention at 1 and 16 rows per decode step: FLOPs and bytes counted from config.json, time from the Nsight Systems profiles",
      command: "scripts/roofline.py results/*_summary.csv --profile results/decode_step.md:1 --profile results/decode_step_b16.md:16 --config models/tinyllama/config.json --context 96 --peak-gb-s 323.6 --peak-tflops 34.5",
    },
    {
      doc: "results",
      section: roof,
      table: 1,
      caption: "the whole decode step (2.07 GB of weights per token) from the median time per output token of each system",
      command: "scripts/reproduce.sh",
    },
  ],
  timeline: [
    {
      date: "Oct 5",
      prs: [8],
      branch: "kernel/cublas-gemm",
      version: "cuBLAS",
      author: "claude",
      title: "The row-major trick",
      text: "One wrapper for every projection: y[B, out] = x[B, in] · Wᵀ in BF16 with FP32 accumulation through a single cublasGemmEx. cuBLAS reads matrices column-major and the model stores them row-major, so the wrapper asks for Yᵀ = W · Xᵀ with the first operand transposed, which reads the row-major W and x as they are and writes a row-major y. Nothing is copied or transposed in memory.",
    },
    {
      date: "Oct 8",
      prs: [12],
      branch: "loader/weight-upload",
      version: "layout",
      author: "claude",
      title: "Seven projections become four",
      text: "The weight layout puts Q, K and V into one [2,560, 2,048] matrix and gate and up into one [11,264, 2,048], so a layer runs four GEMMs instead of seven and each reads the input row once. The loader copies the tensors into those regions in file order, 256-byte aligned.",
    },
    {
      date: "Oct 9",
      prs: [22],
      branch: "model/forward",
      version: "cuBLAS",
      author: "claude",
      title: "Close to Hugging Face, not equal",
      text: "In the forward pass the Q, O, gate and up outputs match Hugging Face exactly, but K, V and down do not: PyTorch runs K and V as separate 256-wide GEMMs through cuBLASLt, and cuBLAS picks other algorithms with other summation orders. Against an FP64 product neither side is more wrong. A one-row decode step also never matches a prefill bit for bit, because one row gets another cuBLAS kernel.",
    },
    {
      date: "Oct 15",
      prs: [37],
      branch: "tools/profiling",
      version: "cuBLAS",
      author: "claude",
      title: "The GEMVs own the step",
      text: "The first Nsight Systems profile tied every kernel to the op that launched it. At batch 1 cuBLAS runs each projection as a gemv kernel, not a GEMM, and even with the naive kernels still around them the step read its weights at 61 to 72% of the peak bandwidth. The work left for hand-written kernels was the launch gaps and the small ops between the GEMVs.",
    },
    {
      date: "Oct 17",
      prs: [44],
      branch: "runtime/cuda-graphs-impl",
      version: "graphs",
      author: "joe",
      title: "cuBLAS inside a CUDA Graph",
      text: "Capturing the decode step means capturing 89 cuBLAS calls. Two things cannot happen during a capture: allocating memory and picking and loading a kernel. The workspace has been the caller's since the wrapper existed, and DecodeGraphs runs every bucket once eagerly before capturing it, so cuBLAS has already chosen.",
    },
    {
      date: "Oct 25",
      prs: [55],
      branch: "tools/roofline",
      version: "roofline",
      author: "claude",
      title: "At the roof",
      text: "A measured compute ceiling (square BF16 GEMMs through the same cublasGemmEx, 34.5 TFLOP/s) next to the measured 323.6 GB/s puts the ridge at 107 FLOP/byte. The five GEMVs sit at 1 FLOP/byte and 91 to 102% of the bandwidth roof: no kernel can make them faster without reading fewer bytes. At 16 rows they take the same time for 16 times the FLOPs.",
      metric: {
        kind: "bars",
        label: "gate/up + down, µs per decode step",
        unit: "µs",
        digits: 0,
        bars: [
          { label: "1 row", value: mlp("1") },
          { label: "16 rows", value: mlp("16"), highlight: true },
        ],
      },
    },
  ],
};
