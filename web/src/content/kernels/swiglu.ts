import { type KernelStory, measured } from "./types";

const fast = "fast-swiglu-branch-32";
const small = "small-batches-branch-33";

const at = (tokens: string) => ({ tokens });

export const swiglu: KernelStory = {
  slug: "swiglu",
  index: 4,
  title: "SwiGLU",
  short: "Eight values per 16-byte load, and a scalar path for batches too small to fill the GPU.",
  formula: "outⱼ = silu(gateⱼ) · upⱼ,  silu(g) = g / (1 + e⁻ᵍ)",
  headline: {
    value: `${(measured(fast, at("4096"), "naive (us)") / measured(fast, at("4096"), "fast (us)")).toFixed(1)}x`,
    label: "faster than the naive kernel at 4,096 tokens",
  },
  sources: [
    { path: "src/kernels/swiglu_fast.cu", label: "fast kernel" },
    { path: "src/kernels/swiglu.cu", label: "v0 naive" },
    { path: "include/engine/kernels/swiglu.h", label: "spec" },
    { path: "src/runtime/paged_runner.cpp", label: "wiring" },
  ],
  tests: [
    { path: "tests/gpu/test_swiglu_fast.cpp", label: "fast kernel" },
    { path: "tests/gpu/test_swiglu.cpp", label: "v0 against Hugging Face" },
  ],
  testing:
    "The naive kernel is the oracle: rebuilt from the traced prompt's gate and up tensors, it reproduces Hugging Face's act output exactly on all 22 layers, and changing one golden element by one ulp fails naming the layer. It also matches a CPU reference on five shapes, and a hand example pins which half is the gate. 28 GPU tests then require the fast kernel to be bit-identical to it: 22 shapes with aligned and shifted pointers (TinyLlama's ff of 5,632, ff 8 on the vector path, odd and tiny ff on the scalar path, 4, 5 and 9 tokens around the switch, 70,000 rows), magnitudes from 1e-3 to 100, 16 special values in every gate and up pairing (infinities, NaN, signed zeros, an overflowing exp), batch independence, the same bits on every run, guard bytes, zero tokens and every rejected argument. compute-sanitizer memcheck reports 0 errors and 0 bytes leaked.",
  excerpts: [
    {
      file: "src/kernels/swiglu_fast.cu",
      from: "__device__ __forceinline__ __nv_bfloat16 swiglu_value",
      to: "return __float2bfloat16(silu_rounded_float * up_float);",
      caption: "The per-value math both kernels share: exp and a true divide in FP32, silu rounded to BF16 before the product",
    },
    {
      file: "src/kernels/swiglu_fast.cu",
      from: "int64_t vec_idx = idx % vecs_per_row;",
      to: "reinterpret_cast<uint4*>(output_data)[idx] = out_vec;",
      caption: "The vector kernel: two 16-byte loads give 8 gate and 8 up values, one 16-byte store writes 8 outputs",
    },
    {
      file: "src/kernels/swiglu_fast.cu",
      from: "bool aligned = hidden_dim_ff % kVecWidth == 0 &&",
      to: "bool vectorizable = aligned",
      caption: "The dispatch: vectors only when aligned and when there are enough of them to give every SM half a block",
    },
  ],
  measurements: [
    {
      section: fast,
      table: 0,
      caption: "the naive kernel, Joe's first draft and the vector kernel, µs per launch and DRAM bytes, ff 5,632",
      command: "ncu --kernel-name regex:swiglu_ --launch-skip 30 --launch-count 3 --metrics gpu__time_duration.sum,dram__bytes.sum <program> <tokens>",
    },
    {
      section: small,
      table: 0,
      caption: "small batches: the vector-only kernel against the dispatching one, which takes the scalar path below 5 tokens",
      command: "ncu --kernel-name regex:swiglu_ --launch-skip 30 --launch-count 3 --metrics gpu__time_duration.sum,dram__bytes.sum <program> <tokens>",
    },
  ],
  timeline: [
    {
      date: "Oct 9",
      prs: [20],
      branch: "kernel/naive-oracles",
      version: "v0",
      author: "claude",
      title: "A naive kernel with Hugging Face's bits",
      text: "One thread per output value on the fused [tokens, 2 × 5,632] row of the gate/up projection: silu of the gate in FP32 with expf and a true divide, rounded to BF16, then times up and rounded again, the two roundings PyTorch does for act_fn(gate) * up. It reproduces Hugging Face's act tensor exactly on all 22 traced layers.",
    },
    {
      date: "Oct 15",
      prs: [35],
      branch: "kernel/m4-stubs",
      version: "spec",
      author: "claude",
      title: "A spec and 23 tests",
      text: "The same values bit for bit with 128-bit accesses, correct signed zeros and infinities, any ff with a scalar path, unaligned pointers allowed. The tests include 16 special values in every pairing and guard bytes around the output. Checked against a throwaway that called the naive kernel.",
    },
    {
      date: "Oct 16",
      prs: [42],
      branch: "kernel/swiglu-vec-impl",
      version: "draft",
      author: "joe",
      title: "One thread per value, on a 2D grid",
      text: "256 threads per block along the row, one row per blockIdx.y, every value with the naive kernel's arithmetic written out step by step. Already a little faster than naive at every size, and bit-identical to it.",
      metric: {
        kind: "bars",
        label: "one token, µs per launch",
        unit: "µs",
        digits: 1,
        bars: [
          { label: "naive", value: measured(fast, at("1"), "naive (us)") },
          { label: "draft", value: measured(fast, at("1"), "first draft (us)"), highlight: true },
        ],
      },
    },
    {
      date: "Oct 16",
      prs: [42],
      branch: "kernel/swiglu-vec-impl",
      version: "fast",
      author: "claude",
      title: "Eight values per load",
      text: "Each thread now loads 8 gate and 8 up values with two 16-byte loads and writes 8 outputs with one store, when ff is a multiple of 8 and both pointers are 16-byte aligned; anything else runs the scalar kernel. Both kernels call one helper for the math, so their roundings cannot drift apart. At 4,096 tokens the kernel moves its 119 MB at about the GPU's measured peak bandwidth.",
      metric: {
        kind: "bars",
        label: "4,096 tokens, µs per launch",
        unit: "µs",
        digits: 0,
        bars: [
          { label: "naive", value: measured(fast, at("4096"), "naive (us)") },
          { label: "draft", value: measured(fast, at("4096"), "first draft (us)") },
          { label: "vector", value: measured(fast, at("4096"), "fast (us)"), highlight: true },
        ],
      },
    },
    {
      date: "Oct 16",
      prs: [42],
      branch: "kernel/swiglu-vec-impl",
      version: "fast",
      author: "claude",
      title: "A flat grid",
      text: "gridDim.y is capped at 65,535, so with the row on blockIdx.y a batch of more than 65,535 rows failed to launch. Both kernels now run on a flat 1D grid over every value (every vector on the vector path) and derive the row and column from the index. Two 70,000-row tests cover it.",
    },
    {
      date: "Oct 16",
      prs: [43],
      branch: "kernel/swiglu-small-batch",
      version: "fast",
      author: "claude",
      title: "The scalar path for small launches",
      text: "At one token the vector kernel was no faster than naive: 5,632 values make 704 vectors, 3 blocks on a 26-SM GPU, each thread running 8 exp and divide chains in a row. Below sm_count × 256 / 2 vectors, 5 tokens here, the kernel now takes the scalar path, whose 22 blocks spread the same chains wider.",
      metric: {
        kind: "bars",
        label: "one token, µs per launch",
        unit: "µs",
        digits: 2,
        bars: [
          { label: "naive", value: measured(small, at("1"), "naive (us)") },
          { label: "vector", value: measured(small, at("1"), "previous (us)") },
          { label: "dispatch", value: measured(small, at("1"), "fast (us)"), highlight: true },
        ],
      },
    },
    {
      date: "Oct 19",
      prs: [46],
      branch: "runtime/wire-fast-kernels",
      version: "fast",
      author: "claude",
      title: "In every layer",
      text: "The paged runner calls the fast kernel in all 22 layers. Bit identity means the logits did not move from this change. The decode step barely noticed: SwiGLU was already about 23 µs of a 9 ms step, and the GEMVs around it are 62%.",
    },
  ],
};
