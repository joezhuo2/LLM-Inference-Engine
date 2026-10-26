import { type KernelStory, measured } from "./types";

const fast = "fast-rope-branch-31";
const wired = "fast-kernels-in-the-forward-pass-branch-36";

const oneToken = { tokens: "1" };
export const rope: KernelStory = {
  slug: "rope",
  index: 3,
  title: "RoPE",
  short: "Pairs of dimensions rotated by position, from one thread per head to every pair in parallel.",
  formula: "x′ᵢ = xᵢ cos pθᵢ − xᵢ₊d/₂ sin pθᵢ,  θᵢ = 10000^(−2i/d)",
  headline: {
    value: `${(measured(fast, oneToken, "naive (us)") / measured(fast, oneToken, "fast (us)")).toFixed(1)}x`,
    label: "faster than the naive kernel, one token",
  },
  sources: [
    { path: "src/kernels/rope_fast.cu", label: "fast kernel" },
    { path: "src/kernels/rope.cu", label: "v0 naive" },
    { path: "include/engine/kernels/rope.h", label: "spec" },
    { path: "src/runtime/paged_runner.cpp", label: "wiring" },
  ],
  tests: [
    { path: "tests/gpu/test_rope_fast.cpp", label: "fast kernel" },
    { path: "tests/gpu/test_rope.cpp", label: "v0 against a CPU reference" },
  ],
  testing:
    "The naive kernel is the oracle: it matches a CPU reference exactly at positions up to 2,047 for three head layouts, leaves V untouched and is the identity at position 0, and was checked once against Hugging Face's own LlamaRotaryEmbedding on layers 0, 7 and 21 of a traced prompt with zero mismatches. 9 GPU tests then require the fast kernel to be bit-identical to it: TinyLlama's layout at 1, 16 and 2,048 tokens with consecutive and random positions, ten other head layouts, a second theta, a base pointer one element off alignment, V and the positions unchanged, equal positions giving equal rotations, position 0, the same bits on every run, zero tokens and every rejected argument. compute-sanitizer memcheck reports 0 errors and 0 bytes leaked.",
  excerpts: [
    {
      file: "src/kernels/rope_fast.cu",
      from: "const int half_head_dim = head_dim / 2;",
      to: "head_ptr[pair + half_head_dim] = __float2bfloat16(hi_cos + lo_sin);",
      caption: "One thread per (pair, head group): cos and sin once, then the same rotation for every head of its group",
    },
    {
      file: "src/kernels/rope_fast.cu",
      from: "InvFreq inv_freq{};",
      to: "int threads_per_block",
      caption: "The frequencies come from the host, as in the naive launcher, and the block is 32 pairs times up to 8 head groups",
    },
  ],
  measurements: [
    {
      section: fast,
      table: 0,
      caption: "the naive kernel, the first draft and the fast kernel, µs per launch and DRAM bytes, TinyLlama's 32 + 4 + 4 heads of 64",
      command: "ncu --kernel-name regex:rope_kernel --launch-skip <n> --launch-count 1 --metrics gpu__time_duration.sum,dram__bytes.sum <program>",
    },
    {
      section: wired,
      table: 0,
      caption: "a batch-1 decode step before and after the fast kernels were wired in (RoPE, SwiGLU and the fused add + RMSNorm together)",
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
      title: "A naive kernel with Hugging Face's bits",
      text: "One block per token, 256 threads striding over every (head, pair) of Q and K, each computing its own cos and sin. Every product is rounded to BF16 on its own, as PyTorch does. The frequencies are computed on the host with std::pow: CUDA's powf differs from PyTorch's CPU value in the last bit for 2 of the 32 frequencies, and near position 2,047 that changed 57 of 1.4 million outputs by up to 64 ulps.",
    },
    {
      date: "Oct 15",
      prs: [35],
      branch: "kernel/m4-stubs",
      version: "spec",
      author: "claude",
      title: "A spec and 9 tests",
      text: "The same rotation, bit-identical to the naive kernel: accurate cosf and sinf, every product rounded on its own, no FMA or fast math, any even head_dim up to 256, unaligned pointers allowed. The spec also asked for 128-bit accesses. Checked against a throwaway that called the naive kernel.",
    },
    {
      date: "Oct 15",
      prs: [40],
      branch: "kernel/rope-vec-impl",
      version: "draft",
      author: "joe",
      title: "One thread per head",
      text: "128 threads per token, one per rotated head, each walking its 32 pairs one after another and computing the frequency on the device with __powf. 3 of the 9 tests failed by a few elements at large positions, and a 128-bit path was started but left empty.",
    },
    {
      date: "Oct 15",
      prs: [40],
      branch: "kernel/rope-vec-impl",
      version: "draft",
      author: "claude",
      title: "The host's frequencies",
      text: "The launcher now computes the frequencies on the host and passes the table by value, like the naive launcher; device powf instead of __powf still failed. All 9 tests pass. The empty vector block went too: it only declared arrays, and nvcc's unused-variable warnings fail the gate. With bit identity settled, the draft was a flat 21 µs at every size, slower than naive until 512 tokens.",
      metric: {
        kind: "bars",
        label: "one token, µs per launch",
        unit: "µs",
        digits: 1,
        bars: [
          { label: "naive", value: measured(fast, oneToken, "naive (us)") },
          { label: "draft", value: measured(fast, oneToken, "first draft (us)"), highlight: true },
        ],
      },
    },
    {
      date: "Oct 15",
      prs: [40],
      branch: "kernel/rope-vec-impl",
      version: "fast",
      author: "claude",
      title: "Every pair in parallel",
      text: "Drafted at Joe's request. Threads own a (pair, head group) slice: 32 pairs times 8 groups is 256 threads, each computing cos and sin once and applying them to 4 or 5 heads. The 36 heads no longer wait on each other, and sin and cos run 256 times per token instead of 1,152. No 128-bit loads in the end: neighbouring threads read neighbouring pairs, so the 2-byte loads of a warp already coalesce.",
      metric: {
        kind: "bars",
        label: "one token, µs per launch",
        unit: "µs",
        digits: 1,
        bars: [
          { label: "naive", value: measured(fast, oneToken, "naive (us)") },
          { label: "draft", value: measured(fast, oneToken, "first draft (us)") },
          { label: "fast", value: measured(fast, oneToken, "fast (us)"), highlight: true },
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
      text: "The paged runner calls the fast kernel in all 22 layers. Bit identity means the logits did not move at all from this change. RoPE was the clearest win of the three fast kernels wired in that day: 96.7 to 56.1 µs per decode step, about 0.5% of the step.",
    },
  ],
};
