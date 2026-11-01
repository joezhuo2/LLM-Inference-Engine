import { type KernelStory, measured } from "./types";

const section = "sampling-branch-65";
const one = (column: string) => measured(section, { rows: "1" }, column);

export const sampling: KernelStory = {
  slug: "sampling",
  index: 10,
  title: "sampling",
  short: "Greedy, temperature, top-k and top-p on the GPU without sorting the vocabulary: Gumbel-max turns sampling into an argmax, and binary-searched thresholds replace the sort.",
  formula: "id = argmax(logits / T + g),  g = −log(−log u)",
  headline: {
    value: `${(one("full sort alone") / one("top-k 50")).toFixed(1)}x`,
    label: "top-k 50 against sorting the row alone, one row",
  },
  sources: [
    { path: "src/sampling/sample.cu", label: "Gumbel-max, top-k, top-p" },
    { path: "src/sampling/greedy.cu", label: "greedy" },
    { path: "src/sampling/argmax.cuh", label: "block argmax" },
    { path: "scripts/sample_bench.cu", label: "bench" },
  ],
  tests: [
    { path: "tests/gpu/test_sample.cpp", label: "sampler" },
    { path: "tests/unit/test_sampling_params.cpp", label: "parameter checks" },
  ],
  testing:
    "23 GPU tests. The draws are checked statistically: tens of thousands of samples per setting against the softmax, the truncated softmax of top-k and the nucleus of top-p, by a chi-square test, on small vocabularies and on Hugging Face logits of real prompts, with no token ever drawn that the distribution forbids. Exact tests cover ties at the top-k and top-p cut, negative values and both zeros, logits whose exponential overflows, rows of −∞, temperature 0 matching the greedy kernel, the same seed and step repeating, and a row drawing the same token alone or in a batch.",
  excerpts: [
    {
      file: "src/sampling/sample.cu",
      from: "__device__ float gumbel(uint64_t seed, uint64_t offset) {",
      to: "return float(-log(-log(u)));",
      caption: "Gumbel noise from Philox, keyed by the request's seed and the token's offset (in double precision, which costs most of the time)",
    },
    {
      file: "src/sampling/sample.cu",
      from: "__device__ uint32_t top_k_threshold(int vocab, int k, Score score) {",
      to: "return lo;",
      caption: "Top-k by binary search over 32-bit keys: count the logits at or above the cut, no sort",
    },
    {
      file: "src/sampling/sample.cu",
      from: "const int arg = block_argmax(vocab, [&](int i) {",
      to: "});",
      caption: "The draw: greedy at temperature 0, otherwise noise on the kept tokens and an argmax",
    },
  ],
  measurements: [
    {
      section,
      table: 0,
      caption: "µs per call, vocab 32,000, temperature 0.7, two runs; the sort column is a lower bound for a sort-based top-p",
      command: "nvcc -std=c++20 -O3 -arch=sm_120 -Iinclude -Isrc scripts/sample_bench.cu build/release/src/libengine_kernels.a build/release/src/libengine_core.a -o /tmp/sample_bench && /tmp/sample_bench",
    },
    {
      section,
      table: 1,
      caption: "Nsight Compute on one temperature-only launch at 1 row",
      command: "ncu --kernel-name regex:sample_kernel --launch-skip 3 --launch-count 1 --metrics sm__pipe_fp64_cycles_active.max,sm__cycles_elapsed.max /tmp/sample_bench 1",
    },
    {
      section,
      table: 2,
      caption: "throwaway variants of gumbel(), interleaved with the unchanged kernel in one session (not committed code)",
      command: "/tmp/sample_bench",
    },
  ],
  timeline: [
    {
      date: "Oct 9",
      prs: [22],
      branch: "model/forward",
      version: "greedy",
      author: "joe",
      title: "Greedy argmax on the GPU",
      text: "One 1,024-thread block per row, each thread keeping its best value and index, then a tree reduction. Ties go to the lowest index, so the answer never depends on thread order.",
    },
    {
      date: "Oct 9",
      prs: [23],
      branch: "sampling/temperature",
      version: "gumbel",
      author: "joe",
      title: "Temperature sampling by Gumbel-max",
      text: "Philox noise added to every scaled logit, then the same block argmax, now a shared device header. No softmax, no prefix sum, no sort.",
    },
    {
      date: "Oct 10",
      prs: [27],
      branch: "sampling/batched",
      version: "batched",
      author: "joe",
      title: "Per-row parameters, top-k and top-p",
      text: "Each row carries its own seed, step and temperature, so greedy and sampled requests share one launch. Top-k binary-searches a count threshold and top-p a mass threshold over 32-bit keys of the logits, with ties at the cut kept.",
    },
    {
      date: "Oct 31",
      prs: [105],
      branch: "tools/sample-bench",
      version: "bench",
      author: "claude",
      title: "First measurement",
      text: "A bench against a segmented radix sort of the rows. Top-k beats the sort alone by about 2x; plain temperature sampling turned out the slowest mode, because the Gumbel noise uses double-precision log for all 32,000 tokens.",
      metric: {
        kind: "bars",
        label: "one row, µs per call",
        unit: "µs",
        digits: 1,
        bars: [
          { label: "sort alone", value: one("full sort alone") },
          { label: "top-k 50", value: one("top-k 50"), highlight: true },
          { label: "temperature", value: one("temperature") },
        ],
      },
    },
  ],
};
