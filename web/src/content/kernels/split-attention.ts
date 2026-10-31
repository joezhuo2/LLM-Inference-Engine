import { type KernelStory, measured } from "./types";

const split = "split-context-attention-branch-40";
const splitWired = "split-attention-in-the-forward-pass-branch-42";
const splitGraphs = "split-attention-in-cuda-graphs-branch-43";
const occupancy = "split-attention-occupancy-branch-66";

const longPrompt = { "max batch": "1", prompt: "1,984" };
const at2048 = { rows: "1", context: "2,048" };

export const splitAttention: KernelStory = {
  slug: "split-attention",
  index: 8,
  title: "split-context attention",
  short: "At batch 1 attention launches one block per head, a fifth of what the GPU can hold. Cutting each context into ranges fills it, and a merge puts the pieces back in a fixed order.",
  formula: "out = Σ eᵐⁱ⁻ᴹ accᵢ / Σ eᵐⁱ⁻ᴹ lᵢ",
  headline: {
    value: `${(measured(splitGraphs, longPrompt, "main graphs") / measured(splitGraphs, longPrompt, "this branch graphs")).toFixed(1)}x`,
    label: "faster per token at a 1,984-token prompt, batch 1, CUDA Graphs",
  },
  sources: [
    { path: "src/kernels/paged_attention_split.cu", label: "partial + merge kernels" },
    { path: "src/sched/paged_batch.cpp", label: "attention_splits rule" },
    { path: "src/runtime/paged_runner.cpp", label: "scratch buffer and wiring" },
    { path: "src/runtime/decode_graphs.cpp", label: "one graph per split count" },
  ],
  tests: [
    { path: "tests/gpu/test_paged_attention_split.cpp", label: "split kernel" },
    { path: "tests/support/paged_attention_harness.h", label: "shared harness" },
    { path: "tests/gpu/test_decode_graphs.cpp", label: "graphs per split count" },
  ],
  testing:
    "94 tests on the shared paged-attention harness against a CPU reference: every split count from 1 to 16, contexts shorter than the split count (empty splits must change nothing), ranges that end inside a block, grouped-query heads, padding rows and contexts up to 8,192 tokens. Repeated runs must give identical bits, which the fixed merge order guarantees. The new tests were checked against a throwaway kernel and deliberate bugs (a stale maximum, the wrong KV head, an off-by-one) and caught them; compute-sanitizer memcheck reports 0 errors on them and on the 12 DecodeGraphs tests.",
  excerpts: [
    {
      file: "src/kernels/paged_attention_split.cu",
      from: "int split_idx = blockIdx.x;",
      to: "if (range_start >= range_end) return;",
      caption: "Each block finds its range: whole KV blocks, an empty range returns at once",
    },
    {
      file: "src/kernels/paged_attention_split.cu",
      from: "float global_max = partials[0];",
      to: "float final_val = (total_sum > 0.0f) ? (acc / total_sum) : 0.0f;",
      caption: "The merge: every split's partial result rescaled to the global maximum, added in split order",
    },
    {
      file: "src/sched/paged_batch.cpp",
      from: "int64_t attention_splits(int64_t rows, int64_t heads, int64_t longest_context) {",
      to: "return splits;",
      caption: "The split count rule: double while the grid stays at or under 256 blocks and each split keeps 256 tokens",
    },
  ],
  measurements: [
    {
      section: occupancy,
      table: 0,
      caption: "Nsight Compute at one row and 2,048 tokens: grid, waves, warps active and time per launch (clocks locked, cold caches)",
      command: "ncu --kernel-name regex:paged_attention_split_partial_kernel --launch-skip 162 --launch-count 1 --metrics launch__grid_size,launch__waves_per_multiprocessor,sm__warps_active.avg.pct_of_peak_sustained_active,gpu__time_duration.sum /tmp/pa_bench 1 2048",
    },
    {
      section: split,
      table: 0,
      caption: "split against v2, µs per call including the merge kernel (loop of 50 after 3 warm-up calls)",
      command: "nvcc -std=c++20 -arch=sm_120 -Iinclude -Isrc scripts/paged_attention_bench.cu build/release/src/libengine_kernels.a build/release/src/libengine_core.a -o /tmp/pa_bench && /tmp/pa_bench",
    },
    {
      section: splitWired,
      table: 0,
      caption: "ms per token (TPOT p50, two runs) with the split wired into the decode step, 64 output tokens",
      command: "build/release/bin/engine bench --model models/tinyllama --max-batch 1 --num-requests 1 --prompt-len 1984 --output-len 64 --warmup 1 --out /tmp/split.csv",
    },
    {
      section: splitGraphs,
      table: 0,
      caption: "ms per token (TPOT p50, two runs) with one CUDA graph per split count",
      command: "build/release/bin/engine bench --model models/tinyllama --max-batch 1 --num-requests 1 --prompt-len 1984 --output-len 64 --warmup 1 --graphs --out /tmp/split.csv",
    },
  ],
  timeline: [
    {
      date: "Oct 20",
      prs: [47],
      branch: "kernel/paged-attention-v2-impl",
      version: "v2",
      author: "joe",
      title: "The problem shows up in v2's profile",
      text: "One row at 8,192 tokens still took 583 µs after the online softmax, because the grid is one block per head. The profiling notes named splitting the context as the next step, and that it needs a scratch buffer the signature did not have.",
      metric: { kind: "value", value: "32", label: "blocks at batch 1, one per query head" },
    },
    {
      date: "Oct 20",
      prs: [50],
      branch: "kernel/paged-attention-split-stub",
      version: "spec",
      author: "claude",
      title: "Split spec: 94 tests",
      text: "A float scratch buffer for each split's partial result joins the arguments. The merge must add splits in a fixed order so results never depend on timing, and an empty split must change nothing.",
    },
    {
      date: "Oct 23",
      prs: [51],
      branch: "kernel/paged-attention-split-impl",
      version: "split",
      author: "claude",
      title: "Split + merge",
      text: "A partial kernel over (split, head, row) blocks, cut on KV block boundaries, then a merge kernel. Drafted at Joe's request so he could study it. With one row, 8 splits make long contexts 2.8x faster; with 16 rows there are already enough blocks, so splitting ties or loses.",
      metric: {
        kind: "bars",
        label: "one row at 2,048 tokens, µs per call",
        unit: "µs",
        digits: 1,
        bars: [
          { label: "v2", value: measured(split, at2048, "v2") },
          { label: "8 splits", value: measured(split, at2048, "8"), highlight: true },
        ],
      },
    },
    {
      date: "Oct 24",
      prs: [52],
      branch: "runtime/wire-attention-split",
      version: "split",
      author: "claude",
      title: "Choosing the split count",
      text: "attention_splits splits only when there are 1 to 4 rows and a context of 512 tokens or more, aiming for about 256 blocks with at least 256 tokens per split. The runner owns a 66 KB scratch buffer. Eager mode gains; graph replays do not move, because every graph was captured with 1 split.",
      metric: {
        kind: "bars",
        label: "ms per token at a 1,984-token prompt, batch 1, eager",
        unit: "ms",
        digits: 1,
        bars: [
          { label: "v2", value: measured(splitWired, longPrompt, "v2 eager") },
          { label: "split", value: measured(splitWired, longPrompt, "split eager"), highlight: true },
        ],
      },
    },
    {
      date: "Oct 25",
      prs: [53],
      branch: "runtime/graphs-split-attention",
      version: "split",
      author: "claude",
      title: "One CUDA graph per split count",
      text: "Each batch size now gets one graph per split count the rule can return, captured at the first context that produces it: 11 graphs instead of 5. Drafted at Joe's request.",
      metric: {
        kind: "bars",
        label: "ms per token at a 1,984-token prompt, batch 1, CUDA Graphs",
        unit: "ms",
        digits: 1,
        bars: [
          { label: "before", value: measured(splitGraphs, longPrompt, "main graphs") },
          { label: "after", value: measured(splitGraphs, longPrompt, "this branch graphs"), highlight: true },
        ],
      },
    },
    {
      date: "Oct 31",
      prs: [],
      branch: "web/kernel-split-attention",
      version: "split",
      author: "claude",
      title: "Measuring the occupancy",
      text: "Nsight Compute on the partial kernel at 1, 2, 4 and 8 splits: 6 blocks fit per SM, so 156 fill one wave. The time halves with each doubling up to 4 splits and stops at 8, whose 256 blocks need a second, mostly empty wave.",
      metric: {
        kind: "bars",
        label: "warps active, one row at 2,048 tokens",
        unit: "%",
        digits: 1,
        bars: [
          { label: "1 split", value: measured(occupancy, { kernel: "split partial", splits: "1" }, "warps active") },
          { label: "8 splits", value: measured(occupancy, { kernel: "split partial", splits: "8" }, "warps active"), highlight: true },
        ],
      },
    },
  ],
};
