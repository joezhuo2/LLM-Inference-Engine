import { type KernelStory, measured } from "./types";

const v2 = "paged-attention-v2-branch-37";
const v2Wired = "paged-attention-v2-in-the-forward-pass-branch-38";
const split = "split-context-attention-branch-40";
const splitWired = "split-attention-in-the-forward-pass-branch-42";
const splitGraphs = "split-attention-in-cuda-graphs-branch-43";

const longPrompt = { "max batch": "1", prompt: "1,984" };

export const pagedAttention: KernelStory = {
  slug: "paged-attention",
  index: 1,
  title: "paged attention",
  short: "One new query per sequence against its whole cached context, read through a block table: v1, then an online softmax, then one context split across the GPU.",
  formula: "out = softmax(q · Kᵀ / √d) · V",
  headline: {
    value: `${(measured(split, { rows: "1", context: "2,048" }, "v2") / measured(split, { rows: "1", context: "2,048" }, "8")).toFixed(1)}x`,
    label: "faster with 8 splits, one row at 2,048 tokens",
  },
  sources: [
    { path: "src/kernels/paged_attention.cu", label: "v1" },
    { path: "src/kernels/paged_attention_v2.cu", label: "v2" },
    { path: "src/kernels/paged_attention_split.cu", label: "split + merge" },
    { path: "src/runtime/paged_runner.cpp", label: "split count rule and wiring" },
  ],
  tests: [
    { path: "tests/gpu/test_paged_attention.cpp", label: "v1" },
    { path: "tests/gpu/test_paged_attention_v2.cpp", label: "v2" },
    { path: "tests/gpu/test_paged_attention_split.cpp", label: "split" },
    { path: "tests/support/paged_attention_harness.h", label: "shared harness" },
  ],
  testing:
    "Every version runs the same harness against a CPU reference: contexts that end inside a block, grouped-query heads, padding rows, scattered block tables, up to 8,192 tokens for v2 and split, plus Hugging Face's attention on real TinyLlama layers. The split kernel has 94 tests, including every split count from 1 to 16 and empty splits. New test files were checked against a throwaway kernel and deliberate bugs (a stale maximum, the wrong KV head, an off-by-one), all of which they caught; compute-sanitizer memcheck reports 0 errors.",
  excerpts: [
    {
      file: "src/kernels/paged_attention.cu",
      from: "int physical_block_idx = sequence_block_table[token_idx / block_size];",
      to: "(block_offset * head_dim);",
      caption: "Finding a token's key through the block table (v1; v2 and split do the same)",
    },
    {
      file: "src/kernels/paged_attention_v2.cu",
      from: "float new_running_max = fmaxf(running_max, chunk_max);",
      to: "running_sum += shared_reduction[0];",
      caption: "v2's online softmax: rescale what came before, add this chunk",
    },
    {
      file: "src/kernels/paged_attention_split.cu",
      from: "float global_max = partials[0];",
      to: "float final_val = (total_sum > 0.0f) ? (acc / total_sum) : 0.0f;",
      caption: "The merge: every split's partial result rescaled to the global maximum, added in split order",
    },
  ],
  measurements: [
    {
      section: v2,
      table: 0,
      caption: "v1 against v2, every row at the full context, µs per launch (loop of 50 after 3 warm-up launches)",
      command: "nvcc -std=c++20 -arch=sm_120 -Iinclude -Isrc scripts/paged_attention_bench.cu build/release/src/libengine_kernels.a build/release/src/libengine_core.a -o /tmp/pa_bench && /tmp/pa_bench",
    },
    {
      section: v2,
      table: 1,
      caption: "Nsight Compute, one launch each (clocks locked, cold caches, so slower than the loop)",
      command: 'ncu --kernel-name "regex:^paged_attention_decode_v2_kernel" --launch-skip 3 --launch-count 1 /tmp/pa_bench 1 2048',
    },
    {
      section: split,
      table: 0,
      caption: "split against v2, µs per call including the merge kernel",
      command: "/tmp/pa_bench",
    },
    {
      section: splitGraphs,
      table: 0,
      caption: "ms per token (TPOT p50, two runs) with one CUDA graph per split count, 64 output tokens",
      command: "build/release/bin/engine bench --model models/tinyllama --max-batch 1 --num-requests 1 --prompt-len 1984 --output-len 64 --warmup 1 --graphs --out /tmp/split.csv",
    },
  ],
  timeline: [
    {
      date: "Oct 10",
      prs: [26],
      branch: "kernel/paged-decode-attention",
      version: "spec",
      author: "claude",
      title: "A spec and 19 tests before any kernel",
      text: "The signature, a written spec and 19 tests (contexts that end inside a block, grouped-query heads, padding rows, scattered blocks), so the kernel had a definition of done before it existed.",
    },
    {
      date: "Oct 13",
      prs: [32],
      branch: "kernel/paged-attention-impl",
      version: "v1",
      author: "both",
      title: "v1: every score in shared memory",
      text: "One thread block per row and query head. Every score of the context goes into shared memory, then a pass for the softmax and a pass for the values. Review fixes: the scores overwrote the query in shared memory, and the host read a device array, which was the segfault.",
      metric: { kind: "value", value: "2,048", label: "tokens: the longest context v1's shared memory holds" },
    },
    {
      date: "Oct 15",
      prs: [34],
      branch: "model/paged-runner",
      version: "v1",
      author: "claude",
      title: "Serving through the block table",
      text: "The paged runner decodes every step with it. 16 concurrent requests produce the same text as 16 runs on their own, also with forced preemptions.",
    },
    {
      date: "Oct 19",
      prs: [45],
      branch: "kernel/paged-attention-v2-stub",
      version: "spec",
      author: "claude",
      title: "v2 spec, to 8,192 tokens",
      text: "Same arguments as v1, no context cap, an online softmax. 18 tests up to 8,192 tokens, including a running maximum that rises, falls and spikes.",
    },
    {
      date: "Oct 20",
      prs: [47],
      branch: "kernel/paged-attention-v2-impl",
      version: "v2",
      author: "joe",
      title: "v2: online softmax",
      text: "Keys stream through in chunks with a running maximum and sum, so shared memory stays the size of one chunk. Eight lanes share each key so the loads coalesce, and every thread adds up values instead of 64. Tried and dropped: warp shuffles in place of the shared-memory reductions were 15 to 20% slower.",
      metric: {
        kind: "bars",
        label: "one row at 2,048 tokens, µs per launch",
        unit: "µs",
        digits: 0,
        bars: [
          { label: "v1", value: measured(v2, { rows: "1", context: "2,048" }, "v1 (us)") },
          { label: "v2", value: measured(v2, { rows: "1", context: "2,048" }, "v2 (us)"), highlight: true },
        ],
      },
    },
    {
      date: "Oct 20",
      prs: [48],
      branch: "runtime/wire-attention-v2",
      version: "v2",
      author: "claude",
      title: "v2 in the decode step",
      text: "The runner switches to v2. Batch 16 gains the most, because attention work grows with the number of rows while the weight reads do not.",
      metric: {
        kind: "bars",
        label: "attention time per decode step (eager profile)",
        unit: "µs",
        digits: 0,
        bars: [
          { label: "v1", value: measured(v2Wired, { "": "attention per decode step (eager profile, context at most 64)" }, "before (e07c022)") },
          {
            label: "v2",
            value: measured(v2Wired, { "": "attention per decode step (eager profile, context at most 64)" }, "after (79c1f91)"),
            highlight: true,
          },
        ],
      },
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
          { label: "v2", value: measured(split, { rows: "1", context: "2,048" }, "v2") },
          { label: "8 splits", value: measured(split, { rows: "1", context: "2,048" }, "8"), highlight: true },
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
      text: "attention_splits splits only when there are 1 to 4 rows and a context of 512 tokens or more, aiming for about 256 blocks with at least 256 tokens per split. The runner owns a 66 KB scratch buffer.",
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
      text: "Graphs were captured on an empty dummy batch, so every replay ran v2. Now each batch size gets one graph per split count: 11 graphs instead of 5. Drafted at Joe's request.",
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
  ],
};
