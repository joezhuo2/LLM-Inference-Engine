import { type KernelStory, measured } from "./types";

const v2 = "paged-attention-v2-branch-37";
const v2Wired = "paged-attention-v2-in-the-forward-pass-branch-38";
const shared = "attention-shared-memory-branch-67";

const attentionRow = { "": "attention per decode step (eager profile, context at most 64)" };
const at2048 = { rows: "1", context: "2,048" };

export const attentionV2: KernelStory = {
  slug: "attention-v2",
  index: 9,
  title: "attention v1 → v2",
  short: "The first paged attention kept every score in shared memory and made three passes. An online softmax made it one pass over constant shared memory, with eight lanes per key.",
  formula: "m′ = max(m, mₖ), l′ = l·eᵐ⁻ᵐ′ + Σ eˢ⁻ᵐ′",
  headline: {
    value: `${(measured(v2Wired, attentionRow, "before (e07c022)") / measured(v2Wired, attentionRow, "after (79c1f91)")).toFixed(1)}x`,
    label: "less attention time per decode step",
  },
  sources: [
    { path: "src/kernels/paged_attention.cu", label: "v1" },
    { path: "src/kernels/paged_attention_v2.cu", label: "v2" },
    { path: "scripts/paged_attention_bench.cu", label: "bench" },
  ],
  tests: [
    { path: "tests/gpu/test_paged_attention.cpp", label: "v1" },
    { path: "tests/gpu/test_paged_attention_v2.cpp", label: "v2" },
    { path: "tests/support/paged_attention_harness.h", label: "shared harness" },
  ],
  testing:
    "v1 and v2 run the same harness against a CPU reference: contexts that end inside a block, grouped-query heads, padding rows and scattered block tables, plus Hugging Face's attention on real TinyLlama layers. v2's 18 tests go to 8,192 tokens and include a running maximum that rises, falls and spikes, the case an online softmax gets wrong when it forgets to rescale. Scores are rounded to BF16 the same way in both, so the greedy serve tests, the CUDA Graph test and the paged HF comparison passed unchanged when the runner switched.",
  excerpts: [
    {
      file: "src/kernels/paged_attention.cu",
      from: "for (int token_idx = threadIdx.x; token_idx < context_length; token_idx += blockDim.x) {",
      to: "shared_scores[token_idx] = static_cast<float>(bf16_scaled);",
      caption: "v1, pass one: one thread per token, a whole dot product each, every score kept in shared memory",
    },
    {
      file: "src/kernels/paged_attention_v2.cu",
      from: "for (int d = lane_in_group; d < head_dim; d += 8) {",
      to: "dot += __shfl_xor_sync(0xffffffffu, dot, 1);",
      caption: "v2: eight lanes share a key, then three shuffles add their partial dot products",
    },
    {
      file: "src/kernels/paged_attention_v2.cu",
      from: "float new_running_max = fmaxf(running_max, chunk_max);",
      to: "running_sum += shared_reduction[0];",
      caption: "v2's online softmax: rescale what came before, add this chunk",
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
      section: shared,
      table: 0,
      caption: "dynamic shared memory per thread block, bytes, from each kernel's launch code",
      command: "grep -n shared_mem_bytes src/kernels/paged_attention.cu src/kernels/paged_attention_v2.cu",
    },
    {
      section: v2Wired,
      table: 0,
      caption: "the decode step before and after the runner switched to v2 (scripts/reproduce.sh)",
      command: "scripts/reproduce.sh --skip-tests",
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
      text: "One thread block per row and query head, every score in shared memory, then a pass for the softmax and a pass for the values. The host rejects block tables past 2,048 tokens. Review fixes: the scores overwrote the query in shared memory, and the host read a device array, which was the segfault.",
      metric: { kind: "value", value: "2,048", label: "tokens: the longest context v1 accepts" },
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
          { label: "v1", value: measured(v2, at2048, "v1 (us)") },
          { label: "v2", value: measured(v2, at2048, "v2 (us)"), highlight: true },
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
          { label: "v1", value: measured(v2Wired, attentionRow, "before (e07c022)") },
          { label: "v2", value: measured(v2Wired, attentionRow, "after (79c1f91)"), highlight: true },
        ],
      },
    },
  ],
};
