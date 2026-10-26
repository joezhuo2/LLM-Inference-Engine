import { cellNumber, launchGaps, profilingTable } from "@/lib/data";
import { type KernelStory, measured } from "./types";

const graphs = "decode-cuda-graphs-branch-34";
const wired = "split-attention-in-the-forward-pass-branch-42";
const splitGraphs = "split-attention-in-cuda-graphs-branch-43";

const eager = launchGaps("eager");
const replay = launchGaps("graphs");

export function twoRuns(section: string, match: Record<string, string>, column: string): number {
  const { header, rows } = profilingTable(section);
  const row = rows.find((r) => Object.entries(match).every(([k, v]) => r[header.indexOf(k)] === v));
  const runs = (row?.[header.indexOf(column)] ?? "").split("/").map(cellNumber);
  if (runs.length !== 2 || runs.some((v) => v === undefined)) throw new Error(`docs/profiling.md ${section}: no two runs for ${JSON.stringify(match)} in "${column}"`);
  return ((runs[0] ?? 0) + (runs[1] ?? 0)) / 2;
}

export const longPrompt = { "max batch": "1", prompt: "1,984" };

const first = (row: string, column: string) => measured(graphs, { "decode step at batch 1": row }, column);

export const cudaGraphs: KernelStory = {
  slug: "cuda-graphs",
  index: 6,
  title: "CUDA Graphs",
  short: "Not a kernel: the whole decode step captured once and replayed with one launch, so the GPU stops waiting for the CPU.",
  formula: "decode step = stage(batch) + cudaGraphLaunch(graph[bucket, splits])",
  headline: {
    value: `${eager.idle_pct}% → ${replay.idle_pct}%`,
    label: "of a batch-1 decode step with the GPU idle, eager against a graph replay",
  },
  sources: [
    { path: "src/runtime/decode_graphs.cpp", label: "capture and replay" },
    { path: "include/engine/runtime/decode_graphs.h", label: "DecodeGraphs" },
    { path: "src/runtime/paged_runner.cpp", label: "stage and run_staged" },
    { path: "src/sched/paged_batch.cpp", label: "pad_decode_batch" },
    { path: "src/runtime/serve.cpp", label: "replay or eager, per step" },
  ],
  tests: [
    { path: "tests/gpu/test_decode_graphs.cpp", label: "replays against the eager step" },
    { path: "tests/gpu/test_serve.cpp", label: "serving with and without graphs" },
  ],
  testing:
    "A replay has to be the eager step, so the 12 DecodeGraphs tests compare them on the real model. Replays of batches of 1, 2, 3, 5, 8, 13 and 16 rows are bit-identical to the eager step on the same padded batch, and within the paged runner's tolerance of the unpadded one (a 4-row GEMM can round differently from a 3-row one). A 16-request greedy decode with staggered arrivals crosses every bucket and matches the padded eager loop token for token; 200 replays with changing inputs are compared one by one; a snapshot of the whole KV cache before and after construction proves the captures wrote nothing. At a 2,048-token max context the constructor captures exactly 11 graphs, replays of 1 to 3 long rows (2 to 4 splits) are bit-identical, and a 1,018-token request stays bit-identical as its replays move from the 2-split to the 4-split graph at 1,024 tokens. Two serving tests give the same outputs for 16 greedy requests with and without graphs, also under forced preemption. Compute-sanitizer memcheck on the 12 tests: 0 errors, 0 bytes leaked.",
  excerpts: [
    {
      file: "src/runtime/decode_graphs.cpp",
      from: "cudaGraphExec_t capture(PagedRunner& runner, const PagedBatch& padded) {",
      to: "    return exec;",
      caption: "Capture: one eager run so cuBLAS has picked its kernels, then the same run_staged recorded instead of executed, and instantiated once",
    },
    {
      file: "src/runtime/decode_graphs.cpp",
      from: "    const int64_t heads = runner_->config().heads;",
      to: "            graphs_[{b, splits}] = capture(*runner_, pad_decode_batch(dummy, b));",
      caption: "One graph per bucket and split count: walk the contexts and capture at the first one that gives a new split count",
    },
    {
      file: "src/runtime/decode_graphs.cpp",
      from: "Tensor DecodeGraphs::forward(const PagedBatch& batch) {",
      to: "    CUDA_CHECK(cudaGraphLaunch(graphs_.at({b, splits}), runner_->stream()));",
      caption: "Replay: pad the batch to its bucket, copy it to the runner's fixed device buffers, launch the graph",
    },
  ],
  measurements: [
    {
      section: graphs,
      table: 0,
      caption: "one batch-1 decode step eager and replayed from a graph: Nsight Systems on 4 requests of 64 prompt and 64 output tokens, TPOT from engine bench on 32",
      command: "scripts/profile.sh nsys decode_eager --max-batch 1 --num-requests 4 --prompt-len 64 --output-len 64 (and again with --graphs)",
    },
    {
      section: graphs,
      table: 1,
      caption: "engine bench output tokens/s on 32 requests of the same shape, median of 3 runs",
      command: "build/release/bin/engine bench --model models/tinyllama --max-batch 1 --num-requests 32 --prompt-len 64 --output-len 64 --graphs (and --max-batch 4 and 16, each also without --graphs)",
    },
    {
      section: splitGraphs,
      table: 0,
      caption: "ms per token (TPOT p50, two runs) with one graph per bucket and split count, 64 output tokens",
      command: "build/release/bin/engine bench --model models/tinyllama --max-batch 1 --num-requests 1 --prompt-len 1984 --output-len 64 --warmup 1 --graphs --out /tmp/split.csv",
    },
  ],
  timeline: [
    {
      date: "Oct 5",
      prs: [8],
      branch: "kernel/cublas-gemm",
      version: "prep",
      author: "claude",
      title: "Ready for a capture before there was one",
      text: "cuBLAS allocates its own workspace unless it is given one, and nothing may allocate while a stream is being captured. The Blas handle has taken its workspace from the caller since the first wrapper, and KERNEL_CHECK (#7), which synchronizes the stream after every kernel in debug builds, skips that while the stream is capturing.",
    },
    {
      date: "Oct 15",
      prs: [35],
      branch: "kernel/m4-stubs",
      version: "stub",
      author: "claude",
      title: "A step that reads nothing from the host",
      text: "A graph can only replay work that lives on the GPU. PagedRunner::forward was split into stage, which uploads the batch into the runner's device buffers, and run_staged, which only enqueues kernels on them; pad_decode_batch fills a batch up to a fixed row count with rows that write and read nothing. The DecodeGraphs stub came with its spec and 8 tests, checked against a throwaway implementation before Joe wrote his.",
    },
    {
      date: "Oct 17",
      prs: [44],
      branch: "runtime/cuda-graphs-impl",
      version: "graphs",
      author: "joe",
      title: "DecodeGraphs",
      text: "One graph per decode batch size (1, 2, 4, 8, 16). For each bucket the constructor stages an all-padding batch, runs it once eagerly so cuBLAS has chosen its kernels, captures a second run on the runner's stream and instantiates it. forward pads the batch to its bucket, stages it outside the graph and launches the executable graph. A tidy pass drafted with Claude kept only the executable graph per bucket.",
    },
    {
      date: "Oct 17",
      prs: [44],
      branch: "runtime/cuda-graphs-impl",
      version: "wiring",
      author: "claude",
      title: "A stream that cannot be captured",
      text: "serve and engine bench --graphs replay every decode step that DecodeGraphs covers and run prefills eagerly. The first release run aborted with cudaErrorStreamCaptureUnsupported: the bench built its runner on the legacy default stream, which cannot be captured, and the tests never saw it because they create their own stream. A --graphs run now creates a non-blocking stream.",
    },
    {
      date: "Oct 17",
      prs: [44],
      branch: "runtime/cuda-graphs-impl",
      version: "graphs",
      author: "claude",
      title: "The gaps close",
      text: "The kernels and their GPU time stayed the same (7.9 against 7.8 ms busy per step), but the idle share of a batch-1 step went from 11.3% to 0.9% and the gaps over 5 µs from 17 to 0. The gain shrinks with the batch, 11% of the throughput at batch 1 and 6% at 16: a step launches the same kernels at any batch size, so the time saved per step stays about the same while the step itself gets longer.",
      metric: {
        kind: "bars",
        label: "ms per token, batch 1",
        unit: "ms",
        digits: 2,
        bars: [
          { label: "eager", value: first("median TPOT", "eager") },
          { label: "graph replay", value: first("median TPOT", "graph replay"), highlight: true },
        ],
      },
    },
    {
      date: "Oct 24",
      prs: [52],
      branch: "runtime/wire-attention-split",
      version: "split",
      author: "claude",
      title: "Eager overtakes the graphs",
      text: "Split attention made the eager step 1.4x faster at long contexts, and the graphs did not follow. They were captured on a dummy batch with a context of 0, so the recorded step always ran the one-split v2 kernel, and a replay cannot change a kernel's grid. From about 1,024 tokens on, eager beat the graph replay.",
      metric: {
        kind: "bars",
        label: "ms per token at a 1,984-token prompt, batch 1, mean of two runs",
        unit: "ms",
        digits: 1,
        bars: [
          { label: "graphs, v2", value: twoRuns(wired, longPrompt, "v2 graphs") },
          { label: "eager, split", value: twoRuns(wired, longPrompt, "split eager"), highlight: true },
        ],
      },
    },
    {
      date: "Oct 25",
      prs: [53],
      branch: "runtime/graphs-split-attention",
      version: "split",
      author: "claude",
      title: "One graph per bucket and split count",
      text: "The constructor now walks every context up to the runner's maximum and captures again whenever the split count changes: 11 graphs instead of 5. forward picks the graph from the padded batch's longest context, the same value the eager step reads, so a replay runs exactly the eager step and is about 1 ms faster than it again. Drafted at Joe's request.",
      metric: {
        kind: "bars",
        label: "ms per token at a 1,984-token prompt, batch 1, CUDA Graphs, mean of two runs",
        unit: "ms",
        digits: 1,
        bars: [
          { label: "before", value: twoRuns(splitGraphs, longPrompt, "main graphs") },
          { label: "after", value: twoRuns(splitGraphs, longPrompt, "this branch graphs"), highlight: true },
        ],
      },
    },
  ],
};
