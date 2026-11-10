import { cellNumber, graphRule, launchGaps, profilingTable } from "@/lib/data";
import { bucketFor, decodeGraphBuckets } from "@/lib/graph-buckets";
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

const pastSixteen = "batch-size-past-16";
const past = (match: Record<string, string>, column: string, table: number) => measured(pastSixteen, match, column, table, "results");
const firstColumn = (table: number) => profilingTable(pastSixteen, table, "results").rows.map((r) => r[0]);

export const sweepPoints = firstColumn(0).map((b) => ({ batch: Number(b), graphs: past({ "max batch": b }, "graphs tokens/s", 0), eager: past({ "max batch": b }, "eager tokens/s", 0), graphsTpot: past({ "max batch": b }, "graphs TPOT p50 (ms)", 0), eagerTpot: past({ "max batch": b }, "eager TPOT p50 (ms)", 0) }));
export const sweepOps = firstColumn(1).map((b) => ({ batch: Number(b), attention: past({ "max batch": b }, "attention µs", 1), gemms: past({ "max batch": b }, "GEMMs µs", 1), busy: past({ "max batch": b }, "busy µs", 1), share: past({ "max batch": b }, "attention share", 1) }));
const gains = sweepPoints.slice(1).map((p, i) => ({ batch: p.batch, gain: p.graphs / sweepPoints[i].graphs - 1 }));
const lastBigGain = gains.filter((g) => g.gain > 0.2).pop();
if (!lastBigGain || sweepPoints.length < 2) throw new Error("docs/results.md Batch size past 16: throughput never rises by more than 20% for a doubling of the max batch");
export const sweepKnee = lastBigGain.batch;

const paddingHeader = profilingTable(pastSixteen, 2, "results").header.find((h) => h.startsWith("bucket for")) ?? "";
const paddingRows = cellNumber(paddingHeader.replace("bucket for", ""));
if (paddingRows === undefined) throw new Error("docs/results.md Batch size past 16: the padding table has no \"bucket for N rows\" column");
const paddingRuns = firstColumn(2).map((steps) => ({ steps: Number(steps), bucket: past({ "graph steps": steps }, paddingHeader, 2), tpot: past({ "graph steps": steps }, "TPOT p50 (ms)", 2), tokens: past({ "graph steps": steps }, "tokens/s", 2) }));
const paddingMaxBatch = Math.max(...paddingRuns.map((r) => r.bucket));
for (const r of paddingRuns) {
  const bucket = bucketFor(paddingRows, decodeGraphBuckets(paddingMaxBatch, r.steps));
  if (bucket !== r.bucket) throw new Error(`docs/results.md Batch size past 16: ${paddingRows} rows with ${r.steps} graph steps replay the ${r.bucket} graph in the table but the ${bucket} graph by the bucket rule`);
}
const defaultSteps = graphRule().steps;
const coarseRun = paddingRuns.find((r) => r.steps === 1);
const defaultRun = paddingRuns.find((r) => r.steps === defaultSteps);
if (!coarseRun || !defaultRun) throw new Error(`docs/results.md Batch size past 16: the padding table has no row for 1 or for ${defaultSteps} graph steps`);
export const padding = {
  rows: paddingRows,
  maxBatch: paddingMaxBatch,
  steps: defaultSteps,
  runs: paddingRuns,
  coarse: decodeGraphBuckets(paddingMaxBatch, 1),
  fine: decodeGraphBuckets(paddingMaxBatch, defaultSteps),
  coarseBucket: coarseRun.bucket,
  fineBucket: defaultRun.bucket,
  costPct: Math.round((100 * (coarseRun.tpot - defaultRun.tpot)) / defaultRun.tpot),
};

const kneePoint = sweepPoints.find((p) => p.batch === sweepKnee);
const halfStep = sweepOps.find((o) => o.share >= 50) ?? sweepOps[sweepOps.length - 1];
const reproBuckets = decodeGraphBuckets(sweepPoints[0].batch, defaultSteps);

const first = (row: string, column: string) => measured(graphs, { "decode step at batch 1": row }, column);

export const cudaGraphs: KernelStory = {
  slug: "cuda-graphs",
  index: 6,
  book: true,
  about: true,
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
    { path: "src/sched/graph_buckets.cpp", label: "decode_graph_buckets" },
    { path: "src/runtime/serve.cpp", label: "replay or eager, per step" },
    { path: "scripts/batch_sweep.sh", label: "the sweep past 16 rows" },
    { path: "scripts/batch_report.py", label: "medians of the sweep" },
  ],
  tests: [
    { path: "tests/gpu/test_decode_graphs.cpp", label: "replays against the eager step" },
    { path: "tests/gpu/test_serve.cpp", label: "serving with and without graphs" },
    { path: "tests/unit/test_graph_buckets.cpp", label: "the bucket rule" },
    { path: "tests/gpu/test_server.cpp", label: "streamed clients across finer buckets" },
    { path: "tests/python/test_batch_report.py", label: "the sweep report" },
  ],
  testing:
    "A replay has to be the eager step, so the 12 DecodeGraphs tests compare them on the real model. Replays of batches of 1, 2, 3, 5, 8, 13 and 16 rows are bit-identical to the eager step on the same padded batch, and within the paged runner's tolerance of the unpadded one (a 4-row GEMM can round differently from a 3-row one). A 16-request greedy decode with staggered arrivals crosses every bucket and matches the padded eager loop token for token; 200 replays with changing inputs are compared one by one; a snapshot of the whole KV cache before and after construction proves the captures wrote nothing. At a 2,048-token max context the constructor captures exactly 11 graphs from the buckets 1, 2, 4, 8 and 16, replays of 1 to 3 long rows (2 to 4 splits) are bit-identical, and a 1,018-token request stays bit-identical as its replays move from the 2-split to the 4-split graph at 1,024 tokens. Two serving tests give the same outputs for 16 greedy requests with and without graphs, also under forced preemption. Compute-sanitizer memcheck on the 12 tests: 0 errors, 0 bytes leaked. The bucket rule has 18 CPU tests of its own: exact lists at 1, 2, 4 and 8 sizes per power of two, a max batch on and off the grid, and for every max batch up to 300 that the list rises from 1 to the max batch, keeps every power of two, contains the coarser list, and never pads a batch by as much as its own rows divided by the sizes per power of two. A server test streams six clients at max batch 6 across the buckets 1, 2, 3, 4 and 6 and gets the greedy answers, and step counts that are not a power of two are rejected.",
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
    {
      file: "src/sched/graph_buckets.cpp",
      from: "    std::vector<int64_t> buckets;",
      to: "    return buckets;",
      caption: "The bucket sizes: from each power of two to the next in steps_per_octave equal steps, then the max batch itself",
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
      caption: "engine bench output tokens/s on 32 requests of the same shape, median of 3 runs, measured with one bucket per power of two",
      command: "build/release/bin/engine bench --model models/tinyllama --max-batch 1 --num-requests 32 --prompt-len 64 --output-len 64 --graphs (and --max-batch 4 and 16, each also without --graphs)",
    },
    {
      section: splitGraphs,
      table: 0,
      caption: "ms per token (TPOT p50, two runs) with one graph per bucket and split count, 64 output tokens",
      command: "build/release/bin/engine bench --model models/tinyllama --max-batch 1 --num-requests 1 --prompt-len 1984 --output-len 64 --warmup 1 --graphs --out /tmp/split.csv",
    },
    {
      section: pastSixteen,
      table: 0,
      doc: "results",
      caption: "max batch 16 to 256, eager and with graphs: twice as many requests as the max batch, all submitted at once, 64 prompt and 64 output tokens, median of 3 runs, measured with one bucket per power of two",
      command: "scripts/batch_sweep.sh && python3 scripts/batch_report.py results/batch",
    },
    {
      section: pastSixteen,
      table: 1,
      doc: "results",
      caption: "GPU µs per decode step by op at each max batch, from eager Nsight Systems profiles; the GEMM column adds qkv, o_proj, gate_up, down and lm_head",
      command: "scripts/profile.sh nsys batch_b64 --max-batch 64 --num-requests 64 --prompt-len 64 --output-len 64 --warmup 1 (and 32, 128, 256)",
    },
    {
      section: pastSixteen,
      table: 2,
      doc: "results",
      caption: "17 requests at max batch 32 with 1, 2, 4 and 8 graph sizes per power of two, 256 output tokens, median of 5 runs",
      command: "SKIP_SWEEP=1 scripts/batch_sweep.sh (17 requests at --max-batch 32 --graphs, once per --graph-steps value)",
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
    {
      date: "Nov 9",
      prs: [183],
      branch: "runtime/graph-buckets-impl",
      version: "buckets",
      author: "joe",
      title: "decode_graph_buckets",
      text: "The bucket sizes become a function of their own: it walks each range from one power of two to the next below the max batch in steps of max(1, 2^k / steps_per_octave) and appends the max batch. One step per power of two gives the old 1, 2, 4, 8, 16; a max batch below 1 or a step count that is not a power of two is rejected. Its 18 tests came with the stub (#176), checked against a throwaway implementation before Joe wrote his. Nothing called it yet.",
    },
    {
      date: "Nov 10",
      prs: [187],
      branch: "runtime/graph-buckets-wiring",
      version: "sweep",
      author: "claude",
      title: "--graph-steps, and a sweep past 16 rows",
      text: `engine bench and engine serve now take their buckets from decode_graph_buckets through a new --graph-steps option, 1 by default at this point. scripts/batch_sweep.sh then ran the max batch from ${sweepPoints[0].batch} to ${sweepPoints[sweepPoints.length - 1].batch}, eager and with graphs, with twice as many requests as the max batch submitted at once. Throughput nearly stops rising after ${sweepKnee} rows: attention grows with the rows to ${halfStep.share}% of the step at ${halfStep.batch}, and the GEMMs pass the time of one read of the weights. An offline workload on one GPU, not a serving result.`,
      metric: {
        kind: "bars",
        label: "output tokens/s with graphs, by max batch, median of 3 runs",
        unit: "tok/s",
        digits: 0,
        bars: sweepPoints.map((p) => ({ label: `max batch ${p.batch}`, value: p.graphs, highlight: p === kneePoint })),
      },
    },
    {
      date: "Nov 10",
      prs: [190],
      branch: "runtime/graph-steps-default",
      version: "default",
      author: "claude",
      title: "Two buckets per power of two",
      text: `--graph-steps ${defaultSteps} became the default: the buckets at max batch ${sweepPoints[0].batch} are ${reproBuckets.join(", ")}. In the padding runs of the sweep, ${padding.rows} requests at max batch ${padding.maxBatch} cost about ${padding.costPct}% more time per token in the ${padding.coarseBucket}-row graph than in the ${padding.fineBucket}-row one. More steps gained nothing and 8 was slower, which was not profiled. The runs were short and noisy, and the capture time and memory of the extra graphs are not measured. Every earlier graph number on this page was measured with one bucket per power of two.`,
      metric: {
        kind: "bars",
        label: `ms per token, ${padding.rows} requests at max batch ${padding.maxBatch}, median of 5 runs`,
        unit: "ms",
        digits: 2,
        bars: [
          { label: `${padding.coarseBucket}-row graph`, value: coarseRun.tpot },
          { label: `${padding.fineBucket}-row graph`, value: defaultRun.tpot, highlight: true },
        ],
      },
    },
  ],
};
