import { measured as tableValue } from "@/content/kernels/types";
import { cellNumber, embeddingLaunch, medianOf, profilingTable } from "@/lib/data";

export type Bar = { label: string; value: number; highlight?: boolean };

export type Metric =
  | { kind: "value"; value: string; label: string }
  | { kind: "bars"; label: string; unit: string; digits: number; bars: Bar[] };

export type Entry = {
  date: string;
  prs: number[];
  branch: string;
  title: string;
  text: string;
  metric?: Metric;
  scene?: "bandwidth" | "reduce" | "gemm" | "weights" | "smem" | "vram" | "warps" | "tokenize" | "generate" | "sample" | "roofline" | "chunked";
};

export type Milestone = { id: string; name: string; dates: string; goal: string; entries: Entry[] };

export const repoUrl = "https://github.com/joezhuo2/LLM-Inference-Engine";

function measured(system: string, batch: number) {
  return medianOf(system, batch, "output_throughput") ?? 0;
}

function largestGap(system: string) {
  return tableValue("under-load", { system, QPS: "4" }, "largest gap ms", 1, "results");
}

function goodputAt(system: string, qps: string) {
  return tableValue("under-load", { system, QPS: qps }, "goodput req/s", 0, "results");
}

const sweep = "batch-size-past-16";

function sweepTokens(maxBatch: string) {
  return tableValue(sweep, { "max batch": maxBatch }, "graphs tokens/s", 0, "results");
}

function padding(steps: string, column: string) {
  return tableValue(sweep, { "graph steps": steps }, column, 2, "results");
}

function graphBuckets(maxBatch: number, steps: number) {
  const sizes: number[] = [];
  for (let lo = 1; lo < maxBatch; lo *= 2) for (let b = lo; b < Math.min(2 * lo, maxBatch); b += Math.max(1, lo / steps)) sizes.push(b);
  return [...sizes, maxBatch];
}

const whole = (n: number) => n.toLocaleString("en-US");
const sweepBatches = profilingTable(sweep, 0, "results").rows.map((r) => r[0]);
const sweepGain = sweepBatches.map((b, i) => (i ? sweepTokens(b) / sweepTokens(sweepBatches[i - 1]) : 0));
const knee = sweepBatches.slice(1, -1).reduce((best, b) => (sweepGain[sweepBatches.indexOf(b)] - sweepGain[sweepBatches.indexOf(b) + 1] > sweepGain[sweepBatches.indexOf(best)] - sweepGain[sweepBatches.indexOf(best) + 1] ? b : best));
const afterKnee = sweepBatches.slice(sweepBatches.indexOf(knee) + 1);
const bucketColumn = profilingTable(sweep, 2, "results").header.find((h) => h.startsWith("bucket for")) ?? "";
const paddedRows = cellNumber(bucketColumn.replace("bucket for", ""));
const graphSteps = profilingTable(sweep, 2, "results").rows.map((r) => r[0]);
const defaultSteps = 2;
const bucket = (steps: string) => padding(steps, bucketColumn);
const padded = { coarse: padding("1", "TPOT p50 (ms)"), fine: padding(String(defaultSteps), "TPOT p50 (ms)") };
const lookup = embeddingLaunch();

export const milestones: Milestone[] = [
  {
    id: "M0",
    name: "practice",
    dates: "Sep 24 to Oct 3",
    goal: "Learn the GPU before writing the engine.",
    entries: [
      {
        date: "Sep 30",
        prs: [2],
        branch: "exercises",
        scene: "bandwidth",
        title: "Measure the hardware first",
        text: "A bandwidth benchmark sets the number every later kernel is judged against: how fast this laptop GPU can read its own memory.",
        metric: { kind: "value", value: "323.6 GB/s", label: "measured DRAM bandwidth, RTX 5060 laptop" },
      },
      {
        date: "Oct 1",
        prs: [3],
        branch: "practice/reduction",
        scene: "reduce",
        title: "Three ways to sum a vector",
        text: "The same reduction with global atomics, then shared memory, then warp shuffles: the building block of RMSNorm and softmax later on.",
      },
      {
        date: "Oct 3",
        prs: [4, 5],
        branch: "practice/tinyllama",
        title: "TinyLlama in Python, by hand",
        text: "The whole model written out in PyTorch without the transformers library, then a KV cache and a generation loop checked against Hugging Face. The C++ engine is built to match it.",
      },
    ],
  },
  {
    id: "M1",
    name: "foundations",
    dates: "Oct 4 to Oct 9",
    goal: "Get the weights onto the GPU, bit for bit.",
    entries: [
      {
        date: "Oct 4",
        prs: [6, 7],
        branch: "build/m1-scaffolding",
        title: "Scaffolding and core types",
        text: "CMake presets, a subcommand CLI and CI, then the types everything else stands on: DType, Tensor, DeviceBuffer, pinned host buffers and error-checking macros.",
      },
      {
        date: "Oct 5",
        prs: [8],
        branch: "kernel/cublas-gemm",
        scene: "gemm",
        title: "BF16 matrix multiply",
        text: "cuBLAS is column-major and the model is row-major, so the engine asks it for the transposed product instead of copying anything.",
      },
      {
        date: "Oct 9",
        prs: [10, 12, 14],
        branch: "loader/safetensors-parser",
        scene: "weights",
        title: "Loading 2.2 GB without a library",
        text: "Memory-map the safetensors file, parse its JSON header, and upload through two pinned buffers so reading the next chunk overlaps copying the last one.",
        metric: { kind: "value", value: "0.26 s", label: "to put 2.20 GB of weights on the GPU" },
      },
      {
        date: "Oct 9",
        prs: [15],
        branch: "tools/checksum",
        title: "Prove the weights arrived intact",
        text: "Checksum every tensor on the device and compare with the same checksum computed by PyTorch.",
        metric: { kind: "value", value: "201 / 201", label: "tensors identical to PyTorch" },
      },
    ],
  },
  {
    id: "M2",
    name: "a correct naive engine",
    dates: "Oct 9 to Oct 10",
    goal: "Generate the same text as Hugging Face, with simple kernels.",
    entries: [
      {
        date: "Oct 9",
        prs: [19],
        branch: "tokenizer/sentencepiece",
        scene: "tokenize",
        title: "Tokenizer and chat template",
        text: "SentencePiece and TinyLlama's chat template, checked token for token against the Hugging Face tokenizer.",
      },
      {
        date: "Oct 9",
        prs: [20, 21],
        branch: "kernel/naive-oracles",
        title: "One plain kernel per operation",
        text: "Embedding, RMSNorm, RoPE, SwiGLU, residual add and causal grouped-query attention, the attention checked against Hugging Face on all 22 layers. Slow on purpose: every fast kernel later must match them.",
      },
      {
        date: "Oct 9",
        prs: [22],
        branch: "model/forward",
        scene: "generate",
        title: "First text out of the engine",
        text: "The forward pass, greedy decoding and engine generate, held to Hugging Face's logits on 20 reference prompts.",
      },
      {
        date: "Oct 10",
        prs: [23],
        branch: "sampling/temperature",
        scene: "sample",
        title: "Sampling",
        text: "Temperature sampling with the Gumbel-max trick and Philox random numbers, so a seed reproduces a run exactly.",
      },
    ],
  },
  {
    id: "M3",
    name: "paged KV cache and batching",
    dates: "Oct 10 to Oct 15",
    goal: "Serve many requests at once without wasting memory.",
    entries: [
      {
        date: "Oct 11",
        prs: [24, 25, 29],
        branch: "kv/block-manager",
        scene: "vram",
        title: "KV cache in fixed-size blocks",
        text: "Like pages of virtual memory: a block manager hands out blocks from one pool on the GPU, so a request only holds the memory its tokens use.",
      },
      {
        date: "Oct 13",
        prs: [26, 31, 32],
        branch: "kernel/paged-attention-impl",
        title: "Paged attention",
        text: "Writing each new key and value into its request's block, and a decode attention kernel that follows a block table to find them wherever they sit in the pool.",
      },
      {
        date: "Oct 14",
        prs: [28, 33],
        branch: "sched/scheduler-impl",
        title: "Continuous batching",
        text: "Requests join and leave the batch at every step instead of waiting for the slowest one; when the cache runs out, the newest request is preempted and recomputed later.",
      },
      {
        date: "Oct 15",
        prs: [34],
        branch: "model/paged-runner",
        title: "Everything wired together",
        text: "16 concurrent requests produce the same text as 16 runs on their own, including with forced preemptions.",
        metric: { kind: "value", value: "16 / 16", label: "concurrent requests identical to their solo runs" },
      },
      {
        date: "Oct 15",
        prs: [36, 38],
        branch: "tools/baselines",
        title: "Benchmark and a baseline",
        text: "A workload of 32 requests, run by the engine and by Hugging Face transformers on the same GPU.",
        metric: {
          kind: "bars",
          label: "output tokens per second",
          unit: "tok/s",
          digits: 0,
          bars: [
            { label: "engine, batch 1", value: 108.7 },
            { label: "HF, batch 16", value: 531.9 },
            { label: "engine, batch 16", value: 1013.8, highlight: true },
          ],
        },
      },
    ],
  },
  {
    id: "M4",
    name: "fast",
    dates: "Oct 15 to Oct 25",
    goal: "Close the gap to the memory bandwidth roof.",
    entries: [
      {
        date: "Oct 15",
        prs: [37],
        branch: "tools/profiling",
        scene: "smem",
        title: "Profile before optimizing",
        text: "NVTX ranges and Nsight Systems on one decode step. Even the naive kernels read the weights at 61 to 72% of peak; the rest goes to idle gaps between hundreds of small launches.",
        metric: { kind: "value", value: "267", label: "kernel launches per decode step" },
      },
      {
        date: "Oct 16",
        prs: [39, 40, 42, 46],
        branch: "runtime/wire-fast-kernels",
        scene: "smem",
        title: "Fused and vectorized kernels",
        text: "Residual add fused into RMSNorm, RoPE and SwiGLU rewritten with 16-byte loads, each tested against its naive oracle before it replaced it.",
        metric: {
          kind: "bars",
          label: "kernel launches per decode step",
          unit: "",
          digits: 0,
          bars: [
            { label: "naive", value: 267 },
            { label: "fused", value: 224, highlight: true },
          ],
        },
      },
      {
        date: "Oct 17",
        prs: [44],
        branch: "runtime/cuda-graphs-impl",
        title: "CUDA Graphs",
        text: "The whole decode step is captured once per batch size and replayed with a single launch, so the GPU stops waiting on the CPU. Time per token at batch 1 went from 9.49 to 8.51 ms.",
        metric: {
          kind: "bars",
          label: "GPU idle during a decode step",
          unit: "%",
          digits: 1,
          bars: [
            { label: "eager", value: 11.3 },
            { label: "graphs", value: 0.9, highlight: true },
          ],
        },
      },
      {
        date: "Oct 20",
        prs: [45, 47, 48],
        branch: "kernel/paged-attention-v2-impl",
        scene: "warps",
        title: "Attention with an online softmax",
        text: "One pass over the keys with a running maximum: scores no longer sit in shared memory, so contexts are no longer capped, and the values are summed by every thread instead of a few.",
        metric: {
          kind: "bars",
          label: "attention time per decode step",
          unit: "µs",
          digits: 0,
          bars: [
            { label: "v1", value: 883 },
            { label: "v2", value: 272, highlight: true },
          ],
        },
      },
      {
        date: "Oct 20",
        prs: [49],
        branch: "kernel/add-rmsnorm-tune",
        scene: "smem",
        title: "Tuning the fused norm",
        text: "128-bit loads and the row kept in registers between the sum of squares and the scaling.",
        metric: {
          kind: "bars",
          label: "add + RMSNorm time per decode step",
          unit: "µs",
          digits: 0,
          bars: [
            { label: "fused", value: 177 },
            { label: "tuned", value: 73, highlight: true },
          ],
        },
      },
      {
        date: "Oct 23",
        prs: [50, 51, 52, 53],
        branch: "runtime/graphs-split-attention",
        scene: "warps",
        title: "Splitting one long context",
        text: "At batch 1 each attention head walked its whole context in one thread block. Now a long context is cut across blocks and the partial softmaxes are merged, with one CUDA graph per split count.",
        metric: {
          kind: "bars",
          label: "ms per token at a 1,984-token prompt, batch 1, graphs",
          unit: "ms",
          digits: 1,
          bars: [
            { label: "v2", value: 14.0 },
            { label: "split", value: 9.1, highlight: true },
          ],
        },
      },
    ],
  },
  {
    id: "M5",
    name: "shipped",
    dates: "Oct 25",
    goal: "Measure it against the tools people actually use.",
    entries: [
      {
        date: "Oct 25",
        prs: [54],
        branch: "tools/llamacpp-baseline",
        title: "Against llama.cpp",
        text: "llama.cpp built from source for the same GPU, on BF16 GGUF weights and the same workload. The engine is ahead at every batch size.",
        metric: {
          kind: "bars",
          label: "output tokens per second at batch 16, current results",
          unit: "tok/s",
          digits: 0,
          bars: [
            { label: "HuggingFace", value: measured("hf", 16) },
            { label: "llama.cpp", value: measured("llamacpp", 16) },
            { label: "engine + graphs", value: measured("engine_graphs", 16), highlight: true },
          ],
        },
      },
      {
        date: "Oct 25",
        prs: [55],
        branch: "tools/roofline",
        scene: "roofline",
        title: "How close to the hardware",
        text: "A roofline from the measured peaks (323.6 GB/s, 34.5 TFLOP/s). At batch 1 the matrix-vector products run at 91 to 102% of the bandwidth roof; at batch 16 attention is what is left.",
        metric: { kind: "value", value: "82%", label: "of the bandwidth roof for a whole batch-1 step" },
      },
      {
        date: "Oct 25",
        prs: [58, 59],
        branch: "bench/chat-prompts",
        title: "A recording for this page",
        text: "A trace format for every scheduler step, and a run of 24 chat requests arriving four per second. It is the rain at the top of this page.",
      },
    ],
  },
  {
    id: "M6",
    name: "refinements",
    dates: "Nov 4 to 10",
    goal: "Refine how the engine serves requests now that it has shipped.",
    entries: [
      {
        date: "Nov 5",
        prs: [133, 134, 149],
        branch: "bench/chunked-prefill",
        scene: "chunked",
        title: "A token budget for every step",
        text: `Every step used to be all prefill or all decode, so one long prompt stopped every running sequence. Now a step has a token budget: decode rows first, the rest for prompts in chunks. Joe wrote the scheduler, and engine bench and engine serve take the budget as a flag. Under Poisson arrivals a budget of 256 lifts goodput at 6 requests per second from ${goodputAt("engine + graphs", "6").toFixed(2)} to ${goodputAt("engine + graphs, budget 256", "6").toFixed(2)}. vLLM, at ${goodputAt("vLLM", "6").toFixed(2)}, is still ahead.`,
        metric: {
          kind: "bars",
          label: "longest wait between two tokens of a request at 4 requests/s",
          unit: "ms",
          digits: 0,
          bars: [
            { label: "whole prompts", value: largestGap("engine + graphs") },
            { label: "budget 2,048", value: largestGap("engine + graphs, budget 2048") },
            { label: "budget 256", value: largestGap("engine + graphs, budget 256"), highlight: true },
          ],
        },
      },
      {
        date: "Nov 10",
        prs: [176, 183, 187, 190],
        branch: "runtime/graph-steps-default",
        title: "Graphs between the powers of two",
        text: `A CUDA Graph is captured for a fixed number of rows, so a step with ${paddedRows} sequences used to replay the graph for ${bucket("1")} and pay for the padding. Joe wrote decode_graph_buckets, and engine bench and engine serve take --graph-steps: at ${defaultSteps}, now the default, max batch ${sweepBatches[0]} gets graphs at ${graphBuckets(Number(sweepBatches[0]), defaultSteps).join(", ")} where it had only the powers of two. Padding ${paddedRows} rows to ${bucket("1")} cost about ${((100 * (padded.coarse - padded.fine)) / padded.fine).toFixed(0)}% per token against the graph for ${bucket(String(defaultSteps))} (calculated from the bars). Finer steps gained nothing.`,
        metric: {
          kind: "bars",
          label: `measured ms per token with ${paddedRows} sequences, by the graph they replay`,
          unit: "ms",
          digits: 2,
          bars: graphSteps.map((steps) => ({ label: `graph for ${bucket(steps)}`, value: padding(steps, "TPOT p50 (ms)"), highlight: steps === String(defaultSteps) })),
        },
      },
      {
        date: "Nov 10",
        prs: [187],
        branch: "runtime/graph-buckets-wiring",
        title: `Batch size past ${sweepBatches[0]}`,
        text: `Every earlier result stopped at max batch ${sweepBatches[0]}. A sweep up to ${sweepBatches[sweepBatches.length - 1]} shows where batching stops paying on this GPU: throughput with graphs climbs from ${whole(sweepTokens(sweepBatches[0]))} to ${whole(sweepTokens(knee))} tok/s at max batch ${knee}, then flattens to ${afterKnee.map((b) => whole(sweepTokens(b))).join(" and ")}. Past ${knee} the step time doubles with the rows: at ${afterKnee[0]} attention is ${tableValue(sweep, { "max batch": afterKnee[0] }, "attention share", 1, "results")}% of a step's GPU time and the GEMMs have turned compute bound.`,
        metric: {
          kind: "bars",
          label: "measured output tokens per second with graphs, by max batch",
          unit: "tok/s",
          digits: 0,
          bars: sweepBatches.map((b) => ({ label: `max batch ${b}`, value: sweepTokens(b), highlight: b === knee })),
        },
      },
      {
        date: "Nov 10",
        prs: [188],
        branch: "kernel/embedding-lookup",
        title: "A faster kernel that moved nothing",
        text: `The embedding lookup copies one row of a table per token. Claude rewrote it at Joe's request: one thread per ${lookup.bytesPerThread} bytes over all the rows at once, where the naive kernel gave every token its own block and copied ${lookup.bytesPerThread / lookup.valuesPerLoad} bytes at a time. The output is bit-identical and a launch is ${(lookup.decodeUs.naive / lookup.decodeUs.fast).toFixed(1)}x faster (calculated). No end-to-end number moved, because the lookup is under ${lookup.underPct}% of a step either way. It was never where the time went.`,
        metric: {
          kind: "bars",
          label: "measured median launch in decode steps",
          unit: "µs",
          digits: 1,
          bars: [
            { label: "naive", value: lookup.decodeUs.naive },
            { label: "fast::embedding", value: lookup.decodeUs.fast, highlight: true },
          ],
        },
      },
    ],
  },
];
