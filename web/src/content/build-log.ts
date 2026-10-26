import { medianOf } from "@/lib/data";

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
};

export type Milestone = { id: string; name: string; dates: string; goal: string; entries: Entry[] };

export const repoUrl = "https://github.com/joezhuo2/LLM-Inference-Engine";

function measured(system: string, batch: number) {
  return medianOf(system, batch, "output_throughput") ?? 0;
}

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
        title: "Measure the hardware first",
        text: "A bandwidth benchmark sets the number every later kernel is judged against: how fast this laptop GPU can read its own memory.",
        metric: { kind: "value", value: "323.6 GB/s", label: "measured DRAM bandwidth, RTX 5060 laptop" },
      },
      {
        date: "Oct 1",
        prs: [3],
        branch: "practice/reduction",
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
        title: "BF16 matrix multiply",
        text: "cuBLAS is column-major and the model is row-major, so the engine asks it for the transposed product instead of copying anything.",
      },
      {
        date: "Oct 9",
        prs: [10, 12, 14],
        branch: "loader/safetensors-parser",
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
        title: "First text out of the engine",
        text: "The forward pass, greedy decoding and engine generate, held to Hugging Face's logits on 20 reference prompts.",
      },
      {
        date: "Oct 10",
        prs: [23],
        branch: "sampling/temperature",
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
        title: "Profile before optimizing",
        text: "NVTX ranges and Nsight Systems on one decode step. Even the naive kernels read the weights at 61 to 72% of peak; the rest goes to idle gaps between hundreds of small launches.",
        metric: { kind: "value", value: "267", label: "kernel launches per decode step" },
      },
      {
        date: "Oct 16",
        prs: [39, 40, 42, 46],
        branch: "runtime/wire-fast-kernels",
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
];
