import { hardware, kvReplay, modelShape } from "@/lib/data";
import { published } from "../published";
import { blockBytes, blockTokens, fullContext, fullMb, maxBatch, poolBlocks, poolFraction, poolTokens, reservedFit, sweepLast, tokenBytes } from "./serving";
import type { KernelStory } from "./types";

export const memFmt = (v: number, d = 0) => v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
export const gb = (bytes: number, d = 2) => `${memFmt(bytes / 1e9, d)} GB`;

const shape = modelShape();
const hw = hardware();
const headDim = shape.hidden / shape.heads;
const layerParams = shape.hidden * (shape.hidden + 2 * shape.kv_heads * headDim) + shape.hidden * shape.hidden + shape.hidden * 2 * shape.intermediate + shape.intermediate * shape.hidden;
const params = shape.layers * layerParams + 2 * shape.vocab * shape.hidden;
const fraction = poolFraction / 100;

const vram = hw.vram_gb * 1e9;
const weights = params * 2;
const pool = poolBlocks * blockBytes;
const free = pool / fraction;
const rest = vram - weights - free;
if (rest < 0) throw new Error("the recorded KV pool is larger than the VRAM left after the weights: check docs/hardware.md and the kv replay");

export const budget = {
  gpu: hw.gpu,
  vram,
  weights,
  params,
  pool,
  free,
  spare: free - pool,
  rest,
  fractionPct: poolFraction,
  poolBlocks,
  poolTokens,
  blockTokens,
  blockBytes,
  tokenBytes,
  fullContext,
  fullMb,
  reservedFit,
  maxBatch,
  layers: shape.layers,
  kvHeads: shape.kv_heads,
  heads: shape.heads,
  headDim,
  poolDate: kvReplay().meta.date.slice(0, 10),
  weightsPct: (weights / vram) * 100,
  poolPct: (pool / vram) * 100,
  wideTokenBytes: 2 * shape.layers * shape.heads * headDim * 2,
};

export const tokenFactors = [
  { value: shape.layers, label: "layers", note: "every layer keeps its own" },
  { value: 2, label: "K and V", note: "a key and a value" },
  { value: shape.kv_heads, label: "KV heads", note: `${shape.heads} query heads share ${shape.kv_heads}` },
  { value: headDim, label: "numbers per head", note: "the head dimension" },
  { value: 2, label: "bytes", note: "BF16" },
];

export const parts = [
  { id: "weights", label: "weights", bytes: weights, tone: "ink" as const, does: `${memFmt(params / 1e9, 2)} billion parameters at 2 bytes each, loaded once`, grows: "fixed: it does not change with the batch or the context" },
  { id: "pool", label: "KV pool", bytes: pool, tone: "blue" as const, does: `${memFmt(poolBlocks)} blocks of ${blockTokens} tokens, allocated once at start`, grows: "this is the part that batch size and context length share" },
  { id: "spare", label: "kept free", bytes: free - pool, tone: "glass" as const, does: `the engine takes ${poolFraction}% of what is free and leaves the rest`, grows: "room for the buffers the forward pass allocates after the pool" },
  { id: "rest", label: "everything else", bytes: rest, tone: "ink" as const, does: "the desktop this laptop draws, the CUDA context, the engine's own buffers", grows: "calculated as the remainder, not measured part by part" },
];

export const contextAt = (batch: number) => Math.min(fullContext, Math.floor(poolTokens / batch / blockTokens) * blockTokens);
export const batchStops = [...new Set([1, maxBatch, 32, 64, reservedFit, 128, sweepLast.batch])].sort((a, b) => a - b);
export const batchRows = batchStops.map((batch) => {
  const context = contextAt(batch);
  return { batch, context, bytes: batch * context * tokenBytes, poolPct: ((batch * context) / poolTokens) * 100, bound: batch <= reservedFit ? ("positions" as const) : ("pool" as const) };
});
export const lastRow = batchRows[batchRows.length - 1];
export const sweepPeak = { batch: sweepLast.batch, blocks: sweepLast.peak, pct: (sweepLast.peak / poolBlocks) * 100 };

type Spec = { model: string; params: number; layers: number; kvHeads: number; headDim: number; bits: number; measured: boolean };
const l3 = published.llama3;
const l2 = published.llama2;
const specs: Spec[] = [
  { model: "TinyLlama", params, layers: shape.layers, kvHeads: shape.kv_heads, headDim, bits: 16, measured: true },
  { model: l3.model, params: l3.params, layers: l3.layers, kvHeads: l3.kvHeads, headDim: l3.headDim, bits: 16, measured: false },
  { model: l3.model, params: l3.params, layers: l3.layers, kvHeads: l3.kvHeads, headDim: l3.headDim, bits: 4, measured: false },
  { model: l2.model, params: l2.params, layers: l2.layers, kvHeads: l2.kvHeads, headDim: l2.headDim, bits: 4, measured: false },
];

export const fits = specs.map((s) => {
  const w = (s.params * s.bits) / 8;
  const left = vram - rest - w;
  const perToken = 2 * s.layers * s.kvHeads * s.headDim * 2;
  const p = s.measured ? pool : Math.max(0, left * fraction);
  const tokens = s.measured ? poolTokens : Math.floor(p / perToken / blockTokens) * blockTokens;
  return { name: `${s.model}, ${s.bits === 16 ? "BF16" : `${s.bits}-bit`}`, model: s.model, bits: s.bits, measured: s.measured, weights: w, perToken, pool: p, spare: Math.max(0, left) - p, over: Math.max(0, -left), tokens, contexts: Math.floor(tokens / fullContext) };
});
export const fitOf = (model: string, bits: number) => {
  const f = fits.find((x) => x.model === model && x.bits === bits);
  if (!f) throw new Error(`no fit for ${model} at ${bits} bits`);
  return f;
};

export const memoryBudget: KernelStory = {
  slug: "memory-budget",
  index: 16,
  book: true,
  about: true,
  title: "the memory budget",
  short: `What the GPU's ${memFmt(hw.vram_gb, 2)} GB has to hold: the weights, which are fixed, and the KV cache, which batch size and context length share. With what a larger model does to both.`,
  formula: `batch × context × bytes per token ≤ ${fraction} × (VRAM − weights − the rest)`,
  headline: {
    value: memFmt(reservedFit),
    label: `full ${memFmt(fullContext)}-token contexts fit in the KV pool beside the weights on this GPU: for TinyLlama, memory is not what limits the batch`,
  },
  sources: [
    { path: "src/kv/cache_size.cpp", label: "bytes per block, blocks in the pool" },
    { path: "include/engine/kv/cache_size.h", label: "block size 16, fraction 0.9" },
    { path: "src/runtime/kv_cache.cpp", label: "free memory, the pool's allocation" },
    { path: "src/kv/block_manager.cpp", label: "who holds which block" },
    { path: "docs/hardware.md", label: "the GPU's memory" },
  ],
  tests: [{ path: "tests/unit/test_kv_cache_size.cpp", label: "bytes per block and the pool size" }],
  testing: "",
  excerpts: [
    {
      file: "src/kv/cache_size.cpp",
      from: "int64_t kv_block_bytes(const ModelConfig& config, int block_size) {",
      to: "    return int(std::min<int64_t>(blocks, INT_MAX));",
      caption: "The whole budget in two functions: the bytes of one block, and how many blocks a fraction of the free memory buys",
    },
  ],
  measurements: [
    { doc: "results", section: "batch-size-past-16", table: 0, caption: "max batch 16 to 256 with short requests: the most KV blocks ever held, against the pool", command: "scripts/batch_sweep.sh, then scripts/batch_report.py" },
  ],
  timeline: [],
};
