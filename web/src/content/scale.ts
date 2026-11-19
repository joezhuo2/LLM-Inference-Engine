import { ceilings, hardware, kvReplay, modelShape } from "@/lib/data";
import { blockTokens, poolFraction } from "./kernels/serving";
import { published } from "./published";

export const bytesPerWeight = 2;
export const kvFraction = poolFraction / 100;

export const weightBytes = (params: number) => params * bytesPerWeight;
export const readBytes = (params: number, vocab: number, hidden: number) => (params - vocab * hidden) * bytesPerWeight;
export const readFloorMs = (bytes: number, bytesPerSecond: number) => (bytes / bytesPerSecond) * 1e3;
export const ridgeOf = (flopsPerSecond: number, bytesPerSecond: number) => flopsPerSecond / bytesPerSecond;
export const kvTokenBytes = (layers: number, kvHeads: number, headDim: number) => 2 * layers * kvHeads * headDim * 2;
export const poolBlocksFor = (freeBytes: number, tokenBytes: number) => Math.floor((freeBytes * kvFraction) / (tokenBytes * blockTokens));
export const poolTokensOf = (blocks: number) => (blocks - 1) * blockTokens;
export const contextBlocksOf = (context: number) => Math.ceil(context / blockTokens);
export const fullContextsIn = (blocks: number, context: number) => Math.floor((blocks - 1) / contextBlocksOf(context));

export type ScaleSide = {
  model: string;
  gpu: string;
  name: string;
  params: number;
  weightBytes: number;
  readBytes: number;
  bandwidth: number;
  flops: number;
  readFloorMs: number;
  ridge: number;
  kvTokenBytes: number;
  blockBytes: number;
  poolBlocks: number;
  poolBytes: number;
  poolTokens: number;
  context: number;
  contextBlocks: number;
  contextBytes: number;
  fullContexts: number;
};

function side(s: { model: string; gpu: string; params: number; vocab: number; hidden: number; layers: number; kvHeads: number; headDim: number; bandwidth: number; flops: number; context: number; poolBlocks: (tokenBytes: number, weights: number) => number }): ScaleSide {
  const weights = weightBytes(s.params);
  const read = readBytes(s.params, s.vocab, s.hidden);
  const token = kvTokenBytes(s.layers, s.kvHeads, s.headDim);
  const blocks = s.poolBlocks(token, weights);
  return {
    model: s.model,
    gpu: s.gpu,
    name: `${s.model} on the ${s.gpu}`,
    params: s.params,
    weightBytes: weights,
    readBytes: read,
    bandwidth: s.bandwidth,
    flops: s.flops,
    readFloorMs: readFloorMs(read, s.bandwidth),
    ridge: ridgeOf(s.flops, s.bandwidth),
    kvTokenBytes: token,
    blockBytes: token * blockTokens,
    poolBlocks: blocks,
    poolBytes: blocks * token * blockTokens,
    poolTokens: poolTokensOf(blocks),
    context: s.context,
    contextBlocks: contextBlocksOf(s.context),
    contextBytes: contextBlocksOf(s.context) * token * blockTokens,
    fullContexts: fullContextsIn(blocks, s.context),
  };
}

const shape = modelShape();
const headDim = shape.hidden / shape.heads;
const layerParams = shape.hidden * (shape.hidden + 2 * shape.kv_heads * headDim) + shape.hidden * shape.hidden + shape.hidden * 2 * shape.intermediate + shape.intermediate * shape.hidden;
const peak = ceilings();
const l3 = published.llama3;
const h = published.h100;

export const engineSide = side({
  model: "TinyLlama",
  gpu: hardware().gpu.replace("NVIDIA GeForce ", ""),
  params: shape.layers * layerParams + 2 * shape.vocab * shape.hidden,
  vocab: shape.vocab,
  hidden: shape.hidden,
  layers: shape.layers,
  kvHeads: shape.kv_heads,
  headDim,
  bandwidth: peak.gbs * 1e9,
  flops: peak.tflops * 1e12,
  context: shape.max_positions,
  poolBlocks: () => kvReplay().meta.kv_blocks,
});

export const bigSide = side({
  model: l3.model,
  gpu: h.model,
  params: l3.params,
  vocab: l3.vocab,
  hidden: l3.hidden,
  layers: l3.layers,
  kvHeads: l3.kvHeads,
  headDim: l3.headDim,
  bandwidth: h.bandwidthTBs * 1e12,
  flops: h.bf16TFlops * 1e12,
  context: l3.context,
  poolBlocks: (token, weights) => poolBlocksFor(h.memoryGB * 1e9 - weights, token),
});

export const bigFreeBytes = h.memoryGB * 1e9 - bigSide.weightBytes;

export const scaleSources = {
  engine: `this engine: TinyLlama's shape from its config.json, the KV pool of ${kvReplay().meta.kv_blocks.toLocaleString("en-US")} blocks from the recorded run, bandwidth and compute ceilings measured on the ${engineSide.gpu} (results.md, Environment)`,
  big: `${l3.source}; ${l3.shapeSource}; ${h.source}`,
  method: `calculated: weights at ${bytesPerWeight} bytes each; the KV pool is ${poolFraction}% of the memory left after the weights, as kKvCacheFraction does, ignoring the CUDA context and the engine's buffers; nothing was measured on an ${h.model} or with ${l3.model}`,
};

const int = (v: number) => v.toLocaleString("en-US", { maximumFractionDigits: 0 });
const dec = (v: number, d: number) => v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });

export type ScaleQuantity = { label: string; value: (s: ScaleSide) => number; text: (v: number) => string };

export const scaleQuantities = {
  weights: { label: "weights in BF16", value: (s) => s.weightBytes, text: (v) => `${dec(v / 1e9, 2)} GB` },
  readFloor: { label: "read floor of one decode step, weights ÷ bandwidth", value: (s) => s.readFloorMs, text: (v) => `${dec(v, 2)} ms` },
  ridge: { label: "ridge, compute ÷ bandwidth", value: (s) => s.ridge, text: (v) => `${int(v)} FLOP per byte` },
  kvToken: { label: "KV cache per token", value: (s) => s.kvTokenBytes, text: (v) => `${int(v)} bytes` },
  pool: { label: "KV pool beside the weights", value: (s) => s.poolBytes, text: (v) => `${dec(v / 1e9, 2)} GB` },
  poolTokens: { label: "tokens the KV pool holds", value: (s) => s.poolTokens, text: (v) => int(v) },
  context: { label: "one full context", value: (s) => s.contextBytes, text: (v) => `${dec(v / 1e6, 1)} MB` },
  fullContexts: { label: "full contexts that fit in the pool", value: (s) => s.fullContexts, text: (v) => int(v) },
} satisfies Record<string, ScaleQuantity>;

export type ScaleId = keyof typeof scaleQuantities;
