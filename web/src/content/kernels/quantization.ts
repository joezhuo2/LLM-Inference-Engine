import { ceilings, hardware, modelShape } from "@/lib/data";
import { published } from "../published";
import { budget } from "./memory-budget";
import type { KernelStory } from "./types";

export const qFmt = (v: number, d = 0) => v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
export const qGb = (bytes: number, d = 2) => `${qFmt(bytes / 1e9, d)} GB`;
export const qMs = (ms: number, d = 2) => `${qFmt(ms, d)} ms`;

const shape = modelShape();
const peak = ceilings();
const hw = hardware();
const headDim = shape.hidden / shape.heads;
const fraction = budget.fractionPct / 100;

export const e4m3 = { sign: 1, exponent: 4, mantissa: 3, bias: 7, max: 448, minNormalExp: -6, minSubExp: -9, perPower: 8, topStep: 32 };
export const bf16Bits = { sign: 1, exponent: 8, mantissa: 7, perPower: 128 };
export const normalReach = e4m3.max * 2 ** -e4m3.minNormalExp;
export const zeroReach = e4m3.max * 2 ** (1 - e4m3.minSubExp);

export const widths = [
  { bits: 16, bytes: 2, name: "BF16", long: "BF16, 2 bytes a weight", tone: "ink" as const },
  { bits: 8, bytes: 1, name: "FP8 E4M3", long: "FP8 E4M3, 1 byte a weight", tone: "blue" as const },
  { bits: 4, bytes: 0.5, name: "4-bit", long: "4-bit, half a byte a weight", tone: "blue" as const },
];

export const matrices = [
  { id: "qkv", label: "QKV", rows: shape.hidden + 2 * shape.kv_heads * headDim, cols: shape.hidden, count: shape.layers },
  { id: "o_proj", label: "o_proj", rows: shape.hidden, cols: shape.hidden, count: shape.layers },
  { id: "gate_up", label: "gate/up", rows: 2 * shape.intermediate, cols: shape.hidden, count: shape.layers },
  { id: "down", label: "down", rows: shape.hidden, cols: shape.intermediate, count: shape.layers },
  { id: "lm_head", label: "LM head", rows: shape.vocab, cols: shape.hidden, count: 1 },
].map((m) => ({ ...m, weights: m.rows * m.cols * m.count }));

export const readWeights = matrices.reduce((s, m) => s + m.weights, 0);
export const floorMs = (bytes: number, gbs = peak.gbs) => (bytes / (gbs * 1e9)) * 1e3;
export const floors = widths.map((w) => ({ ...w, gb: (readWeights * w.bytes) / 1e9, ms: floorMs(readWeights * w.bytes) }));
export const [bf16Floor, fp8Floor, q4Floor] = floors;

export const ridge = (peak.tflops * 1e12) / (peak.gbs * 1e9);
export const rowsBig = budget.maxBatch;
export const intensity = (rows: number, bytes: number) => (2 * rows) / bytes;
export const mathUs = (rows: number) => ((2 * rows * readWeights) / (peak.tflops * 1e12)) * 1e6;
export const arithmetic = [1, rowsBig].flatMap((rows) => widths.map((w) => ({ rows, name: w.name, bytes: w.bytes, intensity: intensity(rows, w.bytes), pct: (intensity(rows, w.bytes) / ridge) * 100, readMs: floorMs(readWeights * w.bytes), mathUs: mathUs(rows) })));
export const mostIntensity = Math.max(...arithmetic.map((a) => a.intensity));
if (mostIntensity >= ridge) throw new Error("quantization: a width and batch now reach the ridge; rewrite the step that says the arithmetic is still free");

export const binade = Array.from({ length: e4m3.perPower }, (_, i) => ({ lo: 1 + i / e4m3.perPower, hi: 1 + (i + 1) / e4m3.perPower }));
export const fp8Error = 1 / (2 * e4m3.perPower);
export const bf16Error = 1 / (2 * bf16Bits.perPower);

const pow = (e: number) => (e >= 0 ? qFmt(2 ** e) : `2^${e}`);
export const ladder = [
  { label: "256 to 448", lo: 256, values: 7, step: `${e4m3.topStep}`, tone: "blue" as const },
  ...Array.from({ length: 8 - e4m3.minNormalExp }, (_, k) => {
    const e = 7 - k;
    return { label: `${pow(e)} to ${pow(e + 1)}`, lo: 2 ** e, values: e4m3.perPower, step: pow(e - 3), tone: "blue" as const };
  }),
  { label: `below 2^${e4m3.minNormalExp}`, lo: 0, values: 7, step: `2^${e4m3.minSubExp}`, tone: "ink" as const },
];

export const kvToken = { bf16: 2 * shape.layers * shape.kv_heads * headDim * 2, fp8: 2 * shape.layers * shape.kv_heads * headDim };
const tokensIn = (pool: number, perToken: number) => Math.max(0, Math.floor(pool / (perToken * budget.blockTokens)) - 1) * budget.blockTokens;
export const kvPool = { bytes: budget.pool, bf16: budget.poolTokens, fp8: tokensIn(budget.pool, kvToken.fp8), contextsBf16: budget.reservedFit, contextsFp8: Math.floor(tokensIn(budget.pool, kvToken.fp8) / budget.fullContext) };
if (tokensIn(budget.pool, kvToken.bf16) !== budget.poolTokens) throw new Error("quantization: the calculated BF16 pool no longer matches the engine's; check the reserved block in serving.ts");

export type QModel = { id: string; name: string; params: number; read: number; layers: number; kvHeads: number; headDim: number; source: string };
export const qModels: QModel[] = [
  { id: "tiny", name: "TinyLlama", params: budget.params, read: readWeights, layers: shape.layers, kvHeads: shape.kv_heads, headDim, source: "TinyLlama config.json" },
  ...[published.llama3, published.llama2].map((m) => ({ id: m.model, name: m.model, params: m.params, read: m.params, layers: m.layers, kvHeads: m.kvHeads, headDim: m.headDim, source: m.source })),
];

export type QGpu = { id: string; name: string; memory: number; rest: number; gbs: number; source: string };
const h = published.h100;
export const qGpus: QGpu[] = [
  { id: "here", name: "this GPU", memory: budget.vram, rest: budget.rest, gbs: peak.gbs, source: `${hw.gpu}: VRAM from hardware.md, bandwidth measured in results.md` },
  { id: "h100", name: h.model, memory: h.memoryGB * 1e9, rest: 0, gbs: h.bandwidthTBs * 1e3, source: h.source },
];

export function qFit(model: QModel, gpu: QGpu, bits: number, kvBytes: 1 | 2) {
  const weights = (model.params * bits) / 8;
  const room = gpu.memory - gpu.rest;
  const pool = Math.max(0, (room - weights) * fraction);
  const perToken = 2 * model.layers * model.kvHeads * model.headDim * kvBytes;
  const tokens = weights > room ? 0 : tokensIn(pool, perToken);
  return { weights, room, fits: weights <= room, over: Math.max(0, weights - room), pool, perToken, tokens, contexts: Math.floor(tokens / budget.fullContext), floorMs: floorMs((model.read * bits) / 8, gpu.gbs) };
}

export const sandboxBits = [16, 8, 4];

export const quantization: KernelStory = {
  slug: "quantization",
  index: 20,
  book: true,
  about: true,
  title: "quantization",
  short: `The next lever after the floor: store each weight in fewer bytes, and a decoded token reads fewer of them. FP8 would take TinyLlama's batch-1 floor from ${qMs(bf16Floor.ms)} to ${qMs(fp8Floor.ms)}. The plan and the arithmetic: the kernels are stubs, so nothing here is measured yet.`,
  formula: "floor per token = weights read × bytes per weight ÷ bandwidth",
  headline: {
    value: qMs(fp8Floor.ms),
    label: `the batch-1 floor with every weight in FP8, where BF16's is ${qMs(bf16Floor.ms)} (calculated, not measured: the kernels are stubs)`,
  },
  sandbox: {
    title: "fit it yourself",
    intro: "Pick the bits per weight, the bytes per cached value, a model and a GPU. Every number is calculated from the model's shape, the GPU's memory and bandwidth, and the engine's rule of a 90% pool; none of it has run.",
  },
  sources: [
    { path: "src/kernels/gemm_fp8.cu", label: "gemm_fp8: the spec of the FP8 weight product (a stub)" },
    { path: "src/kernels/fp8_kv.cu", label: "write_kv_fp8 and paged_attention_decode_fp8: the spec of the FP8 cache (stubs)" },
    { path: "include/engine/kernels/gemm_fp8.h", label: "the weight product's interface" },
    { path: "include/engine/kernels/fp8_kv.h", label: "the cache's write and read" },
    { path: "include/engine/core/tensor.h", label: "DType::F8E4M3, one byte per element" },
  ],
  tests: [
    { path: "tests/todo/gpu/test_gemm_fp8.cpp", label: "what gemm_fp8 must pass" },
    { path: "tests/todo/gpu/test_fp8_kv.cpp", label: "what the FP8 cache must pass" },
    { path: "tests/support/fp8.h", label: "the host's reference rounding" },
    { path: "tests/unit/test_tensor.cpp", label: "the new dtype's size and name" },
  ],
  testing: "",
  excerpts: [
    { file: "include/engine/kernels/gemm_fp8.h", from: "void gemm_fp8(", to: "Stream stream);", caption: "The weight product to write: BF16 activations, FP8 weights, one FP32 scale per output row" },
    { file: "include/engine/kernels/fp8_kv.h", from: "void write_kv_fp8(", to: "Stream stream);", caption: "The cache's write: the same tensors as write_kv, an FP8 cache and one scale for keys, one for values" },
  ],
  measurements: [],
  timeline: [],
};
