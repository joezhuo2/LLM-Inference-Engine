"use client";

import { createPortal } from "react-dom";
import { quantAbout, quantAboutCaption } from "@/components/kernels/quantization/about";
import { QuantSandbox } from "@/components/kernels/quantization/sandbox";
import { quantCaptions, quantSteps } from "@/components/kernels/quantization/story";
import { chapterNumber } from "@/content/book-order";
import { arithmetic, bf16Bits, binade, e4m3, floorMs, floors, kvPool, kvToken, ladder, matrices, normalReach, qFmt, qGb, qMs, quantization, ridge } from "@/content/kernels/quantization";
import type { Beat } from "../book-scene";
import { type ChapterDef, close, type OverlayProps, overview, type WorkbenchPage, withLead } from "../chapter";
import { createQuantModel, type QuantNumbers, setExhibit } from "./quantization-model";
import { joinParts } from "./segments";

const fmt = qFmt;
const num = (v: number) => fmt(v, 3).replace(/0+$/, "").replace(/\.$/, "");
const rowsText = (rows: number) => `${rows} ${rows === 1 ? "row" : "rows"}`;

const setBeats = (what: string, side: string, rail: string): Beat[] => [
  overview(what),
  close(`Close up: ${side}`, [0.5, 0.03, -0.14], 0.9, 0.36, -14, 40),
  close("From the side: the picked block, risen in pink", [0.72, 0.03, -0.1], 0.5, 0.3, 24, 30),
  close(`The rail: ${rail}`, [0.5, 0.02, 0.2], 0.9, 0.34, 0, 46),
];

const beats: Beat[][] = [
  setBeats("The bytes a token reads, at three widths", "BF16, FP8 and 4-bit, to scale", "drag to pick a matrix"),
  [
    overview("The work each byte does, against the ridge"),
    close("Close up: the blue bar and the ridge above it", [0.5, 0.03, -0.22], 0.9, 0.34, -12, 40),
    close("From the side: how far each bar stops short", [0.3, 0.03, -0.18], 0.55, 0.3, 24, 28),
    close("The rail: drag to change the width and the batch", [0.5, 0.02, 0.2], 0.9, 0.34, 0, 46),
  ],
  setBeats("E4M3's ladder, one cell per power of two", "the full-precision cells under the glass", "drag to pick a power of two"),
  setBeats("Between 1 and 2: BF16's 128 steps over FP8's 8", "the comb of BF16 over the eight FP8 steps", "drag to pick a step"),
  setBeats("The same pool, two widths of cache", "the glass pool over the tokens it holds", "drag to switch the width"),
  setBeats("The plan, piece by piece", "three stubs under one bar", "drag to pick a piece"),
];

const instructions = [
  "Drag the pink ball along the matrices: each one rises, and the card gives its bytes and its share of the floor.",
  "Drag the pink ball to change the width and the batch: the blue bar grows, and it never reaches the ridge.",
  "Drag the pink ball down the ladder: each cell is a power of two, and the card says how finely E4M3 cuts it.",
  "Drag the pink ball from step to step: a weight inside a step rounds to its nearer end.",
  "Drag the pink ball between the two widths: the same bytes hold twice the tokens in FP8.",
  "Drag the pink ball to a piece of the plan: the card says what it must promise.",
];

const fp8 = (w: number) => qMs(floorMs(w));
const numbers: QuantNumbers = {
  floor: {
    title: "the weights a decoded token reads",
    sub: "to scale in bytes; the floor is bytes ÷ the measured bandwidth (calculated)",
    refs: floors.map((f, i) => ({ label: f.long, text: `${qGb(f.gb * 1e9)} · ${qMs(f.ms)}`, value: f.gb, tone: i === 0 ? ("ink" as const) : ("blue" as const) })),
    row: { label: "at 1 byte, matrix by matrix", text: "pick one" },
    segments: matrices.map((m) => ({
      label: m.label,
      value: m.weights / 1e9,
      headline: `${m.label}: ${qGb(m.weights)} at 1 byte, ${fp8(m.weights)}`,
      lines: [`${m.count === 1 ? "once" : `${m.count} layers ×`} ${fmt(m.rows)} × ${fmt(m.cols)} weights`, `BF16: ${fp8(2 * m.weights)}; 4-bit: ${fp8(m.weights / 2)} of the floor`],
    })),
    rail: "which matrix",
    rest: 2,
  },
  ridge: {
    value: ridge,
    text: `${fmt(ridge)} FLOP per byte`,
    stops: arithmetic.map((a) => ({
      label: `${a.name}, ${rowsText(a.rows)}`,
      intensity: a.intensity,
      lines: [`${a.name}, ${rowsText(a.rows)}: ${fmt(a.intensity)} FLOP per byte`, `${fmt(a.pct, a.pct < 10 ? 1 : 0)}% of the ridge: the read still sets the time`, `math ${fmt(a.mathUs)} µs at the peak, reading ${qMs(a.readMs)}`],
    })),
  },
  ladder: {
    title: "what E4M3 can hold, from 448 down",
    sub: "one cell per power of two, not to scale in value; from the format's definition",
    refs: [{ label: `full precision: ${e4m3.perPower} values per power of two`, text: `${e4m3.max} down to 2^${e4m3.minNormalExp}`, value: ladder.length - 1, tone: "glass" }],
    row: { label: "E4M3, one cell per power of two", text: "pick one" },
    segments: ladder.map((l, i) => ({
      label: l.label,
      value: 1,
      tone: l.tone,
      headline: `${l.label}: ${l.values} values, ${l.step} apart`,
      lines:
        i === 0
          ? [`a row's largest weight ÷ its scale lands on ${e4m3.max}`, `anything larger is held at ${e4m3.max}: there is no infinity`]
          : i === ladder.length - 1
            ? [`coarse: ${l.step} apart whatever the size`, `under 2^${e4m3.minSubExp - 1} a weight becomes 0`]
            : ["a weight here keeps 3 mantissa bits", `the coarse cells start ${fmt(normalReach)} times below ${e4m3.max}`],
    })),
    rail: "which power of two",
    rest: 0,
  },
  round: {
    title: "between 1 and 2: the values each format holds",
    sub: "the comb is BF16's 128 steps, the row FP8's 8; to scale (calculated)",
    refs: [{ label: "BF16 between 1 and 2", text: `${bf16Bits.perPower} steps of 1/${bf16Bits.perPower}`, value: 1, tone: "ink", count: bf16Bits.perPower }],
    row: { label: "FP8 E4M3 between 1 and 2", text: `${e4m3.perPower} steps of 1/${e4m3.perPower}` },
    segments: binade.map((b) => ({
      label: num(b.lo),
      value: 1 / e4m3.perPower,
      tone: "blue" as const,
      headline: `from ${num(b.lo)} to ${num(b.hi)}: round to the nearer end`,
      lines: [`at most ${num(1 / (2 * e4m3.perPower))} away, 1/${2 * e4m3.perPower} of the power of two`, `BF16 has ${bf16Bits.perPower / e4m3.perPower} steps in here: at most 1/${2 * bf16Bits.perPower} away`],
    })),
    rail: "which step",
    rest: 2,
  },
  pool: {
    title: "the same pool, two widths of cache",
    sub: "to scale in tokens; the pool as the engine allocated it (measured), the FP8 count calculated",
    refs: [{ label: "the pool, the same bytes either way", text: qGb(kvPool.bytes), value: kvPool.fp8, tone: "glass" }],
    row: { label: "the tokens it holds", text: `${fmt(kvPool.fp8)} in FP8` },
    segments: [
      { label: "BF16", value: kvPool.bf16, tone: "ink", headline: `BF16: ${fmt(kvPool.bf16)} tokens, ${fmt(kvPool.contextsBf16)} full contexts`, lines: [`${fmt(kvToken.bf16)} bytes a token`, "the pool the engine allocates today (measured)"] },
      { label: "what FP8 adds", value: kvPool.fp8 - kvPool.bf16, tone: "blue", headline: `FP8: ${fmt(kvPool.fp8)} tokens, ${fmt(kvPool.contextsFp8)} full contexts`, lines: [`${fmt(kvToken.fp8)} bytes a token, twice the blocks`, "calculated: write_kv_fp8 is a stub"] },
    ],
    rail: "which width",
    rest: 1,
  },
  plan: {
    title: "the plan, piece by piece",
    sub: "from the specs in src/kernels; equal widths, illustrative",
    refs: [{ label: "stubs that throw, each with a spec and tests", text: "#126, #104", value: 3, tone: "blue", count: 3 }],
    row: { label: "the pieces of FP8 in the engine", text: "pick one" },
    segments: [
      { label: "write_kv_fp8", value: 1, tone: "blue", headline: "write_kv_fp8: new keys and values in", lines: [`byte = e4m3(x ÷ scale), nearest, held at ±${e4m3.max}`, "one scale for all keys, one for all values"] },
      { label: "paged_attention_decode_fp8", value: 1, tone: "blue", headline: "paged_attention_decode_fp8: read back", lines: ["value = byte × scale, exact in FP32", "then attention, to the paged kernels' tolerance"] },
      { label: "gemm_fp8", value: 1, tone: "blue", headline: "gemm_fp8: BF16 activations, FP8 weights", lines: ["y = bf16(row_scale × Σ x · w), summed in FP32", "one scale per output row"] },
      { label: "DType::F8E4M3", value: 1, tone: "ink", headline: "DType::F8E4M3: one byte per element", lines: ["named F8_E4M3, as safetensors spells it", "done and tested: the one piece that is"] },
      { label: "the wiring", value: 1, tone: "glass", headline: "the wiring: not started", lines: ["scales written at load, a flag, the cache's dtype", "then measured: bytes, step time, logit error"] },
    ],
    rail: "which piece",
    rest: 2,
  },
};

const workbench: WorkbenchPage[] = [
  {
    title: "fit it yourself",
    overlay: "sandbox",
    caption: "Try it yourself: bits per weight, a model and a GPU",
    paragraphs: [
      "Answer the question at the top first, then use the four switches: the bits per weight, the width of the KV cache, the model and the GPU. The bar is the GPU's memory, and the numbers under it are what that choice leaves you.",
      "Start with TinyLlama on this GPU and step from 16 to 8 to 4 bits. The weights halve each time and so does the floor per token, while the pool grows into the room they leave. Then switch the cache to FP8 and watch the tokens double with nothing else moving.",
      "Then pick a published model. On this GPU it does not fit in BF16, and the readout says by how much. On the H100 everything fits, and the question becomes how many tokens the pool holds at each width.",
      "All of it is arithmetic on published shapes and the engine's own rule of a 90% pool. None of it has run: the FP8 kernels are stubs, and no 4-bit kernel exists in this engine at all.",
    ],
  },
];

function QuantOverlay({ kind, target }: OverlayProps) {
  if (!target || kind !== "sandbox") return null;
  return createPortal(<QuantSandbox />, target);
}

const chapter: ChapterDef = {
  slug: "quantization",
  number: chapterNumber("quantization"),
  title: "quantization",
  formula: quantization.formula,
  steps: quantSteps,
  captions: quantCaptions,
  exhibits: [0, 1, 2, 3, 4, 5],
  overlays: [null, null, null, null, null, null],
  instructions,
  beats,
  workbench,
  buildModel: (kit, fonts) => createQuantModel(kit, fonts, numbers),
  Overlay: QuantOverlay,
};

export const quantizationChapter = withLead(chapter, {
  step: quantAbout,
  caption: quantAboutCaption,
  instruction: "Drag the pink ball to pick a field of the FP8 byte: it rises, and the card says what its bits do.",
  beats: setBeats("One weight in bits: BF16, FP8 and 4-bit", "the 16 bits of BF16 over the 8 of FP8", "drag to pick a field"),
  buildModel: (kit, fonts) =>
    joinParts([
      setExhibit(kit, fonts, {
        title: "one weight, in bits",
        sub: "BF16 today, the engine's FP8 plan, a GGUF 4-bit weight; to scale in bits",
        refs: [
          { label: `BF16: ${bf16Bits.sign} sign, ${bf16Bits.exponent} exponent, ${bf16Bits.mantissa} mantissa`, text: "16 bits", value: 16, tone: "glass", count: 16 },
          { label: "4-bit, as in a GGUF Q4 file", text: "4 bits and a block scale", value: 4, tone: "ink", count: 4 },
        ],
        row: { label: "FP8 E4M3: pick a field", text: "8 bits" },
        segments: [
          { label: "sign", value: e4m3.sign, tone: "ink", headline: "sign: 1 bit, plus or minus", lines: ["the same single bit as in BF16", "zero comes in two signs, +0 and −0"] },
          { label: "exponent", value: e4m3.exponent, tone: "blue", headline: `exponent: ${e4m3.exponent} bits, which power of two`, lines: [`2^${e4m3.minNormalExp} to 2^8, and a few small values below`, `BF16 spends ${bf16Bits.exponent} bits here, the range of FP32`] },
          { label: "mantissa", value: e4m3.mantissa, tone: "ink", headline: `mantissa: ${e4m3.mantissa} bits, where inside it`, lines: [`${e4m3.perPower} steps in each power of two: 1, 1.125 … 1.875`, `BF16 spends ${bf16Bits.mantissa} bits here: ${bf16Bits.perPower} steps`] },
        ],
        rail: "which field",
        rest: 2,
      }),
    ]),
});
