"use client";

import type { ReactNode } from "react";
import { createPortal } from "react-dom";
import { tensorAbout, tensorAboutCaption } from "@/components/kernels/tensor-parallelism/about";
import { ForwardFigure, LoadFigure, WireFigure } from "@/components/kernels/tensor-parallelism/figures";
import { TensorSandbox } from "@/components/kernels/tensor-parallelism/sandbox";
import { tensorCaptions, tensorSteps } from "@/components/kernels/tensor-parallelism/story";
import { chapterNumber } from "@/content/book-order";
import { attentionMatrices, measuredRanks, mib, mlpMatrices, mostRanks, rankChoices, rankShare, tensorParallelism, tpFmt, type TpMatrix, tpShape, whole } from "@/content/kernels/tensor-parallelism";
import type { Beat } from "../book-scene";
import { type ChapterDef, close, flat, type OverlayProps, overview, type WorkbenchPage, withLead } from "../chapter";
import { joinParts } from "./segments";
import { createTensorModel, type TpCut, type TpNumbers, type TpPiece, traysExhibit } from "./tensor-parallelism-model";

const fmt = tpFmt;
const s = tpShape;
const pieces = (n: number): TpPiece[] => {
  const r = rankShare(n);
  return [
    { value: r.embed, tone: "ink" },
    { value: r.layers, tone: "blue" },
    { value: r.head, tone: "ink" },
    { value: r.norms, tone: "ink" },
  ];
};
const heads = (n: number, r: number) => (s.heads / n === 1 ? `query head ${r}` : `query heads ${(r * s.heads) / n} to ${((r + 1) * s.heads) / n - 1}`);
const kv = (n: number, r: number) => (s.kv_heads / n === 1 ? `K/V head ${r}` : `K/V heads ${(r * s.kv_heads) / n} to ${((r + 1) * s.kv_heads) / n - 1}`);
const channels = (n: number, r: number) => `channels ${fmt((r * s.intermediate) / n)} to ${fmt(((r + 1) * s.intermediate) / n - 1)}`;
const cutOf = (m: TpMatrix) => ({ label: m.label, sub: `${fmt(m.rows)} × ${fmt(m.cols)}, cut by ${m.cut === "rows" ? "rows" : "columns"}`, cut: m.cut, bands: m.bands.map((b) => ({ label: b.label, share: b.rows / m.rows })) });
const n = mostRanks;
const stageRanks = 2;
const choices = `${rankChoices.slice(0, -1).join(", ")} or ${rankChoices[rankChoices.length - 1]}`;

const attention: TpCut = {
  title: `attention, cut by heads for ${n} ranks`,
  sub: "the rank on the rail lifts its rows of the fused QKV weight and its columns of wo; not to scale",
  ranks: n,
  matrices: attentionMatrices.map(cutOf),
  lines: Array.from({ length: n }, (_, r) => [`rank ${r} of ${n}: ${heads(n, r)}, ${kv(n, r)}`, `its rows of Q, K and V, and the same heads' columns of wo`, `wo gives a partial result, ${fmt(s.hidden)} numbers, to be summed`]),
};
const mlp: TpCut = {
  title: `the MLP, cut by channels for ${n} ranks`,
  sub: "the rank on the rail lifts its rows of the fused gate and up weight and its columns of wdown; not to scale",
  ranks: n,
  matrices: mlpMatrices.map(cutOf),
  lines: Array.from({ length: n }, (_, r) => [`rank ${r} of ${n}: ${channels(n, r)}`, "its rows of gate and of up, and the same channels' columns of wdown", "SwiGLU needs only its own channels; wdown gives a partial result"]),
};

const shown = measuredRanks.filter((r) => r.n > 0);
const numbers: TpNumbers = {
  full: rankShare(1).total,
  trays: rankChoices.map((k) => {
    const r = rankShare(k);
    return {
      n: k,
      parts: pieces(k),
      lane: `${mib(r.total)}, ${fmt(r.pct)}% of the model`,
      lines: [k === 1 ? "1 GPU: the whole model in one place" : `${k} GPUs: each holds ${fmt(r.pct)}% of the model`, k === 1 ? `${s.layers} layers, the embedding and the LM head` : `every layer cut in ${k}; the embedding and the LM head whole in each`, `${s.sums} sums per token: 2 in each of ${s.layers} layers${k === 1 ? ", needed only once there are several" : ""}`],
    };
  }),
  attention,
  mlp,
  stageRanks,
  stages: [
    { label: "RMSNorm", shared: true, lines: ["RMSNorm, on the whole vector", "its weights are whole in every rank", "it needs every number of the vector, so it is not cut"] },
    { label: "QKV and attention", shared: false, lines: ["each rank: its QKV GEMM, RoPE, attention", "over its own heads, with its own K/V cache", "no rank needs another rank's numbers here"] },
    { label: "wo", shared: false, lines: ["each rank: its columns of wo", "a partial result of the full width", "the right size, but only part of the answer"] },
    { label: "sum", shared: true, lines: ["the first sum: an all-reduce", "the partial results add into the residual stream", "now every rank has the whole vector again"] },
    { label: "RMSNorm", shared: true, lines: ["RMSNorm again, before the MLP", "whole weights, on the summed vector", "the same in every rank"] },
    { label: "gate, up and SwiGLU", shared: false, lines: ["each rank: its rows of gate and up", `${fmt(s.intermediate / stageRanks)} channels of ${fmt(s.intermediate)} each, then SwiGLU`, "one channel never needs another"] },
    { label: "wdown", shared: false, lines: ["each rank: its columns of wdown", "a second partial result of the full width", "to be summed like the first"] },
    { label: "sum", shared: true, lines: ["the second sum of the layer", "the next layer starts from the summed vector", `${s.sums} sums for all ${s.layers} layers`] },
  ],
  whole: { parts: pieces(1), text: `${fmt(whole.mib, 1)} MiB` },
  bytes: shown.map((r) => ({
    n: r.n,
    parts: pieces(r.n),
    label: r.n === 1 ? "rank 0 of 1: everything" : `rank 0 of ${r.n}`,
    text: `${fmt(r.mib, 1)} MiB, ${fmt(r.calc.pct)}%`,
    lines: [
      r.n === 1 ? "one rank: the whole model" : `a rank of ${r.n}: ${fmt(r.calc.pct)}% of the model, not ${fmt(100 / r.n)}%`,
      `layers ÷ ${r.n}: ${mib(r.calc.layers)}; embedding and LM head whole: ${mib(r.calc.embed + r.calc.head)}`,
      r.n === 1 ? `measured ${fmt(r.mib, 1)} MiB, the same as the whole model` : `${r.n} ranks together: ${fmt(r.calc.together, 2)} times the model; the pink post is 1/${r.n}`,
    ],
  })),
};

const beats: Beat[][] = [
  [
    overview("A layer's attention weights, cut by heads"),
    close("Close up: the fused QKV weight, Q, K and V cut alike", [0.3, 0.04, -0.3], 0.5, 0.36, -14, 40),
    close("From the side: wo, cut by columns", [0.72, 0.04, -0.3], 0.5, 0.36, 22, 30),
    close("The rail: drag to another rank", [0.5, 0.02, 0.2], 0.9, 0.34, 0, 46),
  ],
  [
    overview("A layer's MLP weights, cut by channels"),
    close("Close up: gate and up, cut alike", [0.3, 0.04, -0.3], 0.5, 0.36, -14, 40),
    close("From the side: wdown, cut by columns", [0.72, 0.04, -0.3], 0.5, 0.36, 22, 30),
    close("The rail: drag to another rank", [0.5, 0.02, 0.2], 0.9, 0.34, 0, 46),
  ],
  [
    overview("One layer, in order"),
    close("Close up: the two sums of a layer", [0.58, 0.04, -0.28], 0.8, 0.34, -12, 38),
    close("From the side: the work of each rank", [0.5, 0.04, -0.28], 0.7, 0.34, 24, 28),
    close("The rail: drag through the layer", [0.5, 0.02, 0.2], 0.9, 0.34, 0, 46),
  ],
  [flat("A forward pass on the shards, one layer")],
  [
    overview("One rank against the whole model"),
    close("Close up: the layers shrink, the tables stay", [0.5, 0.04, -0.3], 0.92, 0.34, -12, 40),
    close("From the side: 1/N, in pink", [0.4, 0.04, -0.2], 0.7, 0.3, 24, 28),
    close("The rail: drag to another number of ranks", [0.5, 0.02, 0.2], 0.9, 0.34, 0, 46),
  ],
  [flat("The load of one rank, before and after")],
  [flat("Where the sums happen")],
];

const instructions = [
  "Drag the pink ball to another rank: its rows of the QKV weight and its columns of wo rise.",
  "Drag the pink ball to another rank: its rows of gate and up and its columns of wdown rise.",
  "Drag the pink ball through the layer: blue blocks run in every rank, ink ones once, and the two sums close each half.",
  "Follow one layer from left to right: the ranks take turns, and the two sums are the only places they meet.",
  "Drag the pink ball to another number of ranks: the blue layers shrink, the ink tables do not.",
  "Compare the two bars of each layout: grey is one call per copy, blue one chunk per tensor.",
  "Count the sums a token needs, then find where they would cross between GPUs.",
];

const workbench: WorkbenchPage[] = [
  {
    title: "cut it yourself",
    overlay: "sandbox",
    caption: `Try it yourself: cut one layer for ${choices} ranks`,
    paragraphs: [
      "Pick how many ranks share the model and which rank to look at. The four matrices are the weights of one layer, and the blue rows and columns are what that rank loads, as plan_shard_layout() picks them.",
      "The bar below is one rank's bytes, calculated from TinyLlama's shapes: the embedding, its share of the layers and the LM head. The dashed line is 1/N of the model, the share it would hold if everything were cut.",
      `Go from 1 to ${mostRanks} ranks and watch the layers shrink while the two tables stay. There is no ${mostRanks * 2}: TinyLlama has ${s.kv_heads} K/V heads, and a rank needs at least one.`,
    ],
  },
];

const figures: Record<string, () => ReactNode> = {
  forward: () => <ForwardFigure />,
  load: () => <LoadFigure />,
  wire: () => <WireFigure />,
};

function TensorOverlay({ kind, target, narrow }: OverlayProps) {
  if (!target || !kind) return null;
  if (kind === "sandbox") return createPortal(<TensorSandbox />, target);
  const figure = figures[kind];
  if (!figure) return null;
  return createPortal(<div className={narrow ? "aspect-[560/400] w-full" : "h-[34rem]"}>{figure()}</div>, target);
}

const chapter: ChapterDef = {
  slug: "tensor-parallelism",
  number: chapterNumber("tensor-parallelism"),
  title: "tensor parallelism",
  formula: tensorParallelism.formula,
  steps: tensorSteps,
  captions: tensorCaptions,
  exhibits: [0, 1, 2, null, 3, null, null],
  overlays: [null, null, null, "forward", null, "load", "wire"],
  instructions,
  beats,
  workbench,
  buildModel: (kit, fonts) => createTensorModel(kit, fonts, numbers),
  Overlay: TensorOverlay,
};

export const tensorParallelismChapter = withLead(chapter, {
  step: tensorAbout,
  caption: tensorAboutCaption,
  instruction: "Drag the pink ball to give the model more GPUs: every row is one GPU, and its blue layers get shorter.",
  beats: [
    overview(`One model, on ${choices} GPUs`),
    close("Close up: the layers, cut in N", [0.5, 0.04, -0.3], 0.92, 0.4, -12, 38),
    close("From the side: the embedding and the LM head in every GPU", [0.3, 0.04, -0.28], 0.6, 0.36, 24, 26),
    close("The rail: drag to change the number of GPUs", [0.5, 0.02, 0.2], 0.9, 0.34, 0, 46),
  ],
  buildModel: (kit, fonts) => joinParts([traysExhibit(kit, fonts, numbers)]),
});
