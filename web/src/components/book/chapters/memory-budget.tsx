"use client";

import type { ReactNode } from "react";
import { createPortal } from "react-dom";
import { memoryAbout, memoryAboutCaption } from "@/components/kernels/memory-budget/about";
import { BytesFigure } from "@/components/kernels/memory-budget/figures";
import { memoryCaptions, memorySteps } from "@/components/kernels/memory-budget/story";
import { chapterNumber } from "@/content/book-order";
import { batchRows, budget, fits, gb, memFmt, memoryBudget, parts } from "@/content/kernels/memory-budget";
import type { Beat } from "../book-scene";
import { type ChapterDef, close, flat, type OverlayProps, overview, withLead } from "../chapter";
import { createTankExhibit } from "./memory-budget-about-model";
import { createMemoryModel } from "./memory-budget-model";
import { joinParts } from "./segments";

const fmt = memFmt;
const slots = budget.reservedFit;

const beats: Beat[][] = [
  [flat("One token in the cache, factor by factor")],
  [
    overview("One pool, shared by every running sequence"),
    close("Close up: the slab under the budget line", [0.5, 0.1, -0.2], 0.9, 0.4, -14, 30),
    close("From the side: one sequence, in pink, at the slab's end", [0.2, 0.1, -0.18], 0.5, 0.36, 26, 24),
    close("The rail: drag to add sequences", [0.5, 0.02, 0.2], 0.9, 0.34, 0, 46),
  ],
  [
    overview("The same memory, four models"),
    close("Close up: weights against pool", [0.5, 0.03, -0.28], 0.96, 0.34, -12, 40),
    close("From the side: the contexts that still fit", [0.4, 0.03, -0.14], 0.7, 0.3, 24, 30),
    close("The rail: drag to change the model", [0.5, 0.02, 0.2], 0.9, 0.34, 0, 46),
  ],
];

const instructions = [
  "Multiply the five boxes, then follow the bars down to the pool.",
  "Drag the pink ball to add sequences: the slab gets wider, and past the bend it has to get lower.",
  "Drag the pink ball from model to model: the weights grow and the blue pool shrinks.",
];

const figures: Record<string, () => ReactNode> = {
  bytes: () => <BytesFigure />,
};

function MemoryOverlay({ kind, target, narrow }: OverlayProps) {
  if (!target || !kind) return null;
  const figure = figures[kind];
  if (!figure) return null;
  return createPortal(<div className={narrow ? "aspect-[560/400] w-full" : "h-[34rem]"}>{figure()}</div>, target);
}

const chapter: ChapterDef = {
  slug: "memory-budget",
  number: chapterNumber("memory-budget"),
  title: "the memory budget",
  formula: memoryBudget.formula,
  steps: memorySteps,
  captions: memoryCaptions,
  exhibits: [null, 0, 1],
  overlays: ["bytes", null, null],
  instructions,
  beats,
  workbench: [],
  buildModel: (kit, fonts) =>
    createMemoryModel(kit, fonts, {
      fullContext: budget.fullContext,
      poolTokens: budget.poolTokens,
      rows: batchRows.map((r) => ({
        batch: r.batch,
        context: r.context,
        lines: [
          `${fmt(r.batch)} ${r.batch === 1 ? "sequence, up to" : "sequences, each up to"} ${fmt(r.context)} tokens`,
          `${fmt(r.batch)} × ${fmt(r.context)} × ${fmt(budget.tokenBytes)} bytes = ${gb(r.bytes)}, ${fmt(r.poolPct, r.poolPct < 10 ? 1 : 0)}% of the pool`,
          r.bound === "positions" ? `the model's ${fmt(budget.fullContext)} positions stop them first; the rest of the pool stays free` : `the pool stops them: past ${budget.reservedFit} sequences a full context no longer fits`,
        ],
      })),
      vram: budget.vram,
      vramText: gb(budget.vram),
      rest: budget.rest,
      slots,
      fits: fits.map((f) => ({
        label: f.name,
        weights: f.weights,
        pool: f.pool,
        spare: f.spare,
        over: f.over > 0,
        contexts: f.contexts,
        lines: [
          `${f.name}: weights ${gb(f.weights, 1)}`,
          f.over > 0 ? `${gb(f.over, 1)} too large for this GPU: no pool, no request` : `pool ${gb(f.pool, 1)} at ${fmt(f.perToken)} bytes per token = ${fmt(f.tokens)} tokens`,
          f.over > 0 ? "calculated from the published shape; the pink block is not to scale" : `${fmt(f.contexts)} contexts of ${fmt(budget.fullContext)} tokens fit (${f.measured ? "pool measured" : "calculated"})`,
        ],
      })),
    }),
  Overlay: MemoryOverlay,
};

export const memoryBudgetChapter = withLead(chapter, {
  step: memoryAbout,
  caption: memoryAboutCaption,
  instruction: "Drag the pink ball to pick a part of the memory: its block rises, and the card says what it holds and whether it grows.",
  beats: [
    overview("The GPU's memory, and the four things in it"),
    close("Close up: the weights are fixed, the pool is what is shared", [0.5, 0.03, -0.14], 0.9, 0.36, -14, 40),
    close("From the side: the tenth that is kept free", [0.72, 0.03, -0.1], 0.5, 0.3, 24, 30),
    close("The rail: drag to pick a part", [0.5, 0.02, 0.2], 0.9, 0.34, 0, 46),
  ],
  buildModel: (kit, fonts) =>
    joinParts([
      createTankExhibit(kit, fonts, {
        vram: budget.vram,
        vramText: gb(budget.vram),
        sumText: `${fmt(parts.length)} parts`,
        parts: parts.map((p) => ({ label: p.label, value: p.bytes, headline: `${p.label}: ${gb(p.bytes)}, ${fmt((p.bytes / budget.vram) * 100)}% of the memory`, lines: [p.does, p.grows], tone: p.tone })),
      }),
    ]),
});
