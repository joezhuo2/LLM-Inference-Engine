"use client";

import type { ReactNode } from "react";
import { createPortal } from "react-dom";
import { flashAboutNumbers, flashPrefillAbout, flashPrefillAboutCaption } from "@/components/kernels/flash-prefill/about";
import { FlashBlocks, FlashFirst, FlashFragment, flashPrefillCaptions, flashPrefillSteps, FlashRegisters, FlashSpec, FlashSquare, FlashTiles, FlashTtft, FlashTurn, LengthPicker } from "@/components/kernels/flash-prefill/story";
import { chapterNumber } from "@/content/book-order";
import { flashPrefill, flashShape } from "@/content/kernels/flash-prefill";
import type { Beat } from "../book-scene";
import { type ChapterDef, close, flat, type OverlayProps, overview, withLead } from "../chapter";
import { createFlashPrefillAboutModel } from "./flash-prefill-about-model";
import { createFlashPrefillModel } from "./flash-prefill-model";

const s = flashShape;

const mmaBeats: Beat[] = [
  overview(`One mma call: ${s.tile} × ${s.tile} times ${s.tile} × ${s.half}`),
  close("Close up: the pieces of A and B one lane holds", [0.33, 0.03, -0.2], 0.66, 0.62, -18, 44),
  close("From the side: the scores rise with every call", [0.46, 0.05, -0.05], 0.5, 0.42, 28, 30),
  close("The rail: drag to pick a lane of the warp", [0.5, 0.02, 0.22], 0.9, 0.34, 0, 46),
];

type Spread = { beats: Beat[]; instruction: string; overlay: string | null; exhibit: number | null };

const spreads: Record<string, Spread> = {
  square: { beats: [flat("One walk per row against one pass per tile of rows")], instruction: "Compare the two triangles: every bar on the left is its own walk.", overlay: "square", exhibit: null },
  tiles: { beats: [flat(`One warp, ${s.tile} query rows, the keys ${s.tile} at a time`)], instruction: "Watch the key tiles pass the query tile, one after another.", overlay: "tiles", exhibit: null },
  mma: { beats: mmaBeats, instruction: "Drag the pink ball to pick a lane: its pieces of A and B move, and its scores rise with each call.", overlay: null, exhibit: 0 },
  fragment: { beats: [flat("Scores become weights where they are")], instruction: "Watch the same cells light up on both sides for each lane.", overlay: "fragment", exhibit: null },
  spec: { beats: [flat(`${s.tests} tests, written before any kernel`)], instruction: "Read the four groups: what must match, what must not change, what must not be read.", overlay: "spec", exhibit: null },
  first: { beats: [flat("The first kernel: the bytes moved, the walk did not")], instruction: "Watch the keys go by one at a time, then compare the two bars.", overlay: "first", exhibit: null },
  registers: { beats: [flat("What one lane keeps in registers")], instruction: "Watch which value each call takes from every group.", overlay: "registers", exhibit: null },
  blocks: { beats: [flat("How many thread blocks run at once")], instruction: "Compare the two grids: the pink blocks on the left had to wait.", overlay: "blocks", exhibit: null },
  result: { beats: [flat("Time to first token, four paths")], instruction: "Find the two lines that stay low: the tiled kernel and llama.cpp.", overlay: "ttft", exhibit: null },
  lengths: { beats: [flat("Pick a prompt length")], instruction: "Answer the prediction, then pick a prompt length.", overlay: "lengths", exhibit: null },
  open: { beats: [flat(`Your turn: ${s.group} heads, one read of the keys`)], instruction: "Answer the prediction first, then read where the kernel starts from.", overlay: "turn", exhibit: null },
};

const ordered = flashPrefillSteps.map((step) => {
  const spread = spreads[step.id];
  if (!spread) throw new Error(`flash-prefill: no book spread for step "${step.id}"`);
  return spread;
});

const figures: Record<string, () => ReactNode> = {
  square: () => <FlashSquare />,
  spec: () => <FlashSpec />,
  first: () => <FlashFirst />,
  tiles: () => <FlashTiles />,
  fragment: () => <FlashFragment />,
  registers: () => <FlashRegisters />,
  blocks: () => <FlashBlocks />,
  ttft: () => <FlashTtft />,
};

function FlashPrefillOverlay({ kind, target, narrow }: OverlayProps) {
  if (!target || !kind) return null;
  if (kind === "lengths")
    return createPortal(
      <div className={narrow ? undefined : "h-[34rem]"}>
        <LengthPicker />
      </div>,
      target,
    );
  if (kind === "turn")
    return createPortal(
      <div className={narrow ? undefined : "h-[34rem]"}>
        <FlashTurn />
      </div>,
      target,
    );
  const figure = figures[kind];
  if (!figure) return null;
  return createPortal(<div className={narrow ? "aspect-[560/400] w-full" : "h-[34rem]"}>{figure()}</div>, target);
}

const chapter: ChapterDef = {
  slug: "flash-prefill",
  number: chapterNumber("flash-prefill"),
  title: "FlashAttention prefill",
  formula: flashPrefill.formula,
  steps: flashPrefillSteps,
  captions: flashPrefillCaptions,
  exhibits: ordered.map((spread) => spread.exhibit),
  overlays: ordered.map((spread) => spread.overlay),
  instructions: ordered.map((spread) => spread.instruction),
  beats: ordered.map((spread) => spread.beats),
  workbench: [],
  buildModel: (kit, fonts) => createFlashPrefillModel(kit, fonts, { tile: s.tile, half: s.half, calls: s.headDim / s.tile, headDim: s.headDim, lanes: s.lanes }),
  Overlay: FlashPrefillOverlay,
};

export const flashPrefillChapter = withLead(chapter, {
  step: flashPrefillAbout,
  caption: flashPrefillAboutCaption,
  instruction: "Drag the pink ball to pick a query tile: its key tiles pass one at a time, and only the tall one exists as scores.",
  beats: [
    overview("The scores of a prompt, one tile at a time"),
    close("Close up: one tile of scores stands, the ones before it are done", [0.36, 0.04, -0.2], 0.6, 0.5, -18, 40),
    close("From the side: the pink key tile moves along the top", [0.36, 0.04, -0.24], 0.62, 0.5, 26, 30),
    close("The rail: drag to pick a query tile", [0.5, 0.02, 0.2], 0.9, 0.34, 0, 46),
  ],
  buildModel: (kit, fonts) => createFlashPrefillAboutModel(kit, fonts, flashAboutNumbers),
});
