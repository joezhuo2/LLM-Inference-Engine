"use client";

import type { ReactNode } from "react";
import { createPortal } from "react-dom";
import { flashAboutNumbers, flashPrefillAbout, flashPrefillAboutCaption } from "@/components/kernels/flash-prefill/about";
import { FlashFirst, FlashFragment, FlashOpen, flashPrefillCaptions, flashPrefillSteps, FlashRegisters, FlashSpec, FlashSquare, FlashTiles, FlashTtft, LengthPicker } from "@/components/kernels/flash-prefill/story";
import { chapterNumber } from "@/content/book-order";
import { flashPrefill, flashShape } from "@/content/kernels/flash-prefill";
import type { Beat } from "../book-scene";
import { type ChapterDef, close, flat, type OverlayProps, overview, withLead } from "../chapter";
import { createFlashPrefillAboutModel } from "./flash-prefill-about-model";
import { createFlashPrefillModel } from "./flash-prefill-model";

const s = flashShape;

const beats: Beat[][] = [
  [flat("One walk per row against one pass per tile of rows")],
  [flat(`${s.tests} tests, written before any kernel`)],
  [flat("The first kernel: the bytes moved, the walk did not")],
  [flat(`One warp, ${s.tile} query rows, the keys ${s.tile} at a time`)],
  [
    overview(`One mma call: ${s.tile} × ${s.tile} times ${s.tile} × ${s.half}`),
    close("Close up: the pieces of A and B one lane holds", [0.33, 0.03, -0.2], 0.66, 0.62, -18, 44),
    close("From the side: the scores rise with every call", [0.46, 0.05, -0.05], 0.5, 0.42, 28, 30),
    close("The rail: drag to pick a lane of the warp", [0.5, 0.02, 0.22], 0.9, 0.34, 0, 46),
  ],
  [flat("Scores become weights where they are")],
  [flat("What one lane keeps in registers")],
  [flat("Time to first token, four paths")],
  [flat("Pick a prompt length")],
  [flat("What is still open")],
];

const instructions = [
  "Compare the two triangles: every bar on the left is its own walk.",
  "Read the four groups: what must match, what must not change, what must not be read.",
  "Watch the keys go by one at a time, then compare the two bars.",
  "Watch the key tiles pass the query tile, one after another.",
  "Drag the pink ball to pick a lane: its pieces of A and B move, and its scores rise with each call.",
  "Watch the same cells light up on both sides for each lane.",
  "Watch which value each call takes from every group.",
  "Find the two lines that stay low: the tiled kernel and llama.cpp.",
  "Answer the prediction, then pick a prompt length.",
  "Compare the two bars, then read what was not done.",
];

const figures: Record<string, () => ReactNode> = {
  square: () => <FlashSquare />,
  spec: () => <FlashSpec />,
  first: () => <FlashFirst />,
  tiles: () => <FlashTiles />,
  fragment: () => <FlashFragment />,
  registers: () => <FlashRegisters />,
  ttft: () => <FlashTtft />,
  open: () => <FlashOpen />,
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
  exhibits: [null, null, null, null, 0, null, null, null, null, null],
  overlays: ["square", "spec", "first", "tiles", null, "fragment", "registers", "ttft", "lengths", "open"],
  instructions,
  beats,
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
