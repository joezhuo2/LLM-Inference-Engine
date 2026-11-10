"use client";

import { createPortal } from "react-dom";
import { splitAbout, splitAboutCaption } from "@/components/kernels/split-attention/about";
import { SplitPicker, splitCaptions, splitNumbers, splitSteps } from "@/components/kernels/split-attention/story";
import type { Beat } from "../book-scene";
import { type ChapterDef, close, flat, type OverlayProps, overview, withLead } from "../chapter";
import { createSplitAboutModel } from "./split-attention-about-model";
import { createSplitModel } from "./split-attention-model";

const beats: Beat[][] = [
  [
    overview("32 blocks for 156 slots: a fifth of one wave"),
    close("Close up: most slots in each SM stay empty", [0.3, 0.04, -0.42], 0.95, 0.68, -24, 34),
    close("From the side: each block is the whole 2,048-token context", [0.72, 0.06, -0.36], 0.85, 0.61, 28, 30),
    close("The readout: blocks, waves and warps from Nsight Compute", [0.5, 0.03, 0.12], 1.0, 0.61, 0, 44),
  ],
  [
    overview("Four warps waiting, three gauges nearly empty"),
    close("Close up: a load, a long wait, a little math", [0.3, 0.03, -0.38], 1.02, 0.68, -20, 36),
    close("From the side: the SM lights only on math", [0.74, 0.06, -0.36], 0.68, 0.58, 26, 30),
    close("The jars: DRAM is barely used", [0.5, 0.06, 0.14], 1.0, 0.6, 0, 50),
  ],
  [
    overview("128 KV blocks, 8 ranges, 8 thread blocks"),
    close("Close up: the cuts fall between whole blocks", [0.3, 0.02, -0.5], 0.85, 0.51, -18, 40),
    close("From the side: every range is walked at the same time", [0.55, 0.03, -0.2], 1.0, 0.85, 24, 30),
  ],
  [
    overview("Pick a split count: the grid doubles each time"),
    close("Close up: 8 splits leave 100 blocks for a second wave", [0.5, 0.03, -0.25], 1.0, 0.85, 0, 40),
    close("From the side: half the work per block, twice the blocks", [0.72, 0.04, -0.36], 0.85, 0.61, 28, 30),
    close("The rail and the clock", [0.5, 0.03, 0.08], 1.0, 0.68, 0, 46),
  ],
  [
    overview("8 records of 66 floats each"),
    close("Close up: m, l and 64 values per record", [0.27, 0.03, -0.38], 0.68, 0.51, -22, 34),
    close("From the side: they fill in a scrambled order", [0.7, 0.03, -0.38], 0.85, 0.58, 26, 30),
  ],
  [
    overview("Four partials become one sum"),
    close("Close up: the pink ruler is the largest m", [0.28, 0.06, -0.38], 0.85, 0.61, -20, 30),
    close("From the side: the stack grows in split order", [0.8, 0.08, -0.36], 0.61, 0.58, 24, 30),
    close("The formula", [0.5, 0.03, 0.1], 1.0, 0.61, 0, 46),
  ],
  [
    overview("5 graphs before, 11 after"),
    close("Close up: the pink plinth marks the graph that replays", [0.45, 0.03, -0.45], 1.02, 0.61, -18, 36),
    close("The bars: ms per token", [0.5, 0.03, 0.1], 1.0, 0.65, 0, 42),
  ],
  [flat("Pick rows and a context")],
];

const instructions = [
  "Watch 32 blocks land in the slots and drain together.",
  "Follow the pink line: the SM only works when a warp reaches its math.",
  "Watch the context cut into 8 ranges that are walked side by side.",
  "Drag the pink ball through the split counts and watch the grid and the clock.",
  "Watch the records fill in a different order from the splits.",
  "Watch each pillar shrink by exp(m − M), then stack in split order.",
  "Watch the pink plinth move from the unsplit graph to the one that matches the context.",
];

const captions = splitCaptions.map((c, i) => (i === 1 ? `${c}; the warp loop is a sketch` : i === 6 ? `${c}; graph tiles from the attention_splits rule` : c));

function SplitOverlay({ kind, target, narrow }: OverlayProps) {
  return (
    <>
      {kind === "picker" &&
        target &&
        createPortal(
          <div className={narrow ? undefined : "h-[34rem]"}>
            <SplitPicker />
          </div>,
          target,
        )}
    </>
  );
}

const chapter: ChapterDef = {
  slug: "split-attention",
  number: 7,
  title: "split attention",
  formula: "out = Σ eᵐⁱ⁻ᴹ accᵢ / Σ eᵐⁱ⁻ᴹ lᵢ",
  steps: splitSteps,
  captions,
  exhibits: [0, 1, 2, 0, 3, 4, 5, null],
  overlays: [null, null, null, null, null, null, null, "picker"],
  instructions,
  beats,
  workbench: [],
  buildModel: (kit, fonts) => createSplitModel(kit, fonts, splitNumbers),
  Overlay: SplitOverlay,
};

export const splitAttentionChapter = withLead(chapter, {
  step: splitAbout,
  caption: splitAboutCaption,
  instruction: "Drag the pink ball through the split counts and count the SMs that one head keeps busy.",
  beats: [
    overview("What an SM and a thread block are, and what splitting does"),
    close("Close up: the context cut into pieces, one block for each", [0.34, 0.03, -0.32], 0.62, 0.38, -20, 40),
    close("From the side: partial results travel to the merge kernel", [0.62, 0.04, -0.28], 0.62, 0.4, 26, 30),
    close("The rail: drag from 1 to 8 splits", [0.5, 0.02, 0.1], 0.9, 0.36, 0, 46),
  ],
  buildModel: createSplitAboutModel,
});
