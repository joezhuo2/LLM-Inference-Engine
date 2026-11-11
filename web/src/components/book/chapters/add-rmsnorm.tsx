"use client";

import { chapterNumber } from "@/content/book-order";
import { createPortal } from "react-dom";
import { addRmsnormAbout, addRmsnormAboutCaption } from "@/components/kernels/add-rmsnorm/about";
import { addRmsnormCaptions, addRmsnormNumbers, addRmsnormSteps, TokensPicker } from "@/components/kernels/add-rmsnorm/story";
import { PatternFigure } from "@/components/kernels/add-rmsnorm/figures";
import type { Beat } from "../book-scene";
import { type ChapterDef, close, flat, type OverlayProps, overview, withLead } from "../chapter";
import { modelShape } from "@/lib/data";
import { createAddRmsnormAboutModel } from "./add-rmsnorm-about-model";
import { createAddRmsnormModel } from "./add-rmsnorm-model";

const beats: Beat[][] = [
  [
    overview("One row: add, square, one rstd, scale"),
    close("Close up: each sum bar is its x bar plus its r bar", [0.55, 0.1, -0.4], 0.84, 0.44, -20, 34),
    close("From the side: the squares grow with the scale, the norm does not", [0.55, 0.08, 0.02], 0.84, 0.4, 30, 26),
    close("The slider: scale x and r together", [0.55, 0.03, 0.27], 0.9, 0.3, 0, 44),
  ],
  [flat("Fusion: a round trip and a launch removed")],
  [
    overview("Two kernels and a round trip, or one kernel"),
    close("Close up: the sum goes out to DRAM and comes back twice", [0.35, 0.05, -0.42], 0.75, 0.42, -20, 42),
    close("The fused kernel: inputs once, outputs once", [0.4, 0.04, -0.02], 0.7, 0.36, 22, 40),
    close("From the side: DRAM bytes at 64 tokens", [0.82, 0.1, -0.2], 0.4, 0.6, 30, 24),
  ],
  [
    overview("Six phases in order, two barriers"),
    close("Close up: the barriers and the read back", [0.5, 0.05, -0.45], 0.9, 0.28, -10, 55),
    close("From the side: the same time as the pair", [0.45, 0.03, -0.01], 0.8, 0.36, 24, 40),
  ],
  [
    overview("Every value in order, every rounding to BF16"),
    close("Close up: the sum is rounded before it is squared", [0.45, 0.03, -0.35], 0.85, 0.42, -16, 50),
    close("From the side: round twice, as the naive kernel does", [0.45, 0.03, 0.1], 0.85, 0.42, 18, 48),
  ],
  [
    overview("8 threads of 256: 2-byte loads against 16-byte loads"),
    close("Close up: one value per thread per round", [0.55, 0.03, -0.5], 0.8, 0.26, -18, 40),
    close("From the side: 8 values in one load, kept in registers", [0.55, 0.04, -0.24], 0.8, 0.3, 24, 32),
    close("Loads per thread: 32 against 3", [0.48, 0.02, 0.12], 0.95, 0.34, 0, 44),
  ],
  [
    overview("8 lanes, 3 xor swaps, the total in every lane"),
    close("Close up: xor 4, each lane adds its partner four lanes away", [0.5, 0.05, -0.42], 0.9, 0.4, -16, 44),
    close("From the side: every lane ends with the same total", [0.5, 0.06, -0.02], 0.9, 0.4, 24, 36),
  ],
  [flat("Pick a token count")],
];

const instructions = [
  "Drag the pink ball to scale x and r; the norm stays the same.",
  "Watch t go out to DRAM and come back between two kernels, and stay put in the fused one.",
  "Watch the sum leave and come back in the top kernels, and stay put in the fused one.",
  "Watch the phases fill one after another and wait at each barrier.",
  "Watch the value become a cube each time it is rounded to BF16.",
  "Compare one value per round with 8 values in one load.",
  "Watch each lane add its partner's value at every swap.",
];

function AddRmsnormOverlay({ kind, target, narrow }: OverlayProps) {
  if (!target) return null;
  if (kind === "fusion") {
    return createPortal(
      <div className={narrow ? "aspect-[560/400] w-full" : "h-[34rem]"}>
        <PatternFigure />
      </div>,
      target,
    );
  }
  if (kind !== "tokens") return null;
  return createPortal(
    <div className={narrow ? undefined : "h-[34rem]"}>
      <TokensPicker />
    </div>,
    target,
  );
}

const chapter: ChapterDef = {
  slug: "add-rmsnorm",
  number: chapterNumber("add-rmsnorm"),
  title: "fused add + RMSNorm",
  formula: "sum = x + r,  norm = w · sum / √(mean(sum²) + ε)",
  steps: addRmsnormSteps,
  captions: addRmsnormCaptions,
  exhibits: [0, null, 1, 2, 3, 4, 5, null],
  overlays: [null, "fusion", null, null, null, null, null, "tokens"],
  instructions,
  beats,
  workbench: [],
  buildModel: (kit, fonts) => createAddRmsnormModel(kit, fonts, addRmsnormNumbers),
  Overlay: AddRmsnormOverlay,
};

const shape = modelShape();

export const addRmsnormChapter = withLead(chapter, {
  step: addRmsnormAbout,
  caption: addRmsnormAboutCaption,
  instruction: "Drag the pink ball to make the arriving stream bigger; the pink copies each block reads stay the same size.",
  beats: [
    overview("What the residual add and RMSNorm are, and where they sit in a layer"),
    close("Close up: RMSNorm hands attention a copy of the stream", [0.36, 0.04, -0.26], 0.58, 0.36, -20, 40),
    close("From the side: the stream grows with each add, the copies do not", [0.5, 0.06, -0.1], 0.9, 0.36, 26, 26),
    close("The rail: drag the size of the stream arriving", [0.5, 0.02, 0.2], 0.9, 0.34, 0, 46),
  ],
  buildModel: (kit, fonts) => createAddRmsnormAboutModel(kit, fonts, { hidden: shape.hidden.toLocaleString("en-US"), blocks: 2 * shape.layers }),
});
