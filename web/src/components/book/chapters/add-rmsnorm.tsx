"use client";

import { createPortal } from "react-dom";
import { addRmsnormCaptions, addRmsnormNumbers, addRmsnormSteps, TokensPicker } from "@/components/kernels/add-rmsnorm/story";
import type { Beat } from "../book-scene";
import { type ChapterDef, close, flat, type OverlayProps, overview } from "../chapter";
import { createAddRmsnormModel } from "./add-rmsnorm-model";

const beats: Beat[][] = [
  [
    overview("One row: add, square, one rstd, scale"),
    close("Close up: x and r stack into the sum, kept as the next residual", [0.55, 0.1, -0.43], 0.8, 0.4, -20, 34),
    close("From the side: the squares grow with the scale, the norm does not", [0.55, 0.08, -0.08], 0.8, 0.42, 30, 26),
    close("The slider: scale x and r together", [0.55, 0.03, 0.2], 0.9, 0.32, 0, 44),
  ],
  [
    overview("Two kernels and a round trip, or one kernel"),
    close("Close up: the sum goes out to DRAM and comes back twice", [0.32, 0.06, -0.44], 0.6, 0.4, -24, 32),
    close("The fused kernel: inputs once, outputs once", [0.32, 0.05, -0.02], 0.6, 0.36, 22, 30),
    close("From the side: DRAM bytes at 64 tokens", [0.82, 0.1, -0.2], 0.4, 0.6, 30, 24),
  ],
  [
    overview("Six phases in order, two barriers"),
    close("Close up: the barriers and the read back", [0.5, 0.06, -0.44], 0.86, 0.32, -18, 30),
    close("From the side: the same time as the pair", [0.45, 0.03, 0.08], 0.8, 0.36, 24, 32),
  ],
  [
    overview("Every value in order, every rounding to BF16"),
    close("Close up: the sum is rounded before it is squared", [0.45, 0.04, -0.35], 0.8, 0.4, -20, 36),
    close("From the side: round twice, as the naive kernel does", [0.45, 0.04, 0.12], 0.8, 0.4, 22, 34),
  ],
  [
    overview("8 threads of 256: 2-byte loads against 16-byte loads"),
    close("Close up: one value per thread per round", [0.55, 0.03, -0.5], 0.8, 0.26, -18, 40),
    close("From the side: 8 values in one load, kept in registers", [0.55, 0.04, -0.24], 0.8, 0.3, 24, 32),
    close("Loads per thread: 32 against 3", [0.5, 0.02, 0.12], 0.86, 0.32, 0, 44),
  ],
  [
    overview("8 lanes, 3 xor swaps, the total in every lane"),
    close("Close up: xor 4, each lane adds its partner four lanes away", [0.55, 0.06, -0.44], 0.82, 0.34, -20, 32),
    close("From the side: every lane ends with the same total", [0.55, 0.08, 0.0], 0.82, 0.34, 26, 26),
  ],
  [flat("Pick a token count")],
];

const instructions = [
  "Drag the pink ball to scale x and r; the norm stays the same.",
  "Watch the sum leave and come back in the top kernels, and stay put in the fused one.",
  "Watch the phases fill one after another and wait at each barrier.",
  "Watch the value become a cube each time it is rounded to BF16.",
  "Compare one value per round with 8 values in one load.",
  "Watch each lane add its partner's value at every swap.",
];

function AddRmsnormOverlay({ kind, target, narrow }: OverlayProps) {
  if (kind !== "tokens" || !target) return null;
  return createPortal(
    <div className={narrow ? undefined : "h-[34rem]"}>
      <TokensPicker />
    </div>,
    target,
  );
}

export const addRmsnormChapter: ChapterDef = {
  slug: "add-rmsnorm",
  number: 8,
  title: "fused add + RMSNorm",
  formula: "sum = x + r,  norm = w · sum / √(mean(sum²) + ε)",
  steps: addRmsnormSteps,
  captions: addRmsnormCaptions,
  exhibits: [0, 1, 2, 3, 4, 5, null],
  overlays: [null, null, null, null, null, null, "tokens"],
  instructions,
  beats,
  workbench: [],
  buildModel: (kit, fonts) => createAddRmsnormModel(kit, fonts, addRmsnormNumbers),
  Overlay: AddRmsnormOverlay,
};
