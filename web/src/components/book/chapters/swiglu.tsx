"use client";

import { createPortal } from "react-dom";
import { swigluAbout, swigluAboutCaption } from "@/components/kernels/swiglu/about";
import { swigluCaptions, swigluNumbers, swigluSteps, TokensPicker } from "@/components/kernels/swiglu/story";
import type { Beat } from "../book-scene";
import { type ChapterDef, close, flat, type OverlayProps, overview, withLead } from "../chapter";
import { createSwigluAboutModel } from "./swiglu-about-model";
import { createSwigluModel } from "./swiglu-model";

const beats: Beat[][] = [
  [
    overview("One token's gate/up row, zipped into one output row"),
    close("Close up: value j of the gate meets value j of up", [0.5, 0.04, -0.45], 0.9, 0.36, 0, 44),
    close("From the side: silu stays near 0 below 0 and follows g above", [0.5, 0.1, -0.1], 0.86, 0.4, -24, 18),
    close("The rail: the gate is the only input", [0.5, 0.03, 0.17], 0.9, 0.34, 0, 46),
  ],
  [
    overview("Two roundings, in PyTorch's order"),
    close("Close up: s lands just under a rounding boundary", [0.5, 0.05, -0.38], 0.9, 0.36, -12, 44),
    close("From the side: after × up the two balls land one step apart", [0.5, 0.05, -0.06], 0.9, 0.4, 16, 38),
  ],
  [
    overview("22 blocks per row, one row per blockIdx.y"),
    close("Close up: the rows stop at gridDim.y's ceiling", [0.4, 0.03, -0.45], 0.76, 0.36, -20, 40),
    close("From the side: the rows flatten into one 1D grid", [0.4, 0.03, -0.12], 0.76, 0.36, 22, 30),
    close("The bars: naive against the draft at one token", [0.86, 0.08, -0.32], 0.34, 0.3, 24, 26),
  ],
  [
    overview("Eight 2-byte loads, or one 16-byte load"),
    close("Close up: the scalar lane makes 8 trips", [0.5, 0.04, -0.5], 0.86, 0.3, -18, 36),
    close("From the side: the vector bar reaches the peak plane", [0.5, 0.08, -0.06], 0.84, 0.36, 24, 18),
  ],
  [
    overview("26 SMs, two ways to cover them"),
    close("Close up: at one token the vector kernel fills 3 SMs, 8 chains deep", [0.5, 0.05, -0.42], 0.9, 0.34, -16, 32),
    close("From the side: the scalar kernel spreads one chain over 22 SMs", [0.5, 0.05, -0.15], 0.9, 0.34, 22, 26),
    close("The rail: drag the token count past 4", [0.5, 0.03, 0.14], 0.9, 0.32, 0, 46),
  ],
  [flat("Pick a token count")],
];

const instructions = [
  "Drag the pink ball along the rail to change the gate.",
  "Watch the blue ball round twice and the pink one once.",
  "Watch the rows hit the ceiling, then flatten into one grid.",
  "Watch the two lanes load, then the bars rise to the plane.",
  "Drag the pink ball to change the token count.",
];

function SwigluOverlay({ kind, target, narrow }: OverlayProps) {
  return kind === "tokens" && target
    ? createPortal(
        <div className={narrow ? undefined : "h-[34rem]"}>
          <TokensPicker />
        </div>,
        target,
      )
    : null;
}

const chapter: ChapterDef = {
  slug: "swiglu",
  number: 9,
  title: "SwiGLU",
  formula: "outⱼ = silu(gateⱼ) · upⱼ,  silu(g) = g / (1 + e⁻ᵍ)",
  steps: swigluSteps,
  captions: swigluCaptions,
  exhibits: [0, 1, 2, 3, 4, null],
  overlays: [null, null, null, null, null, "tokens"],
  instructions,
  beats,
  workbench: [],
  buildModel: (kit, fonts) => createSwigluModel(kit, fonts, swigluNumbers),
  Overlay: SwigluOverlay,
};

export const swigluChapter = withLead(chapter, {
  step: swigluAbout,
  caption: swigluAboutCaption,
  instruction: "Drag the pink ball to set the outlined value's gate and watch how much of its up value passes.",
  beats: [
    overview("What SwiGLU stands for, and where it sits in the MLP"),
    close("Close up: two projections up, one value per pair, one projection down", [0.52, 0.03, -0.33], 0.86, 0.36, -18, 40),
    close("From the side: silu rises with g, sigmoid levels off at 1", [0.45, 0.05, 0.0], 0.7, 0.3, 24, 22),
    close("The rail: drag the gate from −4 to 4", [0.5, 0.02, 0.2], 0.9, 0.32, 0, 46),
  ],
  buildModel: createSwigluAboutModel,
});
