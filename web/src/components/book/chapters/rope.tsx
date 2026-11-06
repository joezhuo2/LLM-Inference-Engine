"use client";

import { createPortal } from "react-dom";
import { ropeAbout, ropeAboutCaption } from "@/components/kernels/rope/about";
import { RopeSandbox } from "@/components/kernels/rope/sandbox";
import { ropeCaptions, ropeNumbers, ropeSteps, TokensPicker } from "@/components/kernels/rope/story";
import type { Beat } from "../book-scene";
import { type ChapterDef, close, flat, type OverlayProps, overview, type WorkbenchPage, withLead } from "../chapter";
import { createRopeAboutModel } from "./rope-about-model";
import { createRopeModel } from "./rope-model";

const beats: Beat[][] = [
  [
    overview("Eight of the 32 pairs of one head"),
    close("Close up: pair 0 turns once every 6.3 positions", [0.26, 0.05, -0.42], 0.56, 0.42, -24, 34),
    close("From the side: pair 31 has barely moved", [0.74, 0.05, -0.08], 0.56, 0.42, 26, 30),
    close("The rail: position is the only input", [0.5, 0.03, 0.22], 0.9, 0.36, 0, 42),
  ],
  [
    overview("One token's QKV row: 40 heads of 64 values"),
    close("Close up: head 0, the tallest block", [0.2, 0.05, -0.42], 0.5, 0.34, -30, 30),
    close("Value i pairs with value i + 32, not its neighbour", [0.5, 0.08, 0.02], 0.98, 0.4, 0, 28),
  ],
  [
    overview("One last bit, 2,047 times over"),
    close("From the side: the two arms spread with the position", [0.32, 0.18, -0.27], 0.72, 0.5, -8, 16),
    close("Close up: the rounding boundary between two BF16 values", [0.61, 0.13, -0.27], 0.42, 0.34, 18, 24),
  ],
  [
    overview("36 heads × 32 pairs, one chain of 32 per head"),
    close("Close up: every thread walks its 32 pairs in order", [0.36, 0.02, 0.02], 0.72, 0.5, -28, 40),
    close("The jar: 1,152 sines and cosines per token", [0.83, 0.1, -0.03], 0.44, 0.4, 22, 26),
  ],
  [
    overview("Every pair at once: 32 pairs × 8 head groups"),
    close("Close up: whole rows rise, not one cell at a time", [0.36, 0.02, 0.02], 0.72, 0.5, 24, 38),
    close("The second jar: 256 instead of 1,152", [0.83, 0.1, -0.03], 0.44, 0.4, -20, 26),
  ],
  [flat("Pick a token count")],
];

const instructions = ["Drag the pink ball along the rail to change the position.", "Watch each value find its partner, 32 places along.", "Watch where the two values land on the steps.", "Watch the chain cross the plate and the jar fill.", "Compare the rows, the clock and the second jar."];

const workbench: WorkbenchPage[] = [
  {
    title: "one position, 32 pairs",
    overlay: "sandbox-one",
    caption: "Rotate it yourself: one position, 32 pairs",
    paragraphs: [
      "Drag the position and pick a dimension pair on the right. The dial is pair i of a query vector, the two values xᵢ and xᵢ₊₃₂, turned by the angle m · θᵢ, where m is the token's position.",
      "Pair 0 has θ = 1, so it turns a full circle every 6.3 positions. The grid under the dial shows all 32 pairs at the same position: the low pairs spin quickly and tell neighbouring tokens apart, the high pairs barely move.",
      "Pair 31 needs about 47,000 positions for one turn, far longer than TinyLlama's 2,048-token context, so its angle never repeats inside a prompt. The fast pairs repeat often, but the slow ones keep every position's set of angles different.",
      "The vectors are made up for illustration, from seeded random numbers rather than the model. The frequencies, the pairing of dimension i with i + 32 and the rotation are the engine's.",
    ],
  },
  {
    title: "the score depends on the gap",
    overlay: "sandbox-two",
    caption: "Rotate it yourself: the score depends on the gap",
    paragraphs: [
      "Attention scores a query against a key with a dot product. After RoPE the query at position m and the key at position n are each turned by their own angles, so their score q·k depends only on the gap m − n, not on where the pair sits in the prompt.",
      "Predict first, then try it: with the gap locked, move both positions together by 100 and watch the score. Then unlock the gap and move one position at a time.",
      "The curve is the score for every gap from −2,048 to 2,048 with the same two vectors. With the gap locked the dot stays put; with it free, the dot slides along the curve.",
      "This is why the kernel only has to turn each vector by its own position: the relative distance falls out of the dot product, with no table of pairwise positions anywhere.",
    ],
  },
];

function RopeOverlay({ kind, target, narrow }: OverlayProps) {
  return (
    <>
      {kind === "tokens" &&
        target &&
        createPortal(
          <div className={narrow ? undefined : "h-[34rem]"}>
            <TokensPicker />
          </div>,
          target,
        )}
      <RopeSandbox pages={{ one: kind === "sandbox-one" ? target : null, two: kind === "sandbox-two" ? target : null }} />
    </>
  );
}

const chapter: ChapterDef = {
  slug: "rope",
  number: 3,
  title: "RoPE",
  formula: "x′ᵢ = xᵢ cos pθᵢ − xᵢ₊d/₂ sin pθᵢ,   θᵢ = 10000^(−2i/d)",
  steps: ropeSteps,
  captions: ropeCaptions,
  exhibits: [0, 1, 2, 3, 3, null],
  overlays: [null, null, null, null, null, "tokens"],
  instructions,
  beats,
  workbench,
  buildModel: (kit, fonts) => createRopeModel(kit, fonts, ropeNumbers),
  Overlay: RopeOverlay,
};

export const ropeChapter = withLead(chapter, {
  step: ropeAbout,
  caption: ropeAboutCaption,
  instruction: "Drag the pink ball to set how far apart the query and key are; as both move along, the angle between them stays put.",
  beats: [
    overview("What RoPE stands for, and what turning does to a score"),
    close("Close up: the query and the key, each turned by its own position", [0.29, 0.04, -0.2], 0.62, 0.42, -22, 40),
    close("From the side: the angle between them stays as both move along", [0.73, 0.05, -0.12], 0.5, 0.36, 24, 30),
    close("The rail: drag the gap from 0 to 7 positions", [0.5, 0.02, 0.2], 0.9, 0.34, 0, 46),
  ],
  buildModel: createRopeAboutModel,
});
