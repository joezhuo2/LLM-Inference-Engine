"use client";

import { createPortal } from "react-dom";
import { samplingAbout, samplingAboutCaption } from "@/components/kernels/sampling/about";
import { SamplingSandbox } from "@/components/kernels/sampling/sandbox";
import { ModePicker, samplingCaptions, samplingNumbers, samplingSteps } from "@/components/kernels/sampling/story";
import type { Beat } from "../book-scene";
import { type ChapterDef, close, flat, type OverlayProps, overview, type WorkbenchPage, withLead } from "../chapter";
import { createSamplingAboutModel } from "./sampling-about-model";
import { createSamplingModel } from "./sampling-model";

const beats: Beat[][] = [
  [
    overview("One row of logits and the tree that picks its largest"),
    close("Close up: each thread keeps the best value it saw", [0.42, 0.06, -0.3], 0.9, 0.5, -20, 40),
    close("From the side: the winner's height climbs the tree", [0.45, 0.08, -0.2], 0.95, 0.55, 30, 26),
    close("The result: one id, lowest index on ties", [0.47, 0.03, 0.22], 0.95, 0.35, 0, 50),
  ],
  [
    overview("Noise on every logit, then the same argmax"),
    close("Close up: the pink noise lifts or sinks each logit", [0.47, 0.06, -0.34], 0.95, 0.45, -18, 38),
    close("The picks fill the softmax outlines", [0.47, 0.04, -0.06], 0.95, 0.45, 20, 40),
  ],
  [
    overview("Two runs of the same request, four decode steps each"),
    close("Close up: each cell's noise depends on its offset alone", [0.5, 0.03, -0.38], 0.9, 0.4, -18, 50),
    close("From the side: the batched copy grows the same heights", [0.5, 0.03, -0.05], 0.95, 0.45, 22, 50),
  ],
  [
    overview("40 example keys and the cut that searches them"),
    close("From the side: the cut plane halves its range each pass", [0.48, 0.1, -0.36], 0.95, 0.5, 30, 30),
    close("Close up: the pass count on the card", [0.5, 0.03, -0.06], 0.95, 0.35, 0, 50),
    close("The handle: drag k and the search starts again", [0.5, 0.03, 0.2], 0.95, 0.35, -14, 46),
  ],
  [
    overview("The same search, on probability mass"),
    close("Close up: the kept mass fills the trough to the pink line", [0.5, 0.04, -0.18], 1.0, 0.45, -12, 52),
    close("From the side: the cut settles where the mass reaches p", [0.48, 0.1, -0.36], 0.95, 0.5, 30, 30),
    close("The handle: drag p", [0.5, 0.03, 0.2], 0.95, 0.35, 14, 46),
  ],
  [
    overview("Six modes against sorting the row alone"),
    close("Close up: top-k 50 stops well short of the pink wall", [0.5, 0.03, -0.3], 1.0, 0.45, -18, 46),
    close("From the side: the two slow bars pass the wall", [0.55, 0.04, -0.18], 1.0, 0.55, 30, 32),
  ],
  [
    overview("Double-precision noise: where the time goes"),
    close("Close up: a small share of instructions keeps the pipe busy", [0.42, 0.08, -0.33], 0.95, 0.45, -18, 38),
    close("Temperature only: every token waits for its noise", [0.5, 0.02, -0.04], 0.95, 0.35, 0, 46),
    close("Top-k 50: one cell here stands for the 50 kept", [0.5, 0.02, 0.17], 0.95, 0.35, 16, 46),
  ],
  [flat("Pick a row count")],
];

const instructions = [
  "Watch each thread rise to the best logit it holds, level by level.",
  "Watch the pink noise reshuffle the winner and the picks fill the outlines.",
  "Compare the two grids: the same seed and step give the same heights.",
  "Drag the pink ball to change k and watch the cut search again.",
  "Drag the pink ball to change p and watch the trough reach the line.",
  "Compare each bar with the pink wall: the sort alone.",
  "Watch the top strip fill one cell at a time while the bottom one is done.",
  "Switch between 1 and 16 rows: the bars barely move.",
];

const workbench: WorkbenchPage[] = [
  {
    title: "turn the knobs",
    overlay: "sandbox-one",
    caption: "Sample it yourself: turn the knobs",
    paragraphs: [
      "Pick a prompt and drag the three knobs on the right. The bars are the next-token logits TinyLlama produced for three real contexts: the twelve likeliest tokens and the rest grouped together.",
      "Switch the stage to watch the engine's steps in order: the raw logits, divide by the temperature, keep the top-k, then keep the smallest set of those whose probability reaches top-p. Grey bars are cut. The thin bar under each token is the running total of what top-k left, and the line across it is top-p.",
      "Drag T toward 0 and the top bar takes nearly all the mass; drag it toward 2 and the odds flatten. At T = 0 the engine is greedy and ignores top-k and top-p, as the greedy kernel does.",
      "The logits are the BF16 output of TinyLlama on the CPU, the largest ones kept and the rest in bins. The steps are the engine's sample kernel's, in the same order.",
    ],
  },
  {
    title: "draw and predict",
    overlay: "sandbox-two",
    caption: "Sample it yourself: draw and predict",
    paragraphs: [
      "Press sample to draw one token with the engine's Gumbel-max rule, or sample 20 to draw twenty. The draws run on a seed and a step, like the kernel's Philox noise, so a new seed gives a new sequence and the same seed repeats it.",
      "The table counts the candidates left after top-k and top-p, the entropy of what is left, the share of the likeliest token and how much mass the cuts removed.",
      "Predict first: with top-p 0.90 and top-k off, raise the temperature from 0.7 to 1.5. Do the candidates left grow, shrink or stay the same? Answer, then set it and check.",
      "Flatter odds spread the mass over more tokens, so the 90% line sits further down the list. The answer under the buttons gives the counts for the prompt you picked.",
    ],
  },
];

function SamplingOverlay({ kind, target, narrow }: OverlayProps) {
  return (
    <>
      {kind === "rows" &&
        target &&
        createPortal(
          <div className={narrow ? undefined : "h-[24rem]"}>
            <ModePicker />
          </div>,
          target,
        )}
      <SamplingSandbox pages={{ one: kind === "sandbox-one" ? target : null, two: kind === "sandbox-two" ? target : null }} />
    </>
  );
}

const chapter: ChapterDef = {
  slug: "sampling",
  number: 10,
  title: "sampling",
  formula: "id = argmax(logits / T + g),  g = −log(−log u)",
  steps: samplingSteps,
  captions: samplingCaptions,
  exhibits: [0, 1, 2, 3, 4, 5, 6, null],
  overlays: [null, null, null, null, null, null, null, "rows"],
  instructions,
  beats,
  workbench,
  buildModel: (kit, fonts) => createSamplingModel(kit, fonts, samplingNumbers),
  Overlay: SamplingOverlay,
};

export const samplingChapter = withLead(chapter, {
  step: samplingAbout,
  caption: samplingAboutCaption,
  instruction: "Drag the pink ball to change the temperature T and watch the blue odds sharpen or flatten under the draws.",
  beats: [
    overview("What sampling does with the 32,000 logits"),
    close("Close up: the logits behind, their probabilities in front", [0.55, 0.04, -0.24], 0.8, 0.46, -18, 46),
    close("From the side: each draw lands on a token by its odds", [0.58, 0.05, -0.2], 0.95, 0.5, 24, 38),
    close("The rail: drag T from 0, greedy, to 2", [0.5, 0.02, 0.16], 0.9, 0.34, 0, 46),
  ],
  buildModel: createSamplingAboutModel,
});
