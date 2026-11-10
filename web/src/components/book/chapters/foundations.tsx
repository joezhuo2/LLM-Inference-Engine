"use client";

import { createPortal } from "react-dom";
import { foundationsAbout, foundationsAboutCaption } from "@/components/kernels/foundations/about";
import { foundationsIdeas, IdeaFigure, ideaCaptions, ideaInstructions } from "@/components/kernels/foundations/ideas";
import { FoundationsSandbox } from "@/components/kernels/foundations/sandbox";
import { foundationsCaptions, foundationsNumbers, foundationsSteps, RowsPicker } from "@/components/kernels/foundations/story";
import { foundations } from "@/content/kernels/foundations";
import type { Beat } from "../book-scene";
import { type ChapterDef, close, flat, type OverlayProps, overview, type WorkbenchPage, withLead } from "../chapter";
import { createFoundationsAboutModel } from "./foundations-about-model";
import { createFoundationsModel } from "./foundations-model";

const n = foundationsNumbers;

const ideaBeats: Beat[][] = foundationsIdeas.map((step) => [flat(`Big idea: ${step.title}`)]);

const beats: Beat[][] = [
  ...ideaBeats,
  [
    overview(`One launch: ${n.blocks} blocks on ${n.sms} SMs`),
    close("Close up: each block sits on one SM", [0.5, 0.03, -0.46], 0.9, 0.36, -18, 40),
    close("From the side: one block opens into 8 warps", [0.5, 0.03, -0.2], 0.9, 0.36, 24, 30),
    close("Close up: the pink warp's 32 threads rise together", [0.5, 0.02, 0.0], 0.9, 0.36, 0, 38),
  ],
  [
    overview("Four places to keep a number, sizes not to scale"),
    close("From the side: the tile waits longest leaving DRAM", [0.45, 0.03, -0.38], 1.0, 0.5, -20, 46),
    close("Close up: the FMA in registers, then the result goes back up", [0.45, 0.03, -0.06], 1.0, 0.5, 18, 48),
    close("The measured tiers, card by card", [0.76, 0.03, -0.24], 0.5, 0.78, 0, 52),
  ],
  [
    overview("The same weights, read once, used by 1 row or 512"),
    close("Close up: decode, the read is almost all of the time", [0.5, 0.02, -0.42], 1.0, 0.4, -14, 52),
    close("Close up: prefill, the arithmetic outlasts the read", [0.5, 0.02, -0.1], 1.0, 0.4, 14, 52),
  ],
  [
    overview("The roofline as a wall, the batch as a ball"),
    close("From the side: the slope, the ridge and the flat top", [0.5, 0.12, -0.3], 0.92, 0.4, -24, 18),
    close("Close up: the measured dots sit on the slope", [0.32, 0.08, -0.24], 0.7, 0.4, 10, 30),
    close("The rail: rows that share one weight read", [0.5, 0.03, 0.18], 0.9, 0.3, 0, 44),
  ],
  [flat("Pick a row count")],
  [
    overview("Six trips to DRAM, then four"),
    close("Close up: the unfused pair writes the sum and reads it twice", [0.5, 0.03, -0.42], 1.0, 0.44, -14, 48),
    close("From the side: fused, the sum stays in registers", [0.5, 0.03, -0.06], 1.0, 0.44, 18, 46),
  ],
];

const instructions = [
  ...ideaInstructions,
  "Watch the blocks land on the SMs and the pink warp's threads move in lockstep.",
  "Watch the tile wait at each tier and the result climb back up.",
  "Watch which bar is still filling after the other has stopped.",
  "Drag the pink ball along the rail to change the rows.",
  undefined,
  "Count the trips to global memory in each row.",
];

const workbench: WorkbenchPage[] = [
  {
    title: "place an op on the roofline",
    overlay: "sandbox-one",
    caption: "Try it yourself: place an op on the roofline",
    paragraphs: [
      foundations.sandbox?.intro ?? "",
      "Predict first: set the batch where you think the op turns compute bound and lock it in, or say it never crosses. A dashed line marks your guess on the chart, and the verdict tells you how far off it was.",
      "For the QKV projection, o_proj, gate/up and down, every row reuses the same weights, so the FLOPs grow with the rows while the bytes barely move and the ring climbs toward the ridge. Attention, RMSNorm, SwiGLU and RoPE have no shared weight matrix, so more rows add bytes as fast as FLOPs.",
      "Attention also takes a context length, because every row reads its own K and V cache. The solid dot appears only where the profiles measured the op, at the batch sizes they ran.",
    ],
  },
  {
    title: "what the ring is made of",
    overlay: "sandbox-two",
    caption: "Try it yourself: the numbers behind the ring",
    paragraphs: [
      "This page shows the numbers behind the ring for the op and the batch you set on the previous spread: the FLOPs and the bytes of one decode step over all the model's layers, their ratio, and the ridge that ratio has to beat.",
      "The two bars are the time the op would take if only memory or only compute limited it: bytes ÷ bandwidth and FLOPs ÷ compute. The longer bar is the op's time at the ceilings, and it names the bound.",
      "Below them is the measured row from the profiles on the same GPU: the time, the share of the bandwidth roof, and the bandwidth and compute it reached. Where nothing was measured the page says so, and only the computed numbers stand.",
      "Go back a page, change the op or the batch, and come back: the sandbox keeps its state across both spreads, so these numbers always match the ring you placed.",
    ],
  },
];

function FoundationsOverlay({ kind, target, narrow }: OverlayProps) {
  return (
    <>
      {kind?.startsWith("idea-") &&
        target &&
        createPortal(
          <div className={narrow ? "aspect-[560/400] w-full" : "h-[34rem]"}>
            <IdeaFigure index={Number(kind.slice(5))} />
          </div>,
          target,
        )}
      {kind === "rows" &&
        target &&
        createPortal(
          <div className={narrow ? undefined : "h-[34rem]"}>
            <RowsPicker />
          </div>,
          target,
        )}
      <FoundationsSandbox pages={{ one: kind === "sandbox-one" ? target : null, two: kind === "sandbox-two" ? target : null }} />
    </>
  );
}

const chapter: ChapterDef = {
  slug: "foundations",
  number: 1,
  title: "foundations",
  formula: foundations.formula,
  steps: [...foundationsIdeas, ...foundationsSteps],
  captions: [...ideaCaptions, ...foundationsCaptions],
  exhibits: [...foundationsIdeas.map(() => null), 0, 1, 2, 3, null, 4],
  overlays: [...foundationsIdeas.map((_, k) => `idea-${k}`), null, null, null, null, "rows", null],
  instructions,
  beats,
  workbench,
  buildModel: (kit, fonts) => {
    const model = createFoundationsModel(kit, fonts, n);
    return { ...model, enter: (stop) => model.enter?.(stop - foundationsIdeas.length) };
  },
  Overlay: FoundationsOverlay,
};

export const foundationsChapter = withLead(chapter, {
  step: foundationsAbout,
  caption: foundationsAboutCaption,
  instruction: "Watch one token go through a decode step, or drag the pink ball to pick which token of the answer is generated.",
  beats: [
    overview("What an inference engine does with one token"),
    close("Close up: the last token's id picks one row of the embedding table", [0.2, 0.03, -0.24], 0.5, 0.36, -20, 40),
    close("Close up: the highest logit wins and joins the answer", [0.72, 0.04, -0.32], 0.52, 0.4, 16, 42),
    close("The rail: pick which token of the answer is generated", [0.5, 0.02, 0.24], 0.9, 0.32, 0, 46),
  ],
  buildModel: createFoundationsAboutModel,
});
