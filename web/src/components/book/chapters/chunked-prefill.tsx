"use client";

import { ChunkedPrefillSandbox } from "@/components/kernels/chunked-prefill/sandbox";
import { chunkedPrefillCaptions, chunkedPrefillNumbers, chunkedPrefillSteps } from "@/components/kernels/chunked-prefill/story";
import { chunkedPrefill } from "@/content/kernels/chunked-prefill";
import type { Beat } from "../book-scene";
import { type ChapterDef, close, flat, type OverlayProps, overview, type WorkbenchPage } from "../chapter";
import { createChunkedPrefillModel } from "./chunked-prefill-model";

const n = chunkedPrefillNumbers;

const beats: Beat[][] = [
  [
    overview("schedule(): one prompt holds a whole step"),
    close("Close up: A, B and C wait while the prompt runs", [0.5, 0.03, -0.3], 0.9, 0.5, -14, 44),
    close("From the side: the prompt's step against two decode steps", [0.54, 0.04, -0.26], 0.95, 0.6, 26, 34),
    close(`The measurements: ${n.alone} ms alone, ${n.largestGap} ms longest gap`, [0.52, 0.03, 0.14], 0.86, 0.36, 0, 50),
  ],
  [
    overview(`A step of ${n.budgets[0]} token slots`),
    close("Close up: the decode rows land first", [0.28, 0.03, -0.4], 0.44, 0.3, -24, 38),
    close("From the side: step 2 continues the prompt", [0.52, 0.03, -0.24], 0.95, 0.55, 24, 36),
    close("The three rules, in order", [0.52, 0.03, 0.08], 0.86, 0.34, 0, 52),
  ],
  [
    overview(`A ${n.prompt}-token prompt spread over steps`),
    close("From the side: every column keeps its decode rows", [0.52, 0.08, -0.34], 0.95, 0.45, 26, 34),
    close("Close up: the budget rail", [0.52, 0.03, 0.04], 0.86, 0.3, 0, 46),
    close(`Measured: no budget against a budget of ${n.chunkBudget}`, [0.52, 0.03, 0.2], 0.86, 0.3, 0, 54),
  ],
  [
    overview("A decoder needs a block and none is free"),
    close("Close up: the back of the running list", [0.66, 0.03, -0.38], 0.5, 0.36, 22, 36),
    close("From the side: P goes to the front of the queue", [0.45, 0.05, -0.12], 0.9, 0.56, -24, 36),
    close("The rule", [0.5, 0.03, 0.22], 0.86, 0.3, 0, 54),
  ],
  [
    overview("Two steps and their sampled lists"),
    close("Close up: a chunk that stops short has no entry", [0.3, 0.03, -0.38], 0.56, 0.36, -22, 38),
    close("From the side: P joins at the end of its prompt", [0.45, 0.03, -0.1], 0.86, 0.45, 24, 38),
    close("update() takes one token per entry", [0.5, 0.03, 0.18], 0.86, 0.3, 0, 54),
  ],
  [flat("Predict, then compare the two rules")],
  [flat("Drop prompts and watch the lanes")],
];

const instructions = [
  "Watch the prompt's step grow while A, B and C wait.",
  "Watch the decode rows fill the budget before the chunk.",
  "Drag the pink ball along the rail to change the budget.",
  "Watch the victim leave from the back of the running list.",
  "Watch which rows reach the sampled list.",
];

const workbench: WorkbenchPage[] = [
  {
    title: "predict, then compare",
    overlay: "sandbox-predict",
    caption: "Try it yourself: predict, then compare the two rules",
    paragraphs: [
      "Four sequences are already decoding. You drop a 2,048-token prompt, and the engine runs schedule_chunked() with a budget of 256 tokens. Before you press run it, pick how many steps you think the prompt waits for its first token, and how many steps the decoders go without a row under schedule().",
      "The sandbox runs the same arrivals under both rules at once, with a TypeScript port of Scheduler::schedule() and Scheduler::schedule_chunked() from src/sched/scheduler.cpp. The table below the controls keeps one column per rule, so you can compare the first-token step, the longest decode stall and the biggest step side by side.",
      "Look at the biggest step. schedule() gives the prompt its first token sooner, because the whole prompt runs in one step, but that one step is the size of the prompt, and every decoder waits for it. With the budget no step is larger than 256 tokens, so the decoders never lose their row.",
      "The model counts steps and tokens; it does not time anything. The milliseconds at the bottom of the page are measured on the engine, docs/results.md: a step's cost grows with its tokens, which is why one large step is the long wait the decoders feel.",
    ],
  },
  {
    title: "drop prompts, watch the lanes",
    overlay: "sandbox-lanes",
    caption: "Try it yourself: drop prompts and watch the lanes",
    paragraphs: [
      "Drop prompts of any length and step the engine one step at a time, or press play. Each lane is a sequence, each column a step: a decode row is one token, and a prompt's chunk fills as much of the step as the budget leaves after the decode rows.",
      "Change the budget and the rule. Under schedule() the budget is ignored and a prompt takes a step of its own, so the decoder lanes show a gap. Under schedule_chunked() every decoder keeps its row in every step, and the prompt spreads over as many steps as its chunks need.",
      "Pick the tight KV pool and a small budget, then drop a long prompt. Blocks for the whole prompt are taken at admission, so when a decoder needs a block and none is free, the victim is the back of the running list: the sequence in prefill. Watch it go back to the waiting queue and start again at position 0.",
      "The list under the lanes says what each sequence is doing and how many KV blocks are held, so you can see a preemption as it happens instead of inferring it from the lanes.",
    ],
  },
];

function ChunkedPrefillOverlay({ kind, target }: OverlayProps) {
  return <ChunkedPrefillSandbox pages={{ predict: kind === "sandbox-predict" ? target : null, lanes: kind === "sandbox-lanes" ? target : null }} />;
}

export const chunkedPrefillChapter: ChapterDef = {
  slug: "chunked-prefill",
  number: 12,
  title: "chunked prefill",
  formula: chunkedPrefill.formula,
  steps: chunkedPrefillSteps,
  captions: chunkedPrefillCaptions,
  exhibits: [0, 1, 2, 3, 4],
  overlays: [null, null, null, null, null],
  instructions,
  beats,
  workbench,
  buildModel: (kit, fonts) => createChunkedPrefillModel(kit, fonts, chunkedPrefillNumbers),
  Overlay: ChunkedPrefillOverlay,
};
