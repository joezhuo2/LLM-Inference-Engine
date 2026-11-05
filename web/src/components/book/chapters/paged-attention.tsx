"use client";

import { prologueCaptions, prologueNumbers, prologueSteps } from "@/components/kernels/attention/prologue-story";
import { PagedAttentionSandbox } from "@/components/kernels/attention/sandbox";
import { attentionCaptions, attentionNumbers, attentionSteps } from "@/components/kernels/attention/story";
import type { Beat } from "../book-scene";
import { type ChapterDef, close, type OverlayProps, overview, type WorkbenchPage } from "../chapter";
import { createPagedAttentionModel } from "./paged-attention-model";

const beats: Beat[][] = [
  [
    overview("Prefill: eight tokens, one wide pass"),
    close("Close up: every position enters the model at once", [0.53, 0.04, -0.4], 1.12, 0.50, -18, 36),
    close("From the side: 8 keys and 8 values filled together", [0.5, 0.03, -0.09], 1.12, 0.42, 24, 30),
    close("The two prompt lengths, to scale", [0.53, 0.02, 0.24], 1.12, 0.48, 0, 48),
  ],
  [
    overview("Decode: one token per step"),
    close("Close up: the model works on one position", [0.62, 0.04, -0.38], 0.84, 0.50, -22, 34),
    close("From the side: every cached pair is read again", [0.5, 0.03, -0.16], 1.12, 0.48, 26, 30),
    close("The bandwidth bar: how much of the peak a step uses", [0.53, 0.02, 0.15], 1.12, 0.42, 0, 46),
  ],
  [
    overview("Prefill writes the cache once, decode reads it every step"),
    close("Close up: write once, read every step", [0.53, 0.04, -0.46], 1.12, 0.45, -16, 34),
    close("The two bars start together, to scale", [0.53, 0.02, 0.08], 1.12, 0.59, 0, 50),
  ],
  [
    overview("A growing cache, cut into 16-token blocks"),
    close("Close up: a block fills, 16 tokens at a time", [0.5, 0.03, -0.48], 1.12, 0.42, -20, 36),
    close("From the side: full blocks land in free blocks of the pool", [0.52, 0.03, -0.08], 1.12, 0.59, 24, 32),
  ],
  [
    overview("One query against every cached key"),
    close("Close up: q · k, one key at a time", [0.45, 0.03, -0.32], 1.01, 0.56, -24, 34),
    close("From the side: softmax weighs every score together", [0.82, 0.03, -0.15], 0.59, 0.70, 28, 30),
    close("The output: the values summed with those weights", [0.4, 0.03, 0.22], 1.06, 0.42, 0, 46),
  ],
  [
    overview("Every key goes through the block table"),
    close("Close up: token, table entry, physical block", [0.5, 0.04, -0.4], 1.12, 0.50, -18, 36),
    close("From the side: the pool, scattered", [0.52, 0.03, -0.08], 1.12, 0.56, 24, 32),
    close("The rule, worked out for this token", [0.53, 0.02, 0.24], 1.12, 0.42, 0, 48),
  ],
  [
    overview("v1: three passes; v2: one chunk at a time"),
    close("Close up: v1 keeps every score and passes three times", [0.5, 0.03, -0.48], 1.12, 0.45, -18, 36),
    close("From the side: v2 streams keys through one chunk", [0.52, 0.04, -0.22], 1.12, 0.48, 24, 30),
    close("Attention per decode step, to scale", [0.53, 0.02, 0.18], 1.12, 0.48, 0, 48),
  ],
  [
    overview("One range at a time, then eight at once"),
    close("Close up: eight ranges and the merge", [0.53, 0.03, -0.43], 1.12, 0.45, -16, 36),
    close("From the side: how much of the GPU is busy", [0.52, 0.03, -0.17], 1.12, 0.48, 24, 30),
    close("One row at 2,048 tokens, to scale", [0.53, 0.02, 0.25], 1.12, 0.48, 0, 48),
  ],
];

const instructions = [
  "Watch all eight tokens enter the model together and fill K and V at once.",
  "Watch each step read every cached pair, then add one more.",
  "Watch both bars start together: the decode step is done almost at once.",
  "Watch each full block land in whichever pool block is free.",
  "Watch the scores fill one key at a time, then the softmax weigh them all.",
  "Drag the pink ball along the sequence to pick a token.",
  "Watch v1 pass over its full tray three times while v2 streams chunks.",
  "Watch the context run as one range, then as eight at once.",
];

const workbench: WorkbenchPage[] = [
  {
    title: "the block table",
    overlay: "sandbox-table",
    caption: "Step through it yourself: the block table",
    paragraphs: [
      "Set a context length and a block size. The context is cut into blocks of that many tokens, and each logical block is given a physical block somewhere in a pool of 64, in no particular order. Block 0 is the null block and is never handed out.",
      "The grid is the pool: a number is the logical block that sits there. The list beside it is the block table, logical to physical, which is all the kernel has to find a key. Click an entry to jump to that block.",
      "Press next block or play and watch the pool light up in logical order while the physical blocks are scattered. New random data reshuffles the blocks and the numbers.",
      "The values are illustrative: head dim 4 and seeded random numbers instead of the model's 64. The order of operations is the kernel's, one block per step here where the engine reads 256 tokens at a time.",
    ],
  },
  {
    title: "one block, one update",
    overlay: "sandbox-step",
    caption: "Step through it yourself: one block, one update",
    paragraphs: [
      "Each step reads one block: its scores q · k / √d, then the running max m, the running sum l and the accumulator acc are updated before the next block arrives.",
      "Watch the m row. When a block's best score beats every earlier one, m is raised and the rescale factor exp(m old - m new) drops below 1; l and acc are multiplied by it before the new weights are added.",
      "When m is unchanged the factor is exactly 1 and the new weights are simply added. Either way no block is read twice, which is what lets v2 keep only one chunk in shared memory.",
      "The table below counts the blocks read and the bytes they would cost at head dim 64: K and V, 2 bytes each in BF16, for one head.",
    ],
  },
  {
    title: "predict, then compare",
    overlay: "sandbox-check",
    caption: "Step through it yourself: predict, then compare",
    paragraphs: [
      "Answer the question first: a block arrives whose best score is bigger than any before. What happens to l and the accumulator?",
      "Then press jump to the next raised m and check your answer against the update on the previous page.",
      "The table compares acc / l so far with ordinary softmax attention computed in one go. Read every block and the two columns agree to float precision.",
      "That agreement is the point of the online softmax: the same output as the three-pass v1, from one pass over the keys.",
    ],
  },
];

function PagedAttentionOverlay({ kind, target }: OverlayProps) {
  return <PagedAttentionSandbox pages={{ table: kind === "sandbox-table" ? target : null, step: kind === "sandbox-step" ? target : null, check: kind === "sandbox-check" ? target : null }} />;
}

export const pagedAttentionChapter: ChapterDef = {
  slug: "paged-attention",
  number: 5,
  title: "paged attention",
  formula: "out = softmax(q · Kᵀ / √d) · V",
  steps: [...prologueSteps, ...attentionSteps],
  captions: [...prologueCaptions, ...attentionCaptions],
  exhibits: [0, 1, 2, 3, 4, 5, 6, 7],
  overlays: [null, null, null, null, null, null, null, null],
  instructions,
  beats,
  workbench,
  buildModel: (kit, fonts) => createPagedAttentionModel(kit, fonts, { ...prologueNumbers, ...attentionNumbers }),
  Overlay: PagedAttentionOverlay,
};
