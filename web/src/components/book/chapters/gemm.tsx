"use client";

import { createPortal } from "react-dom";
import { TilingSandbox } from "@/components/kernels/gemm/sandbox";
import { gemmCaptions, gemmNumbers, gemmSteps, RooflinePicker } from "@/components/kernels/gemm/story";
import { GemmStream } from "@/components/kernels/gemm/stream";
import { gemmAbout, gemmAboutCaption } from "@/components/kernels/gemm/about";
import { gemm, prefillRoof } from "@/content/kernels/gemm";
import { hardware } from "@/lib/data";
import type { Beat } from "../book-scene";
import { type ChapterDef, close, flat, type OverlayProps, overview, type WorkbenchPage, withLead } from "../chapter";
import { createGemmAboutModel } from "./gemm-about-model";
import { createGemmModel } from "./gemm-model";

const sms = hardware().sms;

const beats: Beat[][] = [
  [
    overview("The five projections of one decode step"),
    close("Close up: gate/up, the biggest weights and the longest time", [0.5, 0.1, -0.36], 0.5, 0.42, -22, 34),
    close("From the side: the ink and blue towers rise to the same shape", [0.5, 0.08, -0.34], 0.95, 0.5, 30, 32),
    close("The LM head: one matrix, read once per token", [0.84, 0.05, -0.34], 0.4, 0.38, 24, 36),
  ],
  [
    overview("One row of W in three places"),
    close("Close up: the twelve addresses, one row after the other", [0.53, 0.02, -0.5], 0.92, 0.3, 0, 42),
    close("From the side: row r of W is column r of what cuBLAS sees", [0.5, 0.03, -0.2], 0.9, 0.45, -24, 50),
    close("The call: one cublasGemmEx, nothing copied", [0.5, 0.02, 0.24], 0.92, 0.36, 0, 72),
  ],
  [
    overview("Batch 1: every weight row streams past x once"),
    close("Close up: a weight row read and one value of y", [0.6, 0.02, -0.22], 0.6, 0.4, -24, 34),
    close("From the side: x stays put, the weights stream", [0.42, 0.03, -0.32], 0.7, 0.42, 30, 24),
  ],
  [
    overview("Five GEMVs at the measured roof"),
    close("Close up: the LM head, at 102%, pokes through the glass", [0.66, 0.14, -0.3], 0.6, 0.5, -18, 34),
    close("From the side: attention, far below the roof", [0.78, 0.1, -0.28], 0.6, 0.5, 28, 34),
  ],
  [
    overview("One weight row read once, used by every row of x"),
    close("Close up: sixteen rows share one read", [0.42, 0.03, -0.38], 0.7, 0.42, -22, 34),
    close("The rail: drag the batch from 1 to 16", [0.5, 0.02, 0.06], 0.9, 0.36, 0, 44),
  ],
  [flat(`Prefill at ${prefillRoof.rows.toLocaleString("en-US")} rows: past the ridge`)],
  [flat("Switch between 1 row, 16 rows and prefill")],
];

const instructions = [
  "Watch the ink and blue towers of each projection rise to the same shape.",
  "Watch one row of W lift in all three places: the same four addresses.",
  "Watch each weight row stream past x and leave one value of y.",
  "Watch the five blue towers reach the glass roof while attention stays low.",
  "Drag the pink ball along the rail to change the rows from 1 to 16.",
];

const workbench: WorkbenchPage[] = [
  {
    title: "stream it yourself",
    overlay: "stream-view",
    caption: "Stream it yourself: the weights",
    paragraphs: [
      `The board is one weight matrix of a decode step, gate/up to start with, cut into 16 by 16 tiles. Press play and the GPU's ${sms} SMs read the tiles from DRAM in waves, one tile per SM per wave, colored by the SM that read it.`,
      "When a tile arrives its slab rises: that is its bytes, the same at any batch. The pillar on top is how many multiply-adds each of those weights feeds, which is the number of rows M, one step up for every doubling.",
      "The strip along the front left is x, one row per request in the batch; the strip along the front right is y, and each of its cells fills as the K loop works through the columns of its rows. Hover a tile for its rows, columns, bytes and the FLOPs it feeds.",
      "Switch rows M between 1, 16 and 512 while it plays. The slabs do not change, only the pillars, and at 512 they rise through the dashed ridge. The board is downsampled and which SM reads which tile is illustrative; cuBLAS picks its own.",
    ],
  },
  {
    title: "the bytes and the ceilings",
    overlay: "stream-table",
    caption: "Stream it yourself: the bytes and the ceilings",
    paragraphs: [
      "Answer the prediction first. The table then counts one whole decode step for the matrix you picked at all three batch sizes: the weight bytes of one layer, the x and y bytes, the bytes of the step across all ${gemmNumbers.projections[0].per} layers (the LM head is read once), and the FLOPs.",
      `FLOPs divided by bytes is the arithmetic intensity. The ridge at ${Math.round(gemmNumbers.ridge)} FLOP per byte is where the two measured ceilings meet, ${gemmNumbers.peak.gbs} GB/s and ${gemmNumbers.peak.tflops} TFLOP/s: under it the bytes set the time, over it the tensor cores do.`,
      "Each weight costs 2 bytes and feeds 2 FLOPs per row of x, so the intensity is about M. At 1 and 16 rows that is far under the ridge, and the measured time per step barely moves: 16 tokens for the price of one.",
      "At 512 rows the compute bar passes the memory bar and the matrix is compute bound. The profiles only measured 1 and 16 rows, so the table says not measured there; the two roof times are computed from the measured ceilings.",
    ],
  },
  {
    title: "one thread block, step by step",
    overlay: "sandbox-one",
    caption: "Tile it yourself: one thread block",
    paragraphs: [
      "Answer the prediction first: at one row, does a bigger output tile cut the bytes read from DRAM? Then pick a matrix, the rows M, the k step and the tile, and play or step the K loop.",
      "The figure is one thread block. For each step of the K loop it loads a tile of x (the A panel) and a tile of W (the B panel) into shared memory, and every value it loaded feeds a whole row or column of the output tile, whose accumulators sit in registers.",
      "Watch the line under the block: each loaded x value feeds as many outputs as the tile is wide, each W value feeds as many as there are live rows. At one row a W value is used once however big the tile, so the extra tile rows are padding.",
      "cuBLAS picks its own tile sizes and this engine does not touch them, so this is a model of the idea, not of the kernel. The shapes are TinyLlama's.",
    ],
  },
  {
    title: "what the tile buys",
    overlay: "sandbox-two",
    caption: "Tile it yourself: the bytes and the ceilings",
    paragraphs: [
      "The same controls as on the last spread, now with the totals for every block. The bars compare the naive DRAM traffic, where every multiply-add reads its two operands, with the tiled traffic, and show how much shared memory one block needs.",
      "Set rows to 512 and change the tile: the reuse climbs and the FLOPs per byte with it, until the slower ceiling stops being bandwidth and becomes the tensor cores. Push the tile and the k step too far and the block no longer fits in shared memory.",
      "At one or sixteen rows no tile gets near the ridge, which is the decode case this chapter measured: the time is the time it takes to read the weights.",
      `The byte counts are computed from the tile sizes, counting every tile load as DRAM traffic with no L2 cache. The two ceilings, ${gemmNumbers.peak.gbs} GB/s and ${gemmNumbers.peak.tflops} TFLOP/s, are measured on the same GPU.`,
    ],
  },
];

function GemmOverlay({ kind, target, narrow }: OverlayProps) {
  return (
    <>
      {kind === "prefill-roofline" &&
        target &&
        createPortal(
          <div className={narrow ? undefined : "h-[34rem]"}>
            <RooflinePicker initial="prefill" />
          </div>,
          target,
        )}
      {kind === "roofline" &&
        target &&
        createPortal(
          <div className={narrow ? undefined : "h-[34rem]"}>
            <RooflinePicker />
          </div>,
          target,
        )}
      <GemmStream pages={{ view: kind === "stream-view" ? target : null, table: kind === "stream-table" ? target : null }} />
      <TilingSandbox pages={{ one: kind === "sandbox-one" ? target : null, two: kind === "sandbox-two" ? target : null }} />
    </>
  );
}

const chapter: ChapterDef = {
  slug: "gemm",
  number: 2,
  title: "GEMM",
  formula: gemm.formula,
  steps: gemmSteps,
  captions: gemmCaptions,
  exhibits: [0, 1, 2, 3, 4, null, null],
  overlays: [null, null, null, null, null, "prefill-roofline", "roofline"],
  instructions,
  beats,
  workbench,
  buildModel: (kit, fonts) => createGemmModel(kit, fonts, gemmNumbers),
  Overlay: GemmOverlay,
};

export const gemmChapter = withLead(chapter, {
  step: gemmAbout,
  caption: gemmAboutCaption,
  instruction: "Drag the pink ball to add rows to x: one row is a GEMV, more rows make it a GEMM.",
  beats: [
    overview("What GEMM and GEMV stand for, and what they compute"),
    close("Close up: one column of Wᵀ meets every row of x", [0.5, 0.03, -0.24], 0.62, 0.36, -20, 40),
    close("From the side: each output rises as its dot product finishes", [0.62, 0.04, -0.22], 0.6, 0.38, 26, 30),
    close("The rail: drag the rows of x from 1 to 4", [0.5, 0.02, 0.16], 0.9, 0.34, 0, 46),
  ],
  buildModel: createGemmAboutModel,
});
