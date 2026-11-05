"use client";

import { createPortal } from "react-dom";
import { TilingSandbox } from "@/components/kernels/gemm/sandbox";
import { gemmCaptions, gemmNumbers, gemmSteps, RooflinePicker } from "@/components/kernels/gemm/story";
import { gemm } from "@/content/kernels/gemm";
import type { Beat } from "../book-scene";
import { type ChapterDef, close, flat, type OverlayProps, overview, type WorkbenchPage } from "../chapter";
import { createGemmModel } from "./gemm-model";

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
  [flat("Switch between 1 and 16 rows")],
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
      {kind === "roofline" &&
        target &&
        createPortal(
          <div className={narrow ? undefined : "h-[34rem]"}>
            <RooflinePicker />
          </div>,
          target,
        )}
      <TilingSandbox pages={{ one: kind === "sandbox-one" ? target : null, two: kind === "sandbox-two" ? target : null }} />
    </>
  );
}

export const gemmChapter: ChapterDef = {
  slug: "gemm",
  number: 2,
  title: "GEMM",
  formula: gemm.formula,
  steps: gemmSteps,
  captions: gemmCaptions,
  exhibits: [0, 1, 2, 3, 4, null],
  overlays: [null, null, null, null, null, "roofline"],
  instructions,
  beats,
  workbench,
  buildModel: (kit, fonts) => createGemmModel(kit, fonts, gemmNumbers),
  Overlay: GemmOverlay,
};
