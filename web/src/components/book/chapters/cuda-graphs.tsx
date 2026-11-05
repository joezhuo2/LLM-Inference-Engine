"use client";

import { createPortal } from "react-dom";
import { cudaGraphsCaptions, cudaGraphsNumbers, cudaGraphsSteps, ReplayPicker } from "@/components/kernels/cuda-graphs/story";
import { cudaGraphs } from "@/content/kernels/cuda-graphs";
import type { Beat } from "../book-scene";
import { type ChapterDef, close, flat, type OverlayProps, overview } from "../chapter";
import { createCudaGraphsModel } from "./cuda-graphs-model";

const beats: Beat[][] = [
  [
    overview("Layer 11 of an eager decode step, from Nsight Systems"),
    close("Close up: the CPU launches kernels layers ahead", [0.42, 0.03, -0.42], 0.72, 0.4, -16, 46),
    close("From the side: the pink slivers are the GPU waiting", [0.42, 0.04, -0.12], 0.8, 0.4, 22, 30),
    close("The tally: idle time in this layer", [0.5, 0.05, 0.16], 0.7, 0.3, 0, 44),
  ],
  [
    overview("Three kernels and the gaps between them"),
    close("Close up: a gap of about 3 µs before every kernel", [0.36, 0.04, -0.36], 0.74, 0.46, -18, 36),
    close("From the side: the idle time stacks on top of the busy time", [0.82, 0.13, -0.4], 0.4, 0.44, 22, 26),
  ],
  [
    overview("Stream capture: record once, launch once"),
    close("Close up: each launch becomes a node of the graph", [0.5, 0.04, -0.1], 0.86, 0.42, -18, 40),
    close("From the side: the graph sealed for replay", [0.5, 0.06, -0.05], 0.7, 0.36, 26, 22),
  ],
  [
    overview("stage: new values into the same buffer"),
    close("Close up: the padded row in pink, slot -1", [0.45, 0.05, -0.28], 0.8, 0.44, -16, 40),
    close("From the side: the kernels read fixed offsets", [0.5, 0.04, 0.0], 0.8, 0.4, 22, 26),
  ],
  [
    overview("11 graphs: buckets by split counts"),
    close("Close up: the 1-row graphs, one per split count", [0.5, 0.04, -0.44], 0.8, 0.3, -18, 40),
    close("The rail: the longest context picks the graph", [0.5, 0.04, 0.02], 0.84, 0.3, 0, 44),
    close("From the side: 5 graphs against 11", [0.5, 0.05, 0.24], 0.8, 0.3, 20, 34),
  ],
  [flat("Switch eager and graph, pick a batch")],
];

const instructions = [
  "Watch the GPU lane fill and the pink gaps add up.",
  "Watch each gap open and the idle column climb.",
  "Watch the launches drop into the graph and the lid close.",
  "Watch the rows land at fixed offsets and the kernels read them.",
  "Drag the pink ball along the rail to change the context.",
];

function CudaGraphsOverlay({ kind, target }: OverlayProps) {
  return kind === "replay" && target && createPortal(<ReplayPicker />, target);
}

export const cudaGraphsChapter: ChapterDef = {
  slug: "cuda-graphs",
  number: 11,
  title: "CUDA Graphs",
  formula: cudaGraphs.formula,
  steps: cudaGraphsSteps,
  captions: cudaGraphsCaptions,
  exhibits: [0, 1, 2, 3, 4, null],
  overlays: [null, null, null, null, null, "replay"],
  instructions,
  beats,
  workbench: [],
  buildModel: (kit, fonts) => createCudaGraphsModel(kit, fonts, cudaGraphsNumbers),
  Overlay: CudaGraphsOverlay,
};
