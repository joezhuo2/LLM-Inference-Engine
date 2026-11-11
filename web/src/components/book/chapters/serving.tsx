"use client";

import { createPortal } from "react-dom";
import { lengths, slots } from "@/components/kernels/serving/batching";
import { servingAbout, servingAboutCaption } from "@/components/kernels/serving/about";
import { LoadFigure, tightBlocks } from "@/components/kernels/serving/figures";
import { queueConfig } from "@/components/kernels/serving/queue";
import { replayFacts, ServingReplay } from "@/components/kernels/serving/replay";
import { servingCaptions, servingSteps } from "@/components/kernels/serving/story";
import { KvVisualizer } from "@/components/kernels/serving/visualizer";
import { chapterNumber } from "@/content/book-order";
import { blockTokens, fmt, fullContext, serving, sweepRows } from "@/content/kernels/serving";
import { kvDemo } from "@/lib/data";
import type { Beat } from "../book-scene";
import { type ChapterDef, close, flat, type OverlayProps, overview, type WorkbenchPage, withLead } from "../chapter";
import { createServingAboutModel } from "./serving-about-model";
import { createServingModel } from "./serving-model";

const demo = kvDemo();

const queueBeats = (first: string, second: string): Beat[] => [
  overview(first),
  close(second, [0.52, 0.03, -0.4], 0.9, 0.36, -16, 48),
  close("From the side: the pool, a letter under every block that is held", [0.52, 0.03, -0.16], 0.9, 0.34, 22, 44),
  close("The note: what the scheduler did in this step, and why", [0.52, 0.03, 0.1], 0.9, 0.36, 0, 50),
];

const gridBeats = (first: string, second: string): Beat[] => [
  overview(first),
  close(second, [0.53, 0.03, -0.22], 0.86, 0.42, -18, 52),
  close("The queue: each request's height is its length in steps", [0.6, 0.04, -0.5], 0.7, 0.3, 16, 50),
  close("From the side: the counters and the step rail", [0.53, 0.02, 0.12], 0.86, 0.4, 0, 46),
];

const beats: Beat[][] = [
  [
    overview("One request growing into a full context, block by block"),
    close(`Close up: a tray fills in ${blockTokens} steps, then the next one starts`, [0.4, 0.03, -0.4], 0.6, 0.3, -18, 46),
    close("From the side: the pool, with one full context in pink", [0.42, 0.03, 0.08], 0.7, 0.3, 20, 44),
    close("The rail: drag the length of the request", [0.52, 0.02, 0.24], 0.9, 0.34, 0, 48),
  ],
  [
    overview("The same pool counted two ways"),
    close(`Close up: reserving ${fmt(fullContext)} tokens takes a whole tray per request`, [0.52, 0.03, -0.46], 0.9, 0.3, -16, 50),
    close("From the side: the blocks actually held, packed into the first trays", [0.42, 0.03, -0.17], 0.7, 0.3, 22, 46),
    close("The rail: drag the number of requests", [0.52, 0.02, 0.14], 0.9, 0.36, 0, 48),
  ],
  [
    overview(`max_batch ${sweepRows[0].batch} to ${sweepRows[sweepRows.length - 1].batch}: the worst case against the pool`),
    close("Close up: pink is every request at the full context", [0.52, 0.03, -0.46], 0.9, 0.3, -16, 50),
    close("From the side: blue is the most a measured run held", [0.42, 0.03, -0.17], 0.7, 0.3, 22, 46),
    close("The rail: each max_batch with its measured step time", [0.52, 0.02, 0.14], 0.9, 0.36, 0, 48),
  ],
  queueBeats("Admission: the front of the queue, a free row, free blocks", "Close up: the queue above, the four rows of the running list below"),
  queueBeats("Preemption: no block is free, the newest request goes back", "Close up: the victim turns pink and returns to the front of the queue"),
  gridBeats(`Static batching: ${lengths.length} requests in ${slots} slots`, "Close up: the pink slabs are slots that wait for the longest request"),
  gridBeats("Continuous batching: a free slot is refilled at once", "Close up: the waste is only at the end, when the queue is empty"),
  gridBeats(`A pool of ${tightBlocks} blocks: admission and preemption in the grid`, "Close up: pink cells are recomputed steps, pink balls are evictions"),
  [flat("Goodput against the offered load")],
  [flat("Time per token against the offered load")],
  [flat("Allocate it yourself: the pool")],
  [flat("Allocate it yourself: the tables")],
  [flat("Replay a recorded run")],
];

const instructions = [
  "Drag the pink ball to set the request's length and watch the trays fill.",
  "Drag the pink ball: pink is what a reservation takes, blue what the requests hold.",
  "Drag the pink ball along the max_batch settings.",
  "Watch the queue: a request moves down only when a row and its blocks are free.",
  "Watch the pink request lose its blocks and return to the front of the queue.",
  "Drag the pink ball to step through; the pink slabs are wasted slot-steps.",
  "Watch a finished request's slot go to the front of the queue on the next step.",
  "Watch the pink balls: a full pool sends the newest request back to the queue.",
];

const workbench: WorkbenchPage[] = [
  {
    title: "allocate it yourself",
    overlay: "visualizer-pool",
    caption: "Allocate it yourself: the pool",
    paragraphs: [
      `Three real chat requests, L1 to L3, were answered greedily by the engine on the ${demo.gpu.replace("NVIDIA GeForce ", "")} and recorded in results/kv_demo.json. On the right they are served from a small block pool by the rules of the BlockManager and the scheduler. The demo arrivals play first; press a + button to add another request at any time.`,
      "Each tile is a block of 16 slots. Its height is the number of slots written, and a dashed outline marks a block that is held but not written yet. Hover a block to see its tokens and where its keys and values sit in memory.",
      "A request is admitted only when blocks for its whole length are free, every decode step appends one slot, and when the pool is full the newest request is preempted and waits for room again. This model has no max_batch, so its pool is the only limit.",
      `Switch the cache to contiguous to run the same arrivals with every request reserving its prompt plus the ${demo.max_new_tokens} tokens it may generate, in one run of blocks, and watch how few requests fit at once.`,
    ],
  },
  {
    title: "paged against contiguous",
    overlay: "visualizer-tables",
    caption: "Allocate it yourself: the tables",
    paragraphs: [
      "The table runs both caches side by side on the same arrivals, whichever one the pool shows: requests running, requests waiting for room, blocks held of 63, slots written, slots held but empty, and preemptions.",
      "Under paged, the list shows each running request's block table: its logical blocks in order, each pointing to the physical block that holds it. A request's blocks are scattered wherever there was room when it grew.",
      "Under contiguous, each request holds one run of blocks, listed by its first and last block, and a waiting request needs that many blocks in a row: a pool with enough free blocks in all can still turn it away when they are scattered.",
      "Use step to move one decode step at a time, or replay demo to start over. Compare the slots held but empty row: blocks of 16 keep it under one block per request, while a reservation keeps every slot a request might use.",
    ],
  },
  {
    title: "replay a recorded run",
    overlay: "replay",
    caption: "Replay a recorded run",
    paragraphs: [
      replayFacts ? `This is a run the engine recorded with engine bench --trace: ${replayFacts.requests} chat requests arriving at ${replayFacts.rate} per second, ${fmt(replayFacts.steps)} scheduler steps. Press play, or drag the slider to any step.` : "The engine can record a run with engine bench --trace and replay it here, step by step.",
      "The batch timeline has one row per step. Its length is the time the step took, its colour says whether the step was a prefill of newly admitted requests or a decode step of everyone running, and the dots beside it are the requests in the step. Watch a long prefill row appear each time a request arrives, with the decoders waiting for it.",
      replayFacts ? `The pool shows which blocks each request holds, a colour per request. At most ${replayFacts.seen} requests ran at once here, under a max_batch of ${replayFacts.maxBatch}, and they never held more than a sliver of the ${fmt(replayFacts.blocks)} blocks: this run was limited by how fast requests arrived, not by rows or memory.` : "The pool shows which blocks each request holds, a colour per request.",
      "Step through the end of a request: its blocks all go back at once, and the next request that needs one takes the lowest free block.",
    ],
  },
];

function ServingOverlay({ kind, target, narrow }: OverlayProps) {
  return (
    <>
      <KvVisualizer pages={{ pool: kind === "visualizer-pool" ? target : null, tables: kind === "visualizer-tables" ? target : null }} />
      <ServingReplay pages={{ replay: kind === "replay" ? target : null }} />
      {target && (kind === "goodput" || kind === "tpot") && createPortal(<div className={narrow ? "w-full" : "pt-10"}><LoadFigure kind={kind} /></div>, target)}
    </>
  );
}

const chapter: ChapterDef = {
  slug: "serving",
  number: chapterNumber("serving"),
  title: serving.title,
  formula: serving.formula,
  steps: servingSteps,
  captions: servingCaptions,
  exhibits: [0, 1, 1, 2, 2, 3, 3, 3, null, null],
  overlays: [null, null, null, null, null, null, null, null, "goodput", "tpot"],
  instructions,
  beats,
  workbench,
  buildModel: createServingModel,
  Overlay: ServingOverlay,
};

export const servingChapter = withLead(chapter, {
  step: servingAbout,
  caption: servingAboutCaption,
  instruction: "Drag the pink ball to step through the run: requests wait, run, finish, and one is sent back.",
  beats: [
    overview("What a scheduler decides before every step"),
    close("Close up: waiting above, running below, a request's height is its tokens", [0.52, 0.03, -0.4], 0.9, 0.36, -16, 48),
    close("From the side: the pool the running requests share", [0.52, 0.03, -0.16], 0.9, 0.34, 22, 44),
    close(`The rail: every step of a run with max_batch ${queueConfig.maxBatch}`, [0.52, 0.02, 0.16], 0.9, 0.36, 0, 48),
  ],
  buildModel: createServingAboutModel,
});
