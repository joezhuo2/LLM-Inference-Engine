"use client";

import { pagedKvAbout, pagedKvAboutCaption } from "@/components/kernels/paged-kv/about";
import { lengths, slots } from "@/components/kernels/paged-kv/batching";
import { first, fullContext, lookup, tightBlocks } from "@/components/kernels/paged-kv/figures";
import { batchingCaptions, batchingSteps, pagedKvCaptions, pagedKvSteps } from "@/components/kernels/paged-kv/story";
import { KvVisualizer } from "@/components/kernels/paged-kv/visualizer";
import { pagedKv } from "@/content/kernels/paged-kv";
import { kvDemo, kvReplay } from "@/lib/data";
import type { Beat } from "../book-scene";
import { type ChapterDef, close, type OverlayProps, overview, type WorkbenchPage, withLead } from "../chapter";
import { createPagedKvAboutModel } from "./paged-kv-about-model";
import { createPagedKvModel } from "./paged-kv-model";

const demo = kvDemo();
const B = kvReplay().meta.block_size;

const gridBeats = (first: string, second: string): Beat[] => [
  overview(first),
  close(second, [0.53, 0.03, -0.22], 0.86, 0.42, -18, 52),
  close("The queue: each request's height is its length in steps", [0.6, 0.04, -0.5], 0.7, 0.3, 16, 50),
  close("From the side: the counters and the step rail", [0.53, 0.02, 0.12], 0.86, 0.4, 0, 46),
];

const beats: Beat[][] = [
  gridBeats(`Static batching: ${lengths.length} requests in ${slots} slots`, "Close up: the pink slabs are slots that wait for the longest request"),
  gridBeats("Continuous batching: a free slot is refilled at once", "Close up: the waste is only at the end, when the queue is empty"),
  gridBeats(`A pool of ${tightBlocks} blocks: the engine's two rules`, "Close up: pink cells are recomputed steps, pink balls are evictions"),
  gridBeats("The same run: two counters to watch", "Close up: slots join and leave between steps"),
  [
    overview(`Reserved for ${fullContext.toLocaleString("en-US")} tokens against blocks taken as they grow`),
    close("Close up: the blue is all each row ever wrote", [0.55, 0.03, -0.44], 0.8, 0.34, -20, 54),
    close("From the side: blocks of 16, the last one partly full", [0.55, 0.02, -0.02], 0.8, 0.34, 22, 50),
  ],
  [
    overview("One block table into one shared pool"),
    close(`Close up: request ${first.id}'s table, logical block to physical block`, [0.53, 0.04, -0.46], 0.86, 0.3, -14, 54),
    close("From the side: the pink row is the slot, inside its block", [0.5, 0.04, -0.18], 0.66, 0.42, 24, 52),
    close("The rail: position is the only input", [0.53, 0.02, 0.2], 0.86, 0.36, 0, 44),
  ],
  [
    overview("write_kv: the K and V heads of one token go to one row"),
    close("Close up: K and V at the end of the QKV row", [0.75, 0.03, -0.5], 0.5, 0.26, 20, 52),
    close("From the side: every head lands in the same row of its tile", [0.52, 0.04, -0.24], 0.86, 0.36, -18, 52),
  ],
  [
    overview("One slot per decode step, one block per 16"),
    close(`Close up: block ${first.blocks[3].id} appears at position ${3 * B}`, [0.55, 0.03, -0.3], 0.78, 0.36, -18, 54),
    close("From the side: prompt slots low, decoded slots tall", [0.58, 0.03, -0.42], 0.7, 0.32, 24, 52),
  ],
  [
    overview("Freed blocks go out again first"),
    close("Close up: blue sinks, pink rises in the same place", [0.53, 0.03, -0.4], 0.86, 0.4, -16, 54),
    close("From the side: a full pool gives back its newest request", [0.53, 0.02, 0.1], 0.86, 0.32, 18, 58),
  ],
  [
    overview("The recorded run, replayed at half speed"),
    close("Close up: each block's height is how full it is", [0.53, 0.03, -0.4], 0.86, 0.42, -18, 54),
    close("From the side: blocks held over the run", [0.52, 0.04, 0.06], 0.86, 0.32, 14, 52),
  ],
];

const instructions = [
  "Drag the pink ball to step through; the pink slabs are wasted slot-steps.",
  "Watch a finished request's slot go to the front of the queue on the next step.",
  "Watch the pink balls: a full pool sends the newest request back to the queue.",
  "Watch the two counters, requests running and preemptions.",
  "Watch the blue rows fill against what each row reserved.",
  "Drag the pink ball to pick a position and follow its table entry into the pool.",
  `Watch each key and value land in row ${lookup.row} of its head's tile.`,
  `Drag the pink ball: a new block appears only at a multiple of ${B}.`,
  "Watch the freed blocks go to the next requests, then the newest request give its blocks back.",
  "Drag the pink ball along the run to stop at any step.",
];

const workbench: WorkbenchPage[] = [
  {
    title: "allocate it yourself",
    overlay: "visualizer-pool",
    caption: "Allocate it yourself: the pool",
    paragraphs: [
      `Three real chat requests, L1 to L3, were answered greedily by the engine on the ${demo.gpu.replace("NVIDIA GeForce ", "")} and recorded in results/kv_demo.json. On the right they are served from a 64-block pool by the rules of the BlockManager and the scheduler. The demo arrivals play first; press a + button to add another request at any time.`,
      "Each tile is a block of 16 slots. Its height is the number of slots written, and a dashed outline marks a block that is held but not written yet. Hover a block to see its tokens and where its keys and values sit in memory.",
      "A request is admitted only when blocks for its whole length are free, every decode step appends one slot, and when the pool is full the newest request is preempted and waits for room again.",
      `Switch the cache to contiguous to run the same arrivals with every request reserving its prompt plus the ${demo.max_new_tokens} tokens it may generate, in one run of blocks, and watch how few requests fit at once.`,
    ],
  },
  {
    title: "paged against contiguous",
    overlay: "visualizer-tables",
    caption: "Allocate it yourself: the tables",
    paragraphs: [
      "The table runs both caches side by side on the same arrivals, whichever one the pool shows: requests running, requests waiting for room, blocks held of 63, slots written, slots held but empty, and preemptions.",
      "Under paged, the list shows each running request's block table: logical block 0, 1, 2 and so on, each pointing to the physical block that holds it. A request's blocks are scattered wherever there was room when it grew.",
      "Under contiguous, each request holds one run of blocks, listed by its first and last block, and a waiting request needs that many blocks in a row: a pool with enough free blocks in all can still turn it away when they are scattered.",
      "Use step to move one decode step at a time, or replay demo to start over. Compare the slots held but empty row: blocks of 16 keep it under 16 slots per request, while a reservation keeps every slot a request might use.",
    ],
  },
];

function PagedKvOverlay({ kind, target }: OverlayProps) {
  return <KvVisualizer pages={{ pool: kind === "visualizer-pool" ? target : null, tables: kind === "visualizer-tables" ? target : null }} />;
}

const chapter: ChapterDef = {
  slug: "paged-kv",
  number: 4,
  title: "paged KV cache",
  formula: pagedKv.formula,
  steps: [...batchingSteps, ...pagedKvSteps],
  captions: [...batchingCaptions, ...pagedKvCaptions],
  exhibits: [0, 0, 0, 0, 1, 2, 3, 4, 5, 6],
  overlays: [null, null, null, null, null, null, null, null, null, null],
  instructions,
  beats,
  workbench,
  buildModel: createPagedKvModel,
  Overlay: PagedKvOverlay,
};

export const pagedKvChapter = withLead(chapter, {
  step: pagedKvAbout,
  caption: pagedKvAboutCaption,
  instruction: `Drag the pink ball to set how many tokens the request has; a new block is taken every ${B}.`,
  beats: [
    overview("What a KV cache keeps, and what paging it means"),
    close("Close up: one column per token, its key in front and its value behind", [0.5, 0.03, -0.37], 0.6, 0.32, -18, 44),
    close("From the side: the block table points each logical block at a free physical one", [0.5, 0.05, -0.25], 0.86, 0.42, 24, 36),
    close(`The rail: drag the tokens from 1 to ${4 * B}`, [0.5, 0.02, 0.12], 0.9, 0.34, 0, 46),
  ],
  buildModel: createPagedKvAboutModel,
});
