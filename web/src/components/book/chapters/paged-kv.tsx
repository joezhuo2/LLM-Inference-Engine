"use client";

import { chapterNumber } from "@/content/book-order";
import { pagedKvAbout, pagedKvAboutCaption } from "@/components/kernels/paged-kv/about";
import { first, fullContext, lookup } from "@/components/kernels/paged-kv/figures";
import { pagedKvCaptions, pagedKvSteps } from "@/components/kernels/paged-kv/story";
import { pagedKv } from "@/content/kernels/paged-kv";
import { kvReplay } from "@/lib/data";
import type { Beat } from "../book-scene";
import { type ChapterDef, close, overview, withLead } from "../chapter";
import { createPagedKvAboutModel } from "./paged-kv-about-model";
import { createPagedKvModel } from "./paged-kv-model";

const B = kvReplay().meta.block_size;

const beats: Beat[][] = [
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
    overview("write_kv: the K and V heads of one token go to one row"),
    close("Close up: K and V at the end of the QKV row", [0.75, 0.03, -0.5], 0.5, 0.26, 20, 52),
    close("From the side: every head lands in the same row of its tile", [0.52, 0.04, -0.24], 0.86, 0.36, -18, 52),
  ],
  [
    overview("The recorded run, replayed at half speed"),
    close("Close up: each block's height is how full it is", [0.53, 0.03, -0.4], 0.86, 0.42, -18, 54),
    close("From the side: blocks held over the run", [0.52, 0.04, 0.06], 0.86, 0.32, 14, 52),
  ],
];

const instructions = [
  "Watch the blue rows fill against what each row reserved.",
  "Drag the pink ball to pick a position and follow its table entry into the pool.",
  `Drag the pink ball: a new block appears only at a multiple of ${B}.`,
  "Watch the freed blocks go to the next requests, then the newest request give its blocks back.",
  `Watch each key and value land in row ${lookup.row} of its head's tile.`,
  "Drag the pink ball along the run to stop at any step.",
];

const chapter: ChapterDef = {
  slug: "paged-kv",
  number: chapterNumber("paged-kv"),
  title: "paged KV cache",
  formula: pagedKv.formula,
  steps: pagedKvSteps,
  captions: pagedKvCaptions,
  exhibits: [0, 1, 3, 4, 2, 5],
  overlays: [null, null, null, null, null, null],
  instructions,
  beats,
  workbench: [],
  buildModel: createPagedKvModel,
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
