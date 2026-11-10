"use client";

import { flashShape } from "@/content/kernels/flash-prefill";
import { modelShape } from "@/lib/data";
import { StickyStory, type StoryStep } from "../sticky-story";
import { TileSweepFigure } from "./figures";

const positions = modelShape().max_positions;

export const flashAboutNumbers = { tile: flashShape.tile, tableMb: (positions * positions * 4) / 2 ** 20, tileBytes: flashShape.tile * flashShape.tile * 4, positions };

export const flashPrefillAbout: StoryStep = {
  id: "about",
  title: "what FlashAttention is",
  body: (
    <>
      <p>
        Prefill computes attention for every token of a prompt at once. Each token is a query row, and row t scores itself against the keys of tokens 0 to t. Laid out as a table, query rows down and keys across, the scores fill a triangle, and the triangle grows with the square of the prompt.
      </p>
      <p>
        The direct way computes the whole table, stores it, runs a softmax along every row and then multiplies by the values. For a long prompt the table is far larger than anything that fits on the chip, so it goes out to DRAM and comes back.
      </p>
      <p>
        FlashAttention cuts the table into tiles: a block of query rows against a block of keys. One tile of scores is computed, used and dropped while it is still on the chip. The online softmax of the attention v1 to v2 chapter is what allows it: every query row keeps a running maximum, a running sum and a running weighted sum of the values, and rescales them when a later tile brings a larger score.
      </p>
      <p>
        Two things follow. A tile of keys and values is loaded once for a whole block of query rows instead of once per row. And a tile of queries times a tile of keys is a small matrix multiply, the shape the tensor cores are built for.
      </p>
    </>
  ),
};

export const flashPrefillAboutCaption = `illustrative: a ${8 * flashShape.tile}-token prompt drawn as 8 by 8 tiles of ${flashShape.tile}; calculated: the scores of a ${positions.toLocaleString("en-US")}-token prompt are ${flashAboutNumbers.tableMb} MB per query head in FP32, against ${flashAboutNumbers.tileBytes.toLocaleString("en-US")} bytes for one tile`;

export function FlashAboutFigure() {
  return <TileSweepFigure mode="about" tile={flashShape.tile} tableMb={flashAboutNumbers.tableMb} tileBytes={flashAboutNumbers.tileBytes} />;
}

export function FlashPrefillAboutStory() {
  return <StickyStory kicker="the idea" steps={[flashPrefillAbout]} caption={() => flashPrefillAboutCaption} figure={() => <FlashAboutFigure />} />;
}
