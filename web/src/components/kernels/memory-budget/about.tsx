"use client";

import { budget, gb, memFmt } from "@/content/kernels/memory-budget";
import { StickyStory, type StoryStep } from "../sticky-story";
import { TankFigure } from "./figures";

const fmt = memFmt;

export const memoryAbout: StoryStep = {
  id: "about",
  title: "what the memory has to hold",
  body: (
    <>
      <p>
        Speed was the subject so far. This chapter is about room. The GPU has {gb(budget.vram)} of memory, and two things must live in it for a token to be decoded: the model&apos;s weights, and the KV cache, the keys and values of every token the running requests have seen.
      </p>
      <p>
        The weights are a fixed cost. TinyLlama&apos;s are <span className="text-accent">{gb(budget.weights)}</span>, {fmt(budget.weightsPct)}% of the memory, and they are the same for one request or a hundred. The cache is the part that grows: every token of every request adds to it, and it is what a longer context or a larger batch has to pay for.
      </p>
      <p>
        The engine settles the split once, at start. It loads the weights, asks the GPU how much memory is free, and takes {budget.fractionPct}% of that as one pool of cache blocks: {fmt(budget.poolBlocks)} blocks, {gb(budget.pool)}. Nothing is allocated later, so a request can never fail on memory halfway through an answer. It can only be made to wait for blocks.
      </p>
      <p>The budget is then one line: the tokens in all running requests, times the bytes one token leaves behind, must fit in the pool.</p>
    </>
  ),
};

export const memoryAboutCaption = `${gb(budget.vram)} from docs/hardware.md; weights calculated from the model's shapes at 2 bytes per parameter; the pool is the ${fmt(budget.poolBlocks)} blocks the engine allocated in the recorded run of ${budget.poolDate} (measured); kept free is the other tenth of what was free; everything else is the remainder, calculated`;

export function MemoryAboutStory() {
  return <StickyStory kicker="the idea" steps={[memoryAbout]} caption={() => memoryAboutCaption} figure={() => <TankFigure />} />;
}
