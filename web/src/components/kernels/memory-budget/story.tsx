"use client";

import Link from "next/link";
import { batchRows, budget, fitOf, gb, lastRow, memFmt, sweepPeak } from "@/content/kernels/memory-budget";
import { published } from "@/content/published";
import { StickyStory, type StoryStep } from "../sticky-story";
import { BudgetFigure, BytesFigure, FitFigure } from "./figures";

const fmt = memFmt;
const chapter = (slug: string, text: string) => (
  <Link href={`/kernels/${slug}`} className="link-underline text-accent">
    {text}
  </Link>
);
const l3 = published.llama3;
const l2 = published.llama2;
const big = fitOf(l3.model, 16);
const low = fitOf(l3.model, 4);
const wide = fitOf(l2.model, 4);
const at16 = batchRows.find((r) => r.batch === budget.maxBatch);

const steps: StoryStep[] = [
  {
    id: "bytes",
    title: `a cached token costs ${fmt(budget.tokenBytes)} bytes, whatever it says`,
    body: (
      <>
        <p>
          When a token goes through a layer, attention stores its key and its value so that later tokens can look back at it. That happens in all {budget.layers} layers. A key is {budget.kvHeads} heads of {budget.headDim} numbers, a value the same, and a number is 2 bytes. Multiply it out: <span className="text-accent">{fmt(budget.tokenBytes)} bytes</span> per token.
        </p>
        <p>
          The size does not depend on the token. A comma costs what a rare word costs. So the cache of a request is its length times a constant, and a full context of {fmt(budget.fullContext)} tokens is {fmt(budget.fullMb, 1)} MB. The engine hands the memory out in blocks of {budget.blockTokens} tokens, {fmt(budget.blockBytes / 1e3)} KB each; {chapter("paged-kv", "the paged KV cache")} is about why.
        </p>
        <p>
          One design choice is hidden in the count of heads. TinyLlama has {budget.heads} query heads but only {budget.kvHeads} key and value heads, which the query heads share in groups. With a key and a value per query head the same token would cost {fmt(budget.wideTokenBytes)} bytes, {fmt(budget.wideTokenBytes / budget.tokenBytes)} times as much.
        </p>
      </>
    ),
  },
  {
    id: "budget",
    title: "batch size and context length share one pool",
    body: (
      <>
        <p>
          The pool holds {fmt(budget.poolTokens)} tokens. It does not care whose they are. One request with a long context and many requests with short ones draw on the same blocks, so the two numbers people ask about, how many requests and how long each, are one budget: <span className="text-accent">batch × context ≤ {fmt(budget.poolTokens)}</span>.
        </p>
        <p>
          For TinyLlama on this GPU the budget is generous. The model itself stops at {fmt(budget.fullContext)} positions, and {budget.reservedFit} requests could each reach that before the pool ran out. {at16 ? `At the ${budget.maxBatch} requests the benchmarks run, full contexts would use ${fmt(at16.poolPct)}% of the pool.` : ""} Past {budget.reservedFit} the pool becomes the limit: {lastRow.batch} requests can each reach {fmt(lastRow.context)} tokens, not {fmt(budget.fullContext)}.
        </p>
        <p>
          In practice requests are far shorter than the limit. The batch sweep ran {sweepPeak.batch} at once and they held at most {fmt(sweepPeak.blocks)} blocks, {fmt(sweepPeak.pct)}% of the pool (measured). So here the batch is limited by the time of a step, not by memory. When the pool does fill, the scheduler makes a request wait or takes its blocks back; {chapter("serving", "the next chapter")} is about those rules.
        </p>
      </>
    ),
  },
  {
    id: "bigger",
    title: "a larger model squeezes the pool from both sides",
    body: (
      <>
        <p>
          TinyLlama is small, which is why the pool looked endless. Take a model people serve, {l3.model}, on the same GPU. In BF16 its weights are {gb(big.weights, 1)}: <span className="text-accent">{gb(big.over, 1)} more than there is room for</span>. Before any question of batch or context, the weights have to fit.
        </p>
        <p>
          Store each weight in 4 bits and they take {gb(low.weights, 1)}. Now there is a pool, but a smaller one, {gb(low.pool, 1)}, and each token costs more, {fmt(low.perToken)} bytes, because the model has more layers, more KV heads and wider heads. The pool holds {fmt(low.tokens)} tokens: {fmt(low.contexts)} contexts of {fmt(budget.fullContext)} tokens where TinyLlama had {budget.reservedFit}. Both sides of the budget moved against the batch.
        </p>
        <p>
          {l2.model} shows what the head count is worth. Its weights are a little smaller, but every query head has its own key and value, so a token costs {fmt(wide.perToken)} bytes, {fmt(wide.perToken / low.perToken)} times as much, and {fmt(wide.contexts)} contexts fit. That is one reason later models share KV heads. {chapter("beyond", "Beyond this engine")} comes back to storing weights in fewer bits.
        </p>
      </>
    ),
  },
];

const captions = [
  `calculated from TinyLlama's config.json: ${budget.layers} layers, ${budget.kvHeads} KV heads, head dimension ${budget.headDim}, BF16; block size ${budget.blockTokens} from include/engine/kv/cache_size.h`,
  `calculated: ${fmt(budget.poolBlocks)} blocks, one of them reserved, shared equally and rounded down to whole blocks; measured: the peak block count of ${sweepPeak.batch} short requests, docs/results.md, Batch size past 16`,
  `TinyLlama's pool is measured; the other three are calculated from published shapes (${l3.model}: ${l3.layers} layers, ${l3.kvHeads} KV heads; ${l2.model}: ${l2.layers} layers, ${l2.kvHeads} KV heads; head dimension ${l3.headDim}), weights at exactly 16 or 4 bits, cache in BF16, the same ${gb(budget.rest)} for everything else`,
];

export const memorySteps = steps;
export const memoryCaptions = captions;

const figures = [BytesFigure, BudgetFigure, FitFigure];

export function MemoryStory() {
  return (
    <StickyStory
      steps={steps}
      caption={(i) => captions[i]}
      figure={(i) => {
        const Figure = figures[i] ?? FitFigure;
        return <Figure />;
      }}
    />
  );
}
