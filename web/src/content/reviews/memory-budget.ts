import { budget, fitOf, gb, lastRow, memFmt } from "@/content/kernels/memory-budget";
import { published } from "@/content/published";
import type { ChapterReview } from "./types";

const fmt = memFmt;
const low = fitOf(published.llama3.model, 4);
const wide = fitOf(published.llama2.model, 4);

export const review: ChapterReview = {
  lesson: "Weights are a fixed cost and the KV cache is a shared one: batch size and context length draw on the same pool.",
  why: "It says what a GPU can hold before asking how fast it runs, and why a larger model serves fewer requests at once.",
  takeaways: [
    `The weights take ${gb(budget.weights)} of ${gb(budget.vram)} whatever the load (calculated); the engine gives ${budget.fractionPct}% of what is free after them to one KV pool, ${gb(budget.pool)} (measured).`,
    `A cached token is ${fmt(budget.tokenBytes)} bytes: ${budget.layers} layers × K and V × ${budget.kvHeads} KV heads × ${budget.headDim} numbers × 2 bytes (calculated).`,
    `Batch × context cannot pass ${fmt(budget.poolTokens)} tokens. ${budget.reservedFit} requests fit at the full ${fmt(budget.fullContext)}-token context; ${lastRow.batch} can each reach ${fmt(lastRow.context)} (calculated).`,
  ],
  questions: [
    {
      question: `A request has a ${fmt(budget.fullContext / 2)}-token context. How much of the pool does it hold?`,
      answer: `${fmt((budget.fullContext / 2) * budget.tokenBytes / 1e6, 1)} MB: ${fmt(budget.fullContext / 2)} × ${fmt(budget.tokenBytes)} bytes, ${fmt(budget.fullContext / 2 / budget.blockTokens)} blocks.`,
      why: "The cache of a request is its length times a constant. What the tokens say does not change it.",
    },
    {
      recall: "paged-kv",
      question: "The pool is allocated once at start. What does a request do when no block is free?",
      answer: "It waits, or a running request gives its blocks back. No request fails on memory in the middle of an answer.",
      why: "The paged KV cache hands out blocks from a free list; when the list is empty the scheduler decides who waits. Nothing asks the GPU for more memory.",
    },
    {
      question: `${published.llama3.model} at 4 bits fits ${fmt(low.contexts)} full contexts on this GPU and ${published.llama2.model} at 4 bits only ${fmt(wide.contexts)}, with smaller weights. Why?`,
      answer: `Its cache costs ${fmt(wide.perToken / low.perToken)} times as much per token: ${published.llama2.kvHeads} KV heads against ${published.llama3.kvHeads}.`,
      why: "Bytes per token scale with the number of KV heads. Sharing keys and values between query heads shrinks the cache without touching the weights much.",
    },
  ],
};
