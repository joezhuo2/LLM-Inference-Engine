import { blockTokens, blocksFor, engineBest, fmt, loadPredatesFlash, fullBlocks, fullContext, loadPreemptions, maxBatch, peak, poolBlocks, reservedFit, tokenBytes, usableBlocks, vllm, worstCaseBlocks } from "@/content/kernels/serving";
import { chapterNumber } from "../book-order";
import type { ChapterReview } from "./types";

const half = 1024;
const halfFit = Math.floor(usableBlocks / blocksFor(half));

export const review: ChapterReview = {
  lesson: "Count the memory requests hold, not what they might hold, and decide again before every step.",
  why: "It shows how a scheduler turns a fixed pool of cache and a cap on rows into the number of requests served at once, and that when the engine lost to vLLM under load, neither limit was the reason.",
  takeaways: [
    `Every cached token costs ${fmt(tokenBytes)} bytes, a block is ${blockTokens} tokens and the pool is ${fmt(poolBlocks)} blocks. A request holds ceil(length / ${blockTokens}) of them, so ${reservedFit} requests fit if each reserves the full ${fmt(fullContext)} tokens and far more if each holds only what it has written.`,
    `Admission is first come, first served: the front of the queue starts when a row is free and blocks for its whole length are free, and nothing skips ahead. When a running request needs a block and none is free, the newest running request loses its cache, goes to the front of the queue and is recomputed from position 0.`,
    `With max_batch ${maxBatch} the worst case is ${fmt(worstCaseBlocks)} blocks${worstCaseBlocks <= usableBlocks ? ", so the pool cannot fill," : ","} and the load runs recorded ${loadPreemptions} preemptions. The engine's best goodput was ${engineBest.value.toFixed(2)} requests per second against vLLM's ${peak(vllm).value.toFixed(2)}, with the same batching and the same cap${loadPredatesFlash ? ": the gap is kernel speed, mostly prefill." : "."}`,
  ],
  questions: [
    {
      question: "The pool is full. The request at the front of the running list needs a new block. Which request is evicted, and what does the client of that request notice?",
      answer: "The one at the back of the running list. Its client sees a pause, never a changed token.",
      why: "The victim is the most recently admitted request. It loses its blocks but keeps its prompt and the tokens it generated, and when it is admitted again its prefill rebuilds the same cache from position 0, so a greedy answer continues unchanged.",
    },
    {
      question: `Every request stops at ${fmt(half)} tokens. Counting memory only, how many fit in the pool, and why did the measured runs never get near that?`,
      answer: `About ${halfFit}; max_batch stopped them at ${maxBatch}.`,
      why: `${fmt(half)} tokens are ${blocksFor(half)} blocks, and ${fmt(usableBlocks)} / ${blocksFor(half)} is about ${halfFit}. But every running request adds a row to each step and makes it longer, so the scheduler caps the rows at max_batch, and ${maxBatch} requests can hold at most ${maxBatch} × ${fullBlocks} = ${fmt(worstCaseBlocks)} blocks.`,
    },
    {
      question: `Chapter ${chapterNumber("paged-kv")} handed a request its blocks one at a time from the lowest free id. After a preempted request is admitted again, are its tokens in the same physical blocks as before?`,
      answer: "Not necessarily, and it does not matter.",
      why: "Its old blocks went back to the pool and other requests may have taken them. The new prefill takes whatever blocks are free now and writes a new block table, and attention only ever finds tokens through that table.",
      recall: "paged-kv",
    },
  ],
};
