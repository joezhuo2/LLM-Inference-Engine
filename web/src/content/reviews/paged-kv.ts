import { kvReplay } from "@/lib/data";
import type { ChapterReview } from "./types";

const replay = kvReplay();
const B = replay.meta.block_size;
const fullContext = 2048;
const fullBlocks = fullContext / B;
const reservedAtPeak = replay.peak.running * fullBlocks;
const sharePct = (100 * replay.peak.blocks) / reservedAtPeak;
const shareText = sharePct < 10 ? sharePct.toFixed(1) : String(Math.round(sharePct));
const exampleLen = 100;
const exampleBlocks = Math.ceil(exampleLen / B);
const doubledBlock = 2 * B;

export const review: ChapterReview = {
  why: "It shows how small a share of the pool even a busy run needs when blocks are handed out as requests grow, which is what lets one pool serve many requests side by side instead of each reserving its worst case.",
  takeaways: [
    `Nobody knows a request's final length when it arrives, so a contiguous row must reserve the worst case. Handing out ${B}-token blocks as the request grows took ${shareText}% of what worst-case reservation would have taken at the recorded run's busiest step.`,
    `A request keeps a block table, the list of physical blocks it owns in order, and token pos lives at slot table[pos / ${B}] · ${B} + pos mod ${B}. The blocks can be anywhere in the pool, so write_kv only has to follow the table.`,
    `A finished request gives its blocks back and the lowest free id goes out first, so the pool is reused; the whole recorded run stayed within blocks 1 to ${replay.max_block}. If the pool runs out, the newest request is preempted and its tokens are recomputed later, with the same outputs.`,
  ],
  questions: [
    {
      question: `A request is allowed to grow to ${fullContext.toLocaleString("en-US")} tokens but stops at ${exampleLen}. How many blocks does a paged cache hold for it at the end, compared with a cache that reserved its worst case up front?`,
      answer: `${exampleBlocks} blocks, against ${fullBlocks}.`,
      why: `A request of length L owns ceil(L / ${B}) blocks, so ${exampleLen} tokens need ${exampleBlocks}. The reserving cache holds all ${fullBlocks} blocks of the worst case, whatever the request actually generates.`,
    },
    {
      question: `If blocks held ${doubledBlock} tokens instead of ${B}, what is the most slots a request could leave empty?`,
      answer: `${doubledBlock - 1} slots, up from ${B - 1}.`,
      why: `A request only wastes the rest of its last block, because a new block is taken only when the last one is full. So the worst case is one slot short of a full block, and it grows with the block size.`,
    },
  ],
};
