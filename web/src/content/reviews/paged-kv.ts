import { kvReplay } from "@/lib/data";
import { about } from "@/lib/round";
import { published } from "../published";
import { bigSide } from "../scale";
import type { ChapterReview } from "./types";

const replay = kvReplay();
const B = replay.meta.block_size;
const fullContext = 2048;
const fullBlocks = fullContext / B;
const reservedAtPeak = replay.peak.running * fullBlocks;
const sharePct = (100 * replay.peak.blocks) / reservedAtPeak;
const exampleLen = 100;
const exampleBlocks = Math.ceil(exampleLen / B);
const doubledBlock = 2 * B;
const l3 = published.llama3;
const fmt = (v: number) => v.toLocaleString("en-US");

export const review: ChapterReview = {
  lesson: "Hand out memory in small blocks as tokens arrive, never for a worst case up front.",
  why: "It shows how small a share of the pool even a busy run needs when blocks are handed out as requests grow, which is what lets one pool serve many requests side by side instead of each reserving its worst case.",
  takeaways: [
    `Nobody knows a request's final length when it arrives, so a contiguous row must reserve the worst case. Handing out ${B}-token blocks as the request grows took ${about(sharePct)}% of what worst-case reservation would have taken at the recorded run's busiest step.`,
    `A request keeps a block table, the list of physical blocks it owns in order, and token pos lives at slot table[pos / ${B}] · ${B} + pos mod ${B}. The blocks can be anywhere in the pool, so write_kv only has to follow the table.`,
    `A finished request gives its blocks back and the lowest free id goes out first, so the pool is reused; replayed by these rules, the whole recorded run stays within blocks 1 to ${replay.max_block}. If the pool runs out, the newest request is preempted and its tokens are recomputed later, with the same outputs.`,
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
    {
      question: `RoPE turns each key right before write_kv copies it into a ${B}-token block. Why can the cache keep that rotated key for the rest of the request instead of turning it again as new tokens arrive?`,
      answer: "A key is turned by its own position only, and that never changes.",
      why: "RoPE gives each key the angle of its own position, and the score with any later query depends only on their gap. So the stored key stays correct in whatever block it lands, and each token is written once and only read after.",
      recall: "rope",
    },
    {
      question: `${bigSide.model} has ${l3.layers} layers and ${l3.kvHeads} KV heads of ${l3.headDim} dimensions, cached in BF16, and a context of ${fmt(bigSide.context)} tokens. How many bytes does one cached token take, and how many ${B}-token blocks does a cache that reserves the full context hold for a request that stops at ${exampleLen} tokens, against a paged cache?`,
      answer: `${fmt(bigSide.kvTokenBytes)} bytes, and ${bigSide.contextBlocks} blocks against ${exampleBlocks}.`,
      why: `K and V, ${l3.layers} layers, ${l3.kvHeads} heads, ${l3.headDim} numbers of 2 bytes: 2 × ${l3.layers} × ${l3.kvHeads} × ${l3.headDim} × 2 = ${fmt(bigSide.kvTokenBytes)}. ${fmt(bigSide.context)} ÷ ${B} = ${bigSide.contextBlocks} blocks, ${fmt(bigSide.contextBytes)} bytes, held whatever the request does; ceil(${exampleLen} ÷ ${B}) = ${exampleBlocks} blocks, ${fmt(exampleBlocks * bigSide.blockBytes)} bytes. Calculated from ${l3.vendor}'s published specs; nothing was run with that model.`,
    },
  ],
};
