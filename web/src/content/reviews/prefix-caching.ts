import { pcFmt as fmt, prefix } from "@/content/kernels/prefix-caching";
import type { ChapterReview } from "./types";

const p = prefix.largest;
const odd = p.shared - 24;
const third = prefix.smallest;

export const review: ChapterReview = {
  lesson: "A token's keys and values depend only on the tokens up to it, so a request that starts like an earlier one can point its block table at the earlier blocks and prefill only the rest.",
  why: "Chat turns, system prompts and shared documents repeat the same start in many requests, and the prefill of that start is work done again for nothing.",
  takeaways: [
    `A block is shared only whole: its key is every token from the start of the sequence to the end of the block, ${prefix.block} tokens at a time, and the block of the last prompt token is always computed.`,
    `A finished step publishes the full blocks it wrote, generated tokens included. A released published block stays cached and counts as free; it is evicted only when no blank block is left, least recently released first.`,
    `At a ${fmt(p.shared)}-token shared prefix the later requests' time to the first token fell from ${fmt(p.quiet.off.later_ttft_ms)} to ${fmt(p.quiet.on.later_ttft_ms)} ms, and with all ${p.quiet.on.requests} requests at once the engine wrote ${fmt(prefix.speedup(p), 1)} times the tokens per second (measured).`,
  ],
  questions: [
    {
      question: `${p.quiet.on.requests} requests each send the same ${fmt(p.shared)}-token prefix and ${prefix.own} tokens of their own, one at a time. What share of all their prompt tokens comes from the cache?`,
      answer: `${fmt(100 * p.quiet.on.hit_rate, 1)}%: ${p.quiet.on.requests - 1} × ${fmt(p.shared)} cached tokens out of ${p.quiet.on.requests} × ${fmt(p.shared + prefix.own)}.`,
      why: "The first request finds nothing. Every later one finds the whole prefix, since its length is a whole number of blocks, and computes its own tokens.",
    },
    {
      recall: "paged-kv",
      question: `The shared prefix is ${fmt(odd)} tokens instead of ${fmt(p.shared)}. How many of them can a later request reuse?`,
      answer: `${fmt(prefix.perHit(odd))}: the whole blocks inside the prefix. The block that holds the prefix's last ${odd % prefix.block} tokens also holds the request's own first tokens, so its key differs.`,
      why: `The cache shares blocks, never single tokens, and a block's key covers all ${prefix.block} of its tokens.`,
    },
    {
      question: `Why did the later requests' time to the first token barely change between a ${fmt(third.shared)}-token and a ${fmt(p.shared)}-token prefix with the cache on (${fmt(third.quiet.on.later_ttft_ms, 1)} against ${fmt(p.quiet.on.later_ttft_ms, 1)} ms)?`,
      answer: `What is left is the prefill of the request's own ${prefix.own} tokens, which is about the same at any prefix length.`,
      why: `It grows only a little with the prefix, because those ${prefix.own} rows attend to every cached key before them.`,
    },
    {
      question: `Two requests with the same system prompt are admitted in the same step. Does the second one reuse the first one's blocks?`,
      answer: "No. Nothing is published until the step that wrote it finishes, so both compute the prompt.",
      why: "Their keys and values are written during that step. A request admitted in a later step finds them.",
    },
  ],
};
