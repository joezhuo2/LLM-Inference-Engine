"use client";

import type { ReactNode } from "react";
import { createPortal } from "react-dom";
import { prefixAbout, prefixAboutCaption } from "@/components/kernels/prefix-caching/about";
import { ChainFigure, PublishFigure, SameKeysFigure, SameOutputFigure } from "@/components/kernels/prefix-caching/figures";
import { PrefixSandbox } from "@/components/kernels/prefix-caching/sandbox";
import { prefixCaptions, prefixSteps } from "@/components/kernels/prefix-caching/story";
import { chapterNumber } from "@/content/book-order";
import { pairs, pcFmt as fmt, prefix, prefixCaching } from "@/content/kernels/prefix-caching";
import type { Beat } from "../book-scene";
import { type ChapterDef, close, flat, type OverlayProps, overview, type WorkbenchPage, withLead } from "../chapter";
import { poolBlock, poolOwn } from "@/components/kernels/prefix-caching/pool";
import type { StoryStep } from "@/components/kernels/sticky-story";
import { createPrefixModel, lanesExhibit, type PcLanes, type PcNumbers, type PcPool, type PcSandbox, type PcSlot } from "./prefix-caching-model";
import { joinParts } from "./segments";

const own = prefix.own;
const last = pairs.length - 1;
const range = pairs.map((q) => fmt(q.shared)).join(", ");
const quiet = prefix.quietArrivals.replace("/s", " a second");

const lead: PcLanes = {
  title: "a later request's prompt, with and without the cache",
  sub: `a shared start plus ${own} tokens of its own; blocks to scale by tokens; times measured`,
  rail: "shared start, tokens",
  range,
  full: pairs[last].shared + own,
  rest: last,
  choices: pairs.map((q) => ({
    lanes: [
      { label: "cache off: prefill every token", text: `${fmt(q.shared + own)} tokens`, parts: [{ value: q.shared + own, tone: "ink" }, { value: 0, tone: "glass" }] },
      { label: "cache on: the start comes from the cache", text: `${fmt(q.shared + own - prefix.perHit(q.shared))} tokens prefilled`, parts: [{ value: prefix.perHit(q.shared), tone: "glass" }, { value: q.shared + own - prefix.perHit(q.shared), tone: "blue" }] },
    ],
    lines: [`${fmt(q.shared)} shared tokens: ${fmt(q.quiet.off.later_ttft_ms)} → ${fmt(q.quiet.on.later_ttft_ms)} ms`, `to a later request's first token, cache off and on (measured, arriving ${quiet})`, `the glass part is ${fmt(prefix.perHit(q.shared) / prefix.block)} cached blocks of ${prefix.block} tokens`],
  })),
};

const S = (text: string): PcSlot[] => [...text].map((c) => ({ ".": "blank", h: "held", s: "shared", c: "cached", t: "taken" })[c] as PcSlot);

const tables: PcPool = {
  title: "two block tables, one pool",
  sub: "twelve blocks of an illustrative pool; height is how many sequences hold a block",
  rail: "what happens",
  range: "A, then B, then both finish",
  events: [
    { slots: S("hhhhh......."), lines: ["A prefills: blocks 1 to 5", "4 full blocks of the system prompt and question, 1 partial", "nothing is published until the step ends"] },
    { slots: S("sssshh......"), lines: ["B starts with the same 4 full blocks", "its table points at blocks 1 to 4: each held by 2 now", "only block 6, its own last tokens, is new"] },
    { slots: S("hhhhch......"), lines: ["A finishes: each block loses a holder", "blocks 1 to 4 are still B's; A's block 5 goes cached", "a block nobody holds is cached if published, blank if not"] },
    { slots: S("cccccc......"), lines: ["B finishes too", "every published block stays cached and counts as free", "a third request with the same start would find blocks 1 to 4"] },
  ],
};

const pool: PcPool = {
  title: "a released block stays cached",
  sub: `an illustrative pool of twelve; the engine's holds ${fmt(prefix.pool)}`,
  rail: "a new request needs 4 blocks",
  range: "blank first, then evict",
  events: [
    { slots: S("cccchh..cchh"), lines: ["the pool: 4 held, 6 cached, 2 blank", "cached blocks keep their keys and values", "and count as free: 8 blocks can be handed out"] },
    { slots: S("cccchhttcchh"), lines: ["the first 2 blocks come from the blank ones", "no cached block is touched while a blank one is left", "blocks 7 and 8, in pink"] },
    { slots: S("cc..hhttcchh"), lines: ["no blank block left: evict cached ones", "released longest ago first, the end of a chain before its start", "blocks 4 and 3 lose their keys"] },
    { slots: S("cctthhttcchh"), lines: ["handed out like blank blocks", "blocks 1 and 2, the chain's start, are still cached", "a later prompt can still match that start"] },
  ],
};

const ttft: PcLanes = {
  title: "a later request's time to the first token",
  sub: `requests arriving ${quiet}, median of ${pairs[last].quiet.on.runs} runs (measured)`,
  rail: "shared start, tokens",
  range,
  full: Math.max(...pairs.map((q) => q.quiet.off.later_ttft_ms)),
  rest: last,
  choices: pairs.map((q) => ({
    lanes: [
      { label: "cache off", text: `${fmt(q.quiet.off.later_ttft_ms, 1)} ms`, parts: [{ value: q.quiet.off.later_ttft_ms, tone: "ink" }] },
      { label: "cache on", text: `${fmt(q.quiet.on.later_ttft_ms, 1)} ms`, parts: [{ value: q.quiet.on.later_ttft_ms, tone: "blue" }] },
    ],
    lines: [`${fmt(q.shared)} shared + ${own} own: ${fmt(q.quiet.off.later_ttft_ms, 1)} → ${fmt(q.quiet.on.later_ttft_ms, 1)} ms`, `with the cache, a prefill step of ${fmt(q.quiet.on.prefill_ms, 1)} ms: the ${own} own tokens`, `hit rate ${fmt(100 * q.quiet.on.hit_rate, 1)}% of all prompt tokens`],
  })),
};

const burst: PcLanes = {
  title: `all ${prefix.requests} requests at once`,
  sub: `output tokens per second, max batch ${prefix.maxBatch}, median of ${pairs[last].burst.on.runs} runs (measured)`,
  rail: "shared start, tokens",
  range,
  full: Math.max(...pairs.map((q) => q.burst.on.output_tps)),
  rest: last,
  choices: pairs.map((q) => ({
    lanes: [
      { label: "cache off", text: `${fmt(q.burst.off.output_tps)} tokens/s`, parts: [{ value: q.burst.off.output_tps, tone: "ink" }] },
      { label: "cache on", text: `${fmt(q.burst.on.output_tps)} tokens/s`, parts: [{ value: q.burst.on.output_tps, tone: "blue" }] },
    ],
    lines: [`${fmt(q.shared)} shared: ${fmt(prefix.speedup(q), 1)}× the tokens per second`, `the first step holds ${prefix.firstStep(q.shared)} prompts, which find nothing`, `every later one counts only its ${own} own rows; hit rate ${fmt(100 * q.burst.on.hit_rate, 1)}%`],
  })),
};

const blocks = (n: number) => `${n} ${n === 1 ? "block" : "blocks"}`;
const stepNames = ["arrive", "finish", "another start", "the start again"];

const sandbox: PcSandbox = {
  title: "run a pool yourself",
  sub: "blue: held, taller when shared · ink: cached · glass: blank · pink: evicted, handed out again",
  shared: { label: "shared start", values: [0, 1, 2, 3, 4], start: 3, text: (v) => `${v * poolBlock} tokens` },
  requests: { label: "requests", values: [1, 2, 3, 4, 5], start: 2, text: (v) => String(v) },
  pool: { label: "pool", values: [6, 8, 10, 12, 16], start: 2, text: (v) => blocks(v) },
  steps: stepNames,
  restStep: 2,
  lines: (ph, step, k) => {
    const waiting = ph.rows.filter((r) => r.state === "waiting").length;
    const head = [
      waiting ? `${waiting} of ${k.requests} ${k.requests === 1 ? "request waits" : "requests wait"}: no room in ${blocks(k.pool)}` : `${k.requests} ${k.requests === 1 ? "request" : "requests"} with the same ${k.shared * poolBlock}-token start, one step each`,
      ph.late ? `the waiting ones run as others finish; ${blocks(ph.cached)} stay cached` : `they all finish: ${blocks(ph.cached)} stay cached, and count as free`,
      ph.evicted.length ? `another start: blank blocks first, then ${blocks(ph.evicted.length)} evicted` : "another start: it fits in blank blocks, nothing evicted",
      ph.rows[ph.rows.length - 1]?.reused ? `the same start again: it finds ${blocks(ph.rows[ph.rows.length - 1].reused / poolBlock)} still cached` : "the same start again: nothing of it is left to find",
    ][step];
    return [head, `tokens prefilled ${ph.prefilled} · tokens reused ${ph.reused}`, ph.evicted.length ? `evicted: ${ph.evicted.join(", ")}, the end of a chain first` : `${blocks(ph.held)} held · ${blocks(ph.cached)} cached · ${blocks(ph.blank)} blank`];
  },
};

const numbers: PcNumbers = { lead, tables, pool, ttft, burst, sandbox };

const poolStep: StoryStep = {
  id: "pool-sandbox",
  group: "try it yourself",
  title: "run a pool yourself",
  body: (
    <>
      <p>
        The pool on the right follows the block manager&apos;s rules, ported to TypeScript from match_blocks, allocate_tokens, publish and free in src/kv/block_manager.cpp. Requests arrive one per step with the same start and {poolOwn} tokens of their own, so each one&apos;s last block is its own.
      </p>
      <p>
        The first three pink balls set the shared start, the number of requests and the size of the pool. The fourth steps through what happens: the requests arrive and share the start&apos;s blocks, they finish and the shared blocks stay cached, another start needs room and evicts, and the shared start comes back and finds what is left.
      </p>
      <p>A block held by several requests stands taller and is marked ×2, ×3. Each table lists a request&apos;s blocks, shared ones in blue. Make the pool small and the requests many: sharing needs the start&apos;s blocks once, plus one block per request.</p>
    </>
  ),
};

const rail = close("The rail: drag to another step", [0.5, 0.02, 0.2], 0.9, 0.34, 0, 46);

const beats: Beat[][] = [
  [flat("Two prompts, the same start")],
  [flat("A block's key, every token before its end")],
  [overview("Two block tables, one pool"), close("Close up: blocks held by two sequences stand taller", [0.4, 0.04, -0.32], 0.6, 0.3, -12, 38), close("From the side: the blocks a request finishes with", [0.5, 0.04, -0.3], 0.8, 0.3, 24, 26), close("The rail: drag to the next event", [0.5, 0.02, 0.2], 0.9, 0.34, 0, 46)],
  [flat("Published when the step ends")],
  [overview("A released block stays cached"), close("Close up: blank blocks go first", [0.5, 0.04, -0.32], 0.7, 0.3, -12, 38), close("From the side: what eviction leaves cached", [0.3, 0.04, -0.3], 0.6, 0.3, 24, 26), close("The rail: drag to the next event", [0.5, 0.02, 0.2], 0.9, 0.34, 0, 46)],
  [overview("A later request's time to the first token"), close("Close up: cache off against cache on", [0.4, 0.04, -0.35], 0.8, 0.3, -12, 38), close("From the side: the bars by prefix length", [0.5, 0.04, -0.3], 0.8, 0.3, 24, 26), rail],
  [overview(`All ${prefix.requests} requests at once`), close("Close up: tokens per second, off and on", [0.4, 0.04, -0.35], 0.8, 0.3, -12, 38), close("From the side: the gain grows with the prefix", [0.5, 0.04, -0.3], 0.8, 0.3, 24, 26), rail],
  [flat("Bit-identical keys, nearly identical answers")],
  [overview("A pool you can change"), close("Close up: the pool, block by block", [0.5, 0.04, -0.42], 0.9, 0.24, -10, 40), close("From the side: the block tables", [0.5, 0.04, -0.24], 0.9, 0.3, 22, 30), close("The rails: drag to change the pool", [0.5, 0.02, 0.22], 0.9, 0.24, 0, 46)],
];

const instructions = [
  "Follow the two prompts token by token: blue keys and values are the same in both, pink ones differ.",
  "Find where the second prompt differs: every key from that block on changes.",
  "Drag the pink ball through the events: tall blocks are held by two sequences, ink ones are cached.",
  "Compare A and B with C: only a request admitted after the publish finds the start.",
  "Drag the pink ball through the events: blank blocks go first, then the cached ones released longest ago.",
  "Drag the pink ball to another prefix length: the blue bar barely moves.",
  "Drag the pink ball to another prefix length: the longer the shared start, the larger the gain.",
  "Count the pink outputs: each one split at a near tie, not at a different key or value.",
  "Drag the pink balls: the first three change the start, the requests and the pool; the fourth steps through arrive, finish, another start and the start again.",
];

const workbench: WorkbenchPage[] = [
  {
    title: "share a prefix yourself",
    overlay: "sandbox",
    caption: "Try it yourself: what the cache can reuse",
    paragraphs: [
      `Set the length of the shared start, the tokens each request adds and how the ${prefix.requests} requests arrive. The arithmetic is the block manager's: only whole ${prefix.block}-token blocks are shared, the block of the last prompt token is computed, and requests admitted in the same step find nothing.`,
      `Move the start off a multiple of ${prefix.block}, to ${fmt(pairs[last].shared - 8)} say. The last few shared tokens now sit in a block with the request's own, and that block is computed in every request.`,
      `Switch to all at once and watch the first step: it holds as many whole prompts as fit in ${fmt(prefix.budget)} tokens, and none of them can share. At the measured settings the sandbox gives exactly the cached tokens the runs counted.`,
    ],
  },
];

const figures: Record<string, () => ReactNode> = {
  keys: () => <SameKeysFigure />,
  chain: () => <ChainFigure />,
  publish: () => <PublishFigure />,
  same: () => <SameOutputFigure />,
};

function PrefixOverlay({ kind, target, narrow }: OverlayProps) {
  if (!target || !kind) return null;
  if (kind === "sandbox") return createPortal(<PrefixSandbox />, target);
  const figure = figures[kind];
  if (!figure) return null;
  return createPortal(<div className={narrow ? "aspect-[560/400] w-full" : "h-[34rem]"}>{figure()}</div>, target);
}

const chapter: ChapterDef = {
  slug: "prefix-caching",
  number: chapterNumber("prefix-caching"),
  title: "prefix caching",
  formula: prefixCaching.formula,
  steps: [...prefixSteps, poolStep],
  captions: [...prefixCaptions, `a TypeScript port of the BlockManager rules in src/kv/block_manager.cpp; blocks of ${poolBlock} tokens; the pool is illustrative, the engine's holds ${fmt(prefix.pool)} blocks`],
  exhibits: [null, null, 0, null, 1, 2, 3, null, 4],
  overlays: ["keys", "chain", null, "publish", null, null, null, "same", null],
  instructions,
  beats,
  workbench,
  buildModel: (kit, fonts) => createPrefixModel(kit, fonts, numbers),
  Overlay: PrefixOverlay,
};

export const prefixCachingChapter = withLead(chapter, {
  step: prefixAbout,
  caption: prefixAboutCaption,
  instruction: "Drag the pink ball to another shared start: the glass part of the lower bar comes from the cache, only the blue part is prefilled.",
  beats: [
    overview("The same prompt, with and without the cache"),
    close("Close up: the glass blocks come from the cache", [0.4, 0.04, -0.3], 0.7, 0.34, -12, 38),
    close("From the side: only the blue part is prefilled", [0.75, 0.04, -0.3], 0.5, 0.3, 24, 26),
    close("The rail: drag to another shared start", [0.5, 0.02, 0.2], 0.9, 0.34, 0, 46),
  ],
  buildModel: (kit, fonts) => joinParts([lanesExhibit(kit, fonts, lead)]),
});
