"use client";

import Link from "next/link";
import { chapterNumber } from "@/content/book-order";
import { pairs, pcFmt as fmt, prefix } from "@/content/kernels/prefix-caching";
import { expectAnswer, type Prediction, Predict } from "../predict";
import { StickyStory, type StoryStep } from "../sticky-story";
import { BurstFigure, ChainFigure, PoolFigure, PublishFigure, SameKeysFigure, SameOutputFigure, TablesFigure, TtftFigure } from "./figures";

const chapter = (slug: string, text: string) => (
  <Link href={`/kernels/${slug}`} className="link-underline text-accent">
    {text}
  </Link>
);
const p = prefix.largest;
const small = prefix.smallest;
const s = prefix.sameOut;
const ms = (v: number) => `${fmt(v, 1)} ms`;
const quiet = prefix.quietArrivals.replace("/s", " a second");

const prediction: Prediction = expectAnswer({
  question: `With the cache on, the shared start grows from ${fmt(small.shared)} to ${fmt(p.shared)} tokens, four times longer. What happens to a later request's time to the first token?`,
  choices: ["about four times longer", "about the same", "it halves"],
  answer: p.quiet.on.later_ttft_ms < 2 * small.quiet.on.later_ttft_ms ? "about the same" : "about four times longer",
  why: `${ms(small.quiet.on.later_ttft_ms)} against ${ms(p.quiet.on.later_ttft_ms)} (measured). The cached start costs no prefill; what is left is the request's own ${prefix.own} tokens, which only read more keys as the start grows.`,
}, "about the same");

const steps: StoryStep[] = [
  {
    id: "same-keys",
    title: "same start, same keys and values",
    body: (
      <>
        <p>
          In {chapter("one-token", "one token, end to end")}, every layer turns each token into a query, a key and a value, and attention lets a token look only at the tokens before it. So a token&apos;s key and value are computed from that token and everything before it, and from nothing after it.
        </p>
        <p>
          Take two prompts that begin with the same words and then differ. Up to the first different token, every key and value is computed from the same inputs, so they are the same numbers. From the first different token on, they differ. <span className="text-accent">The shared start&apos;s keys and values do not depend on what follows it.</span>
        </p>
        <p>That is what makes them reusable: computed once for the first request, they are exactly what the second request would compute for the same tokens.</p>
      </>
    ),
  },
  {
    id: "chain",
    title: "every block is keyed by its prefix",
    body: (
      <>
        <p>
          The engine already keeps keys and values in blocks of {prefix.block} tokens, the {chapter("paged-kv", "paged KV cache")}. The cache shares whole blocks, so it needs a way to name a block&apos;s contents: a key to look it up by.
        </p>
        <p>
          A block&apos;s own {prefix.block} tokens are not enough, because its keys and values also depend on every token before it. So <span className="text-accent">the key of a block is every token from the start of the sequence to the end of that block</span>. Two blocks with the same key really do hold the same numbers.
        </p>
        <p>
          The block manager hashes that run of tokens to find the block fast in a map, and keeps the tokens themselves to compare on a hit, since two different runs can share a hash. A match walks the new prompt block by block and stops at the first block it does not find, so it can only ever find a start.
        </p>
      </>
    ),
  },
  {
    id: "tables",
    title: "two block tables, one physical block",
    body: (
      <>
        <p>
          A request finds its keys and values through its block table, the list of pool blocks that hold its tokens in order. Nothing stops two tables from listing the same block. So when a new prompt starts with cached blocks, admission puts <span className="text-accent">those same physical blocks</span> into the new request&apos;s table and takes fresh blocks only for the rest.
        </p>
        <p>
          Each block keeps a count of the sequences that hold it. A shared block is never written into: a block is shared only once every one of its tokens has its keys and values, and a sequence only ever writes past them. The request&apos;s prefill then starts after its cached tokens, in plain, flash and chunked prefill alike, and its rows read the shared keys through its own table.
        </p>
        <p>The last prompt token is always computed, even when the whole prompt is cached: its row gives the logits that pick the first new token. So a block that ends at the last token is never matched.</p>
      </>
    ),
  },
  {
    id: "publish",
    title: "published when its step ends",
    body: (
      <>
        <p>
          A block can be shared only once its keys and values are written. So the scheduler publishes blocks when a step finishes: <span className="text-accent">every full block whose keys and values that step wrote</span> goes into the map, prompt blocks after a prefill and a block of generated tokens when a decode step fills it.
        </p>
        <p>
          Generated tokens count too. A second chat turn sends the first turn and its answer back, followed by a new question, and it finds both the question and the answer in the cache.
        </p>
        <p>
          One case gains nothing: requests {chapter("serving", "admitted")} in the same step. Their keys and values are being written during that step, so neither can use the other&apos;s, and both compute the shared start. A request admitted in any later step finds it.
        </p>
      </>
    ),
  },
  {
    id: "cached",
    title: "a released block stays cached",
    body: (
      <>
        <p>
          When a request finishes, its blocks lose a holder. A block nobody holds any more used to go back to the pool blank. Now a published block <span className="text-accent">keeps its keys and values and stays in the map</span>, and it counts as free: the pool can still hand it out.
        </p>
        <p>
          New blocks come from the blank ones first. Only when no blank block is left does the block manager evict a cached block, the one released longest ago. A request releases its blocks from the end of its table backwards, so the end of a chain goes before its start, and what stays cached is a start a later prompt can still match.
        </p>
        <p>An evicted block leaves the map and is handed out like a blank one. In the measured runs the pool of {fmt(prefix.pool)} blocks never ran out of blank blocks, so nothing was evicted there; the tests cover eviction.</p>
      </>
    ),
  },
  {
    id: "ttft",
    group: "measured",
    title: "a hit leaves only its own prefill",
    body: (
      <>
        <p>
          The measurement: {p.quiet.on.requests} requests, each a shared start of {pairs.map((q) => fmt(q.shared)).join(", ")} tokens plus {prefix.own} random tokens of its own, arriving {quiet}, so a request usually finds the engine idle or decoding. With the cache off a later request waits {pairs.map((q) => ms(q.quiet.off.later_ttft_ms)).join(", ")} for its first token. With the cache on it waits <span className="text-accent">{pairs.map((q) => ms(q.quiet.on.later_ttft_ms)).join(", ")}</span> (measured).
        </p>
        <p>
          What is left is the prefill of its own {prefix.own} tokens, a step of {fmt(prefix.prefillLow)} to {fmt(prefix.prefillHigh)} ms that grows only a little with the start, because those rows attend to every key before them. So a hit saves the prefill of the shared tokens.
        </p>
        <p>
          The first request is the cold case: it finds nothing and prefills everything. One number here is not understood. At {fmt(p.shared)} shared tokens the first request measured {ms(p.quiet.on.first_ttft_ms)} with the cache on and {ms(p.quiet.off.first_ttft_ms)} with it off, although it does the same work in both; the cause was not found.
        </p>
        <Predict {...prediction} />
      </>
    ),
  },
  {
    id: "burst",
    group: "measured",
    title: "all at once, the batch fills sooner",
    body: (
      <>
        <p>
          Now all {p.quiet.on.requests} requests arrive at once. A prefill step takes prompts until their tokens reach a budget of {fmt(prefix.budget)}, so the first step admits {pairs.map((q) => `${prefix.firstStep(q.shared)} at ${fmt(q.shared)}`).join(", ")} shared tokens, and none of them can share. Every later admission finds the start and <span className="text-accent">counts only its {prefix.own} uncached rows</span> against the budget, so more requests join each prefill step.
        </p>
        <p>
          The batch fills sooner and the engine writes {pairs.map((q) => `${fmt(prefix.speedup(q), 1)} times`).join(", ")} the tokens per second, at {pairs.map((q) => fmt(q.shared)).join(", ")} shared tokens ({pairs.map((q) => `${fmt(q.burst.on.output_tps)} against ${fmt(q.burst.off.output_tps)}`).join(", ")}, measured).
        </p>
        <p>
          The hit rates are block arithmetic. One at a time, {p.quiet.on.requests - 1} of the {p.quiet.on.requests} requests find all {fmt(p.shared)} shared tokens: {p.quiet.on.requests - 1} × {fmt(p.shared)} of {p.quiet.on.requests} × {fmt(p.shared + prefix.own)} tokens is {fmt(100 * p.quiet.on.hit_rate, 1)}%, the measured rate. All at once at {fmt(small.shared)}, the {prefix.firstStep(small.shared)} requests of the first step find nothing, and the rate falls to {fmt(100 * small.burst.on.hit_rate, 1)}%.
        </p>
      </>
    ),
  },
  {
    id: "same-output",
    group: "how it got built",
    title: "same keys, nearly the same answers",
    body: (
      <>
        <p>
          The GPU tests check what a hit reads: the keys and values it takes from the cache are <span className="text-accent">bit-identical, in all {s.layers} layers</span>, to the ones the request writes when it prefills its whole prompt itself.
        </p>
        <p>
          The answers are compared too, greedy, with the cache on and off, over shared system prompts, repeated prompts, concurrent requests, eviction in a small pool, speculation, chunked prefill and a second chat turn. {s.same} of {s.outputs} outputs are identical. The other {s.split} split at a near tie: the token picked instead was within {s.ulps} BF16 ulps of the best logit.
        </p>
        <p>
          The cache is not what changes them. A hit computes the request&apos;s own rows as a separate, smaller piece, and the matrix products round a little differently depending on how many rows a step holds. {chapter("chunked-prefill", "Chunked prefill")} cuts a prompt the same way, and with the cache off it splits one of the same requests at the same token.
        </p>
      </>
    ),
  },
];

const source = `results.md, Prefix caching, measured ${prefix.date} on the ${prefix.gpu}`;
const captions = [
  "illustrative words; causal attention, as in every layer of TinyLlama",
  `the block key of match_blocks() in src/kv/block_manager.cpp; blocks of ${prefix.block} tokens; token numbers illustrative`,
  "allocate_tokens() in src/kv/block_manager.cpp and admit() in src/sched/scheduler.cpp; block numbers illustrative",
  "the publish rule of Scheduler::update in src/sched/scheduler.cpp; steps illustrative",
  `free_block() and allocate_block() in src/kv/block_manager.cpp; the engine's pool holds ${fmt(prefix.pool)} blocks; the pool here is illustrative`,
  `${source}: median TTFT of the ${p.quiet.on.requests - 1} later requests, median of ${p.quiet.on.runs} runs; the prefill step from the step traces`,
  `${source}: all ${p.quiet.on.requests} requests submitted at once, max batch ${prefix.maxBatch}; median of ${p.burst.on.runs} runs`,
  `tests/gpu/test_serve_prefix_cache.cpp, as results.md reports it; chapter ${chapterNumber("chunked-prefill")} has the same near ties`,
];

export const prefixSteps = steps;
export const prefixCaptions = captions;

const figures = [SameKeysFigure, ChainFigure, TablesFigure, PublishFigure, PoolFigure, TtftFigure, BurstFigure, SameOutputFigure];

export function PrefixStory() {
  return (
    <StickyStory
      steps={steps}
      caption={(i) => captions[i]}
      figure={(i) => {
        const Figure = figures[i] ?? SameOutputFigure;
        return <Figure />;
      }}
    />
  );
}
