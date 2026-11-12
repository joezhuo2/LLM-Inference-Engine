"use client";

import Link from "next/link";
import { piece, spec, specFmt } from "@/content/kernels/speculative-decoding";
import { StickyStory, type StoryStep } from "../sticky-story";
import { CheckFigure, GuessFigure, PayoffFigure, RowsFigure, TruncateFigure } from "./figures";

const fmt = specFmt;
const chapter = (slug: string, text: string) => (
  <Link href={`/kernels/${slug}`} className="link-underline text-accent">
    {text}
  </Link>
);
const best = spec.bestStep;
const miss = spec.missStep;
const chat = spec.chatAt(spec.best.k) ?? spec.chatWorst;
const quote = (tokens: string[]) => tokens.map(piece).join("").trim();

const steps: StoryStep[] = [
  {
    id: "rows",
    title: "a step has rows to spare",
    body: (
      <>
        <p>
          A decode step for one request reads every weight to produce one token. {chapter("foundations", "The first chapters")} showed that the reading is the cost: the arithmetic for one row is tiny beside it. So a second row in the same step is nearly free, because the weights are read once for all the rows.
        </p>
        <p>
          The numbers say how nearly. One request alone takes {fmt(spec.step1, 2)} ms per token. Sixteen requests in one step take {fmt(spec.step16, 2)} ms: <span className="text-accent">{fmt(spec.maxBatch)} rows for {fmt(spec.rowCost, 2)} times the price of one</span> (measured). When fewer than {fmt(spec.maxBatch)} requests are running, the other rows go unused.
        </p>
        <p>Speculative decoding fills those rows with more tokens of the same request. The trouble is that the tokens after the next one are not known yet. So it guesses them.</p>
      </>
    ),
  },
  {
    id: "guess",
    title: "the guess is copied from the context",
    body: (
      <>
        <p>
          Many answers repeat their prompt. A summary quotes the text, an edited function is mostly the old function, a list is copied with one change. When the answer is in the middle of such a copy, the next tokens are already on the page, a little earlier.
        </p>
        <p>
          So the engine looks back. It takes the last {spec.maxNgram} tokens of the context and searches for the latest earlier place where the same {spec.maxNgram} appeared. The tokens that followed them there are the guess, the draft. If there is no match for {spec.maxNgram} tokens it tries {spec.maxNgram - 1}, then 1, and with no match at all there is no draft and the step is a plain one. This is called prompt lookup.
        </p>
        <p>
          In the recorded answer, step {best.index} found its last {best.ngram === 1 ? "token" : `${best.ngram} tokens`} {fmt(best.at - best.ngram - best.match)} tokens back and copied the next {best.draft.length} as its guess, <span className="text-accent">&ldquo;{quote(best.draft)}&rdquo;</span> in the prompt, and the model agreed with {best.kept === best.draft.length ? "all of them" : `${best.kept} of them`}. The search is a scan of the context on the CPU. It needs no second model and no extra pass on the GPU.
        </p>
      </>
    ),
  },
  {
    id: "check",
    title: "one step checks every guess",
    body: (
      <>
        <p>
          The step then runs the real input and the draft together, as rows of the same sequence. Row 0 is the last real token. Row 1 is the first guessed token, row 2 the second, and so on. Each row attends to everything before it, the earlier rows of this step included, the way the rows of a prompt do in {chapter("flash-prefill", "prefill")}.
        </p>
        <p>
          Every row samples a token. Row 0&apos;s token is what plain decoding would have produced, so it is always kept. If it equals the first guess, then row 1 was computed from the right input, and its token is kept too. The chain stops at the first row whose guess was wrong: that row&apos;s own token is still right, and everything after it was computed from a wrong token and is dropped.
        </p>
        <p>
          So every kept token is <span className="text-accent">the token plain decoding would have sampled</span>. This holds with a temperature too, because the sampler&apos;s random number depends on the seed and on the token&apos;s place in the answer, not on the step it ran in. In step {miss.index} the model agreed with {miss.kept} of {miss.draft.length} guesses, and the step wrote {miss.kept + 1} tokens.
        </p>
        <p>
          One caveat is the one batching already has. A step with more rows rounds its BF16 arithmetic a little differently, so where the two best tokens are one rounding step apart, the choice can go the other way. {spec.changed === 0 ? `It did not happen in the ${fmt(spec.compared)} measured prompts.` : `It happened in ${fmt(spec.changed)} of the ${fmt(spec.compared)} measured prompts, each time at such a near tie.`}
        </p>
      </>
    ),
  },
  {
    id: "truncate",
    title: "a wrong guess gives its cache slots back",
    body: (
      <>
        <p>
          Every row writes its key and value into the {chapter("paged-kv", "paged KV cache")} before attention reads them. That is what lets a guessed row see the guessed rows before it. It also means a rejected row has left a key and a value behind, computed from a token that is not in the answer.
        </p>
        <p>
          The scheduler reserves one slot per row before the step and gives the unused ones back after it. The sequence&apos;s length in the block table is cut to the last kept row, and a block that is no longer needed returns to the free list. The next step writes its own key and value over the stale slot before anything reads it.
        </p>
        <p>
          Two rules keep guesses from hurting other requests. A draft only takes rows that no request needs: with {fmt(spec.maxBatch)} requests running there are none, and nothing changes. And a draft never takes a block that would make another request wait: if the pool is short, the draft is shortened instead.
        </p>
      </>
    ),
  },
  {
    id: "payoff",
    title: "it pays when the answer repeats the prompt",
    body: (
      <>
        <p>
          On four prompts whose answers repeat them (repeat a paragraph, fix its spelling, add a docstring to a function, rename a variable in it), with up to {spec.best.k} guesses per step, the model agreed with {fmt(spec.best.acceptedPct)}% of the guessed tokens. A step wrote {fmt(spec.best.tokensPerStep, 2)} tokens on average, and the time per token fell from {fmt(spec.plain.ms, 2)} to <span className="text-accent">{fmt(spec.best.ms, 2)} ms per token</span> (measured, one request at a time). That is {fmt(spec.best.speedup, 2)} times the tokens per second.
        </p>
        <p>
          Ordinary chat answers have less to copy: there the model agreed with {fmt(chat.acceptedPct)}% of the guesses. The time per token still fell, from {fmt(spec.chatPlain.ms, 2)} to {fmt(chat.ms, 2)} ms. A wrong guess only costs a row, and a row is cheap; and chat answers repeat themselves more than one would think.
        </p>
        <p>
          {spec.best.k < spec.most.k ? `More guesses are not always better: with up to ${spec.most.k} the time per token was ${fmt(spec.most.ms, 2)} ms, because the extra rows cost more than the few extra tokens they bring. ` : ""}This is also a gain for a quiet server. Under load the rows are full of other requests, which is the better use of them. {chapter("beyond", "Beyond this engine")} describes the version with a small draft model, which can guess where there is nothing to copy.
        </p>
      </>
    ),
  },
];

const source = `recorded ${spec.date} on the ${spec.example.gpu}: the prompt "${spec.example.name}" of scripts/prompts_repeat.json, greedy, up to ${spec.maxDraft} guesses per step`;
const captions = [
  `time per token of the engine with CUDA Graphs at max batch 1 and ${spec.maxBatch}, docs/results.md, Throughput and latency (measured); below, the median time of a step by its number of rows over the traced runs of docs/results.md, Speculative decoding (measured, ${spec.date})`,
  `${source}; the match and the draft of each step are rebuilt from the recorded tokens by scripts/speculative_example.py and checked against the counts the engine recorded`,
  `${source}; kept and dropped rows as the engine recorded them (measured)`,
  `the slots of step ${miss.index} of the recorded answer; block size 16 from include/engine/kv/cache_size.h`,
  `docs/results.md, Speculative decoding: one request at a time at max batch ${spec.maxBatch} with CUDA Graphs, greedy, to the end of the answer; time per output token after the first, median of the runs (measured)`,
];

export const speculativeSteps = steps;
export const speculativeCaptions = captions;

const figures = [RowsFigure, GuessFigure, CheckFigure, TruncateFigure, PayoffFigure];

export function SpeculativeStory() {
  return (
    <StickyStory
      steps={steps}
      caption={(i) => captions[i]}
      figure={(i) => {
        const Figure = figures[i] ?? PayoffFigure;
        return <Figure />;
      }}
    />
  );
}
