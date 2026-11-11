"use client";

import { useState } from "react";
import { chapterNumber } from "@/content/book-order";
import { chunkBudget, chunked, chunkedSteps, fmt, largestGap, medianGap, peakGoodput, perRowPrefill, prefillMs, stallDecoding, stallMs, stallPrompt } from "@/content/kernels/chunked-prefill";
import { StickyStory, type StoryStep } from "../sticky-story";
import { Toggle } from "../toggle";
import { AllPrefillFigure, BudgetBarFigure, PreemptFigure, SampledFigure, SpreadFigure } from "./figures";
import { expectAnswer, type Prediction, Predict } from "../predict";

const aloneTokens = "2,048";
const alone = prefillMs(perRowPrefill, aloneTokens);
const aloneNow = prefillMs("engine", aloneTokens);
const decoders = 3;
const prompt = 96;
const budgets = ["16", "32", "64"];

const steps16 = chunkedSteps(16, decoders, prompt).length;
const steps64 = chunkedSteps(64, decoders, prompt).length;
const prediction: Prediction = expectAnswer({
  question: "Raise the budget from 16 tokens to 64. Does the long prompt's first token come sooner or later?",
  choices: ["sooner", "later", "no change"],
  answer: steps64 < steps16 ? "sooner" : steps64 > steps16 ? "later" : "no change",
  why: `With the decoders taking a row each, the prompt needs ${steps16} steps at the smallest budget and ${steps64} at the largest. Fewer steps, each a little longer: a small budget keeps every step short for the decoders and makes the prompt wait longer.`,
}, "sooner");

function SpreadPicker() {
  const [budget, setBudget] = useState("16");
  const steps = chunkedSteps(Number(budget), decoders, prompt);
  return (
    <div className="flex h-full flex-col gap-3 tabular-nums text-xs">
      <Predict {...prediction} />
      <Toggle label="budget" choices={budgets} value={budget} onChange={setBudget} />
      <div className="min-h-0 flex-1">
        <SpreadFigure key={budget} steps={steps} budget={Number(budget)} prompt={prompt} />
      </div>
    </div>
  );
}

const steps: StoryStep[] = [
  {
    id: "stall",
    title: "one prompt stops everyone",
    body: (
      <>
        <p>
          The scheduler of chapter {chapterNumber("serving")} makes every step all prefill or all decode, and prefill goes first. A long prompt is one step, and
          every running sequence waits for it: alone, with the per-row prefill attention these runs used, the long prompt takes{" "}
          <span className="text-warn">{fmt(alone)} ms</span>.
        </p>
        <p>
          Measured under Poisson arrivals, the longest wait between two tokens of one request is{" "}
          <span className="text-warn">{largestGap("engine + graphs", "4")} ms</span> with whole-prompt steps, many times the
          median gap.
        </p>
        <p>
          Every load number in this chapter was measured before the FlashAttention prefill kernel of Part II, which shortens the long step itself. They have not been rerun with it.
        </p>
      </>
    ),
  },
  {
    id: "budget",
    title: "a step is a token budget",
    body: (
      <>
        <p>
          schedule_chunked(token_budget) caps the rows of a step. Every sequence whose prefill is done gets its decode
          row first. What is left goes to the <span className="text-accent">one</span> sequence still in prefill, as a chunk that starts where it stopped.
        </p>
        <p>
          Only if budget remains is the next waiting request admitted, first come, first served. A step never exceeds
          the budget, and a budget below max_batch is refused.
        </p>
      </>
    ),
  },
  {
    id: "spread",
    title: "the budget trades the prompt's wait for everyone's",
    body: (
      <>
        <p>
          The prompt takes as many steps as its chunks need, and the decoders keep their row in each. A smaller budget
          means more steps, so the prompt&apos;s first token comes later: the budget trades the prompt&apos;s wait against
          everyone else&apos;s.
        </p>
        <p>
          Measured with a long prompt arriving among decoding requests, a budget cuts the longest gap to{" "}
          <span className="text-accent">{fmt(stallMs(chunkBudget, "largest gap ms"))} ms</span>, and the prompt&apos;s first
          token comes later.
        </p>
        <p>
          Under load the same budget raises peak goodput to <span className="text-accent">{peakGoodput(chunked).value.toFixed(2)}</span> requests
          per second, still below vLLM, so the stalls were only part of the gap. Pick a budget on the right; the prompt and the
          decoders there are an example, not a benchmark.
        </p>
      </>
    ),
  },
  {
    id: "preempt",
    title: "preemption takes the prompt first",
    body: (
      <>
        <p>
          The preemption rule of chapter {chapterNumber("serving")} does not change: when a decoder needs a block and none is free, the victim is the back of the running list. What changes is who stands there. Blocks for the whole prompt are allocated at admission, so a sequence in prefill never runs out halfway, but it can be taken from.
        </p>
        <p>
          The sequence in prefill is always last, so partly prefilled work goes first. It returns to the front of the
          waiting queue and starts again at position <span className="text-accent">0</span>.
        </p>
      </>
    ),
  },
  {
    id: "sampled",
    group: "how it got built",
    title: "only finished chunks get logits",
    body: (
      <>
        <p>
          The sampled list holds every decode row, then each sequence whose chunk reaches the end of its length. A chunk
          that stops short has no entry and no logits.
        </p>
        <p>
          update() takes exactly one token per sampled entry, in that order. It can be empty, and the call is still
          required. first_token is stamped when the token arrives, not at the first chunk.
        </p>
        <p>
          This list is what the runner was wired to. The spec came first, with 25 tests that 14 deliberately broken schedulers fail. Joe implemented it, and one fix drafted with Claude followed: a preempted decoder now replays its prompt and its output. The batch builder then put the decode rows first and the chunk rows after them, with logit rows only for the sampled sequences. Pure decode steps kept their CUDA Graph, and a step that holds a chunk runs without one.
        </p>
      </>
    ),
  },
];

const captions = [
  `TTFT of one ${aloneTokens}-token prompt alone with per-row prefill attention and the largest gap in the QPS sweep at 4 requests per second, docs/results.md, against a median gap of ${medianGap("engine + graphs", "4")} ms; with the FlashAttention prefill kernel the same prompt alone takes ${fmt(aloneNow)} ms, and the sweep has not been rerun with it; step widths are drawn, not measured`,
  "an illustrative step of 16 token slots; the engine's budget is a parameter",
  `an example of ${decoders} decoders and a ${prompt}-token prompt, computed by the same rule as the scheduler; measured, docs/results.md: a ${fmt(stallPrompt)}-token prompt arriving among ${stallDecoding} decoding requests has a longest gap of ${fmt(stallMs("none", "largest gap ms"))} ms and a first token after ${fmt(stallMs("none", "TTFT of the long prompt ms"))} ms with no budget, and ${fmt(stallMs(chunkBudget, "largest gap ms"))} ms and ${fmt(stallMs(chunkBudget, "TTFT of the long prompt ms"))} ms with a budget of ${chunkBudget}; peak goodput in requests per second is ${peakGoodput("engine + graphs").value.toFixed(2)} for the engine, ${peakGoodput(chunked).value.toFixed(2)} with that budget and ${peakGoodput("vLLM").value.toFixed(2)} for vLLM`,
  "the running list and waiting queue of Scheduler; the sizes are a sketch",
  "the sampled list of ChunkedStep; the sizes are a sketch",
];

export const chunkedPrefillSteps = steps;
export const chunkedPrefillCaptions = captions;
export const chunkedPrefillNumbers = {
  alone,
  aloneTokens: Number(aloneTokens.replace(",", "")),
  decoders,
  prompt,
  budgets: budgets.map(Number),
  largestGap: largestGap("engine + graphs", "4"),
  medianGap: medianGap("engine + graphs", "4"),
  stallPrompt,
  stallDecoding,
  chunkBudget: Number(chunkBudget),
  gapNone: stallMs("none", "largest gap ms"),
  gapBudget: stallMs(chunkBudget, "largest gap ms"),
  ttftNone: stallMs("none", "TTFT of the long prompt ms"),
  ttftBudget: stallMs(chunkBudget, "TTFT of the long prompt ms"),
};

export function ChunkedPrefillStory() {
  return (
    <StickyStory
      steps={steps}
      caption={(i) => captions[i]}
      figure={(i) => {
        switch (i) {
          case 0:
            return <AllPrefillFigure ms={`${fmt(alone)} ms with per-row prefill attention`} />;
          case 1:
            return <BudgetBarFigure budget={16} decoders={decoders} />;
          case 2:
            return <SpreadPicker />;
          case 3:
            return <PreemptFigure />;
          default:
            return <SampledFigure />;
        }
      }}
    />
  );
}
