"use client";

import { useState } from "react";
import { chunkBudget, chunked, chunkedSteps, fmt, largestGap, medianGap, peakGoodput, prefillMs, stallDecoding, stallMs, stallPrompt } from "@/content/kernels/chunked-prefill";
import { StickyStory, type StoryStep } from "../sticky-story";
import { Toggle } from "../toggle";
import { AllPrefillFigure, BudgetBarFigure, PreemptFigure, SampledFigure, SpreadFigure } from "./figures";

const alone = prefillMs("engine", "2,048");
const decoders = 3;
const prompt = 96;
const budgets = ["16", "32", "64"];

function SpreadPicker() {
  const [budget, setBudget] = useState("16");
  const steps = chunkedSteps(Number(budget), decoders, prompt);
  return (
    <div className="flex h-full flex-col gap-3 font-mono text-xs">
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
          schedule() makes every step all prefill or all decode, and prefill goes first. A long prompt is one step, and
          every running sequence waits for it: alone, a 2,048-token prompt takes{" "}
          <span className="font-mono text-warn">{alone} ms</span>.
        </p>
        <p>
          Measured under Poisson arrivals at 4 requests per second, the longest wait between two tokens of one request is{" "}
          <span className="font-mono text-warn">{largestGap("engine + graphs", "4")} ms</span> with whole-prompt steps, against a
          median gap of {medianGap("engine + graphs", "4")} ms.
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
          row first. What is left goes to the one sequence still in prefill, as a chunk that starts where it stopped.
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
    title: "a long prompt spreads over steps",
    body: (
      <>
        <p>
          The prompt takes as many steps as its chunks need, and the decoders keep their row in each. A smaller budget
          means more steps, so the prompt&apos;s first token comes later: the budget trades the prompt&apos;s wait against
          everyone else&apos;s.
        </p>
        <p>
          Measured with a {fmt(stallPrompt)}-token prompt arriving among {stallDecoding} decoding requests: with no budget the longest gap is{" "}
          <span className="font-mono text-text">{fmt(stallMs("none", "largest gap ms"))} ms</span> and the prompt&apos;s first
          token comes after <span className="font-mono text-text">{fmt(stallMs("none", "TTFT of the long prompt ms"))} ms</span>; with
          a budget of {chunkBudget} the gap is <span className="font-mono text-text">{fmt(stallMs(chunkBudget, "largest gap ms"))} ms</span> and
          the first token <span className="font-mono text-text">{fmt(stallMs(chunkBudget, "TTFT of the long prompt ms"))} ms</span>.
        </p>
        <p>
          Under load the same budget moves peak goodput from {peakGoodput("engine + graphs").value.toFixed(2)} to{" "}
          {peakGoodput(chunked).value.toFixed(2)} requests per second; vLLM reaches {peakGoodput("vLLM").value.toFixed(2)}, so the
          stalls were only part of the gap. Pick a budget on the right; the prompt and the three decoders there are an
          example, not a benchmark.
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
          Blocks for the whole prompt are allocated at admission, so a sequence in prefill never runs out halfway. A
          decoder can still need a block when none is free, and the victim is the back of the running list.
        </p>
        <p>
          The sequence in prefill is always last, so partly prefilled work goes first. It returns to the front of the
          waiting queue and starts again at position 0.
        </p>
      </>
    ),
  },
  {
    id: "sampled",
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
      </>
    ),
  },
];

const captions = [
  `TTFT of one 2,048-token prompt alone and the largest gap in the QPS sweep, docs/results.md; step widths are drawn, not measured`,
  "an illustrative step of 16 token slots; the engine's budget is a parameter",
  "an example of 3 decoders and a 96-token prompt, computed by the same rule as the scheduler; the numbers in the text are measured, docs/results.md",
  "the running list and waiting queue of Scheduler; the sizes are a sketch",
  "the sampled list of ChunkedStep; the sizes are a sketch",
];

export function ChunkedPrefillStory() {
  return (
    <StickyStory
      steps={steps}
      caption={(i) => captions[i]}
      figure={(i) => {
        switch (i) {
          case 0:
            return <AllPrefillFigure ms={`${alone} ms`} />;
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
