"use client";

import { measured } from "@/content/kernels/types";
import { StickyStory, type StoryStep } from "../sticky-story";
import { CacheHomeFigure, DecodeFigure, PrefillFigure, SideBySideFigure } from "./prologue-figures";

const prefill = "prefill-against-prompt-length";
const ttft = (tokens: string) => measured(prefill, { system: "engine" }, tokens, 0, "results");
const perToken = (tokens: string) => measured(prefill, { system: "engine" }, tokens, 1, "results");
const ttft128 = ttft("128");
const ttft2048 = ttft("2,048");
const perToken128 = perToken("128");
const perToken2048 = perToken("2,048");

const roof = "roofline";
const whole = (column: string) => measured(roof, { "whole decode step": "engine", rows: "1" }, column, 1, "results");
const gbs = whole("GB/s");
const percent = whole("% of roof");

const longStep = "split-attention-in-cuda-graphs-branch-43";
const stepMs = measured(longStep, { "max batch": "1", prompt: "1,984" }, "this branch graphs");
const ratio = Math.round(ttft2048 / stepMs);

const steps: StoryStep[] = [
  {
    id: "prefill",
    title: "prefill: the whole prompt at once",
    body: (
      <>
        <p>
          The prompt is known before generation starts, so the model takes all of it in one pass: every position goes
          through every layer side by side, and each layer writes the keys and values of every position into the KV cache.
        </p>
        <p>
          Time to first token is this pass. On this engine a long prompt took{" "}
          <span className="text-accent">{ttft2048.toLocaleString("en-US")} ms</span>, and each prompt token costs more than it does in a short prompt.
        </p>
      </>
    ),
  },
  {
    id: "decode",
    title: "decode: one token at a time",
    body: (
      <>
        <p>
          After the prompt the model emits one token per step, and each token is the input of the next, so steps cannot
          overlap. A step computes little for one new position, but it reads every weight and every cached key and value.
        </p>
        <p>
          Counted from the model config, with one request in flight the engine reaches{" "}
          <span className="text-accent">{percent}%</span> of the GPU&apos;s measured peak memory bandwidth: the step waits on memory, not
          on arithmetic.
        </p>
      </>
    ),
  },
  {
    id: "cache",
    title: "why the cache exists",
    body: (
      <>
        <p>
          Prefill computes the keys and values of the prompt once and writes them to the KV cache. Decode reads them back.
          Without the cache every new token would rerun the whole prefill over a context that grew by one.
        </p>
        <p>
          One decode step at a context as long as the prompt took about <span className="text-accent">{ratio}x</span> less time than the prefill that built the cache. Both times are in the caption.
        </p>
      </>
    ),
  },
  {
    id: "home",
    title: "the cache has to live somewhere",
    body: (
      <>
        <p>
          The cache grows by one token per step for every request in flight, and nobody knows in advance how long an answer
          will be. Reserving the longest possible answer for every request leaves memory the others could use.
        </p>
        <p>
          So the cache is cut into 16-token blocks handed out as a sequence grows, all from one shared pool. Paged attention is
          the kernel that reads them back, and how it does that is next.
        </p>
      </>
    ),
  },
];

const captions = [
  `an illustrative 8-token prompt; single-request TTFT, median of 5 runs, from docs/results.md: ${ttft128} ms for 128 tokens (${perToken128} ms per token) and ${ttft2048.toLocaleString("en-US")} ms for 2,048 tokens (${perToken2048} ms per token)`,
  `illustrative tokens; ${gbs} GB/s is ${percent}% of the measured peak, the engine at batch 1 without CUDA Graphs, from docs/results.md`,
  `${ttft2048.toLocaleString("en-US")} ms prefill of 2,048 tokens (docs/results.md) against ${stepMs} ms per decode step at a 1,984-token prompt, batch 1, with CUDA Graphs (docs/profiling.md)`,
  "an illustrative 96-token sequence in a 16-block pool; the blocks and their order are made up",
];

export const prologueSteps = steps;
export const prologueCaptions = captions;
export const prologueNumbers = { ttft128, ttft2048, perToken128, perToken2048, gbs, percent, stepMs, ratio };

export function PrologueStory() {
  return (
    <StickyStory
      steps={steps}
      caption={(i) => captions[i]}
      figure={(i) => {
        switch (i) {
          case 0:
            return <PrefillFigure />;
          case 1:
            return <DecodeFigure percent={percent} />;
          case 2:
            return <SideBySideFigure prefillMs={ttft2048} stepMs={stepMs} />;
          default:
            return <CacheHomeFigure />;
        }
      }}
    />
  );
}
