"use client";

import { eagerStage, timeFmt } from "@/content/kernels/where-the-time-went";
import { modelShape } from "@/lib/data";
import { StickyStory, type StoryStep } from "../sticky-story";
import { ProfileFigure } from "./figures";

const s = eagerStage;
const layers = modelShape().layers;

export const timeAbout: StoryStep = {
  id: "about",
  title: "where a token's time goes",
  body: (
    <>
      <p>
        Every chapter changed one kernel and then asked the same question: did the step get shorter? The answer comes from a step profile. Nsight Systems records when each kernel of a decode step starts and ends on the GPU, and a script follows every kernel back to the operation that launched it: the QKV projection, RoPE, attention and so on, through all {layers} layers.
      </p>
      <p>
        A profile gives three lengths for one step. The span runs from the start of the first kernel to the end of the last. The busy time is the part of the span in which a kernel was running. The rest of the span is launch gaps, a few microseconds before each kernel. Time per op splits the kernel time by operation.
      </p>
      <p>
        The time a user waits for a token is longer than the span. Between two steps the host schedules, builds the batch, samples and reads the token back. So the kernels add up to less than the step, and the step to less than the token: <span className="text-accent">{timeFmt(s.busy)} µs</span> busy, {timeFmt(s.span)} µs of span and {timeFmt(s.tpot ?? 0, 2)} ms per token in the same run.
      </p>
      <p>
        One caution. This laptop GPU&apos;s clocks move between runs, so two profiles taken weeks apart can differ by more than a small kernel saved. Compare shares inside one profile, and trust a change between profiles only where it is larger than that.
      </p>
    </>
  ),
};

export const timeAboutCaption = `results/decode_step.md, the profile docs/results.md shows: batch 1, 64 prompt and 64 output tokens, ${s.kernels} kernels per step launched one by one; span, busy time and time per token are medians, time per op is a mean over the steps, so the ops add up to ${timeFmt(s.kernelSum)} µs`;

export function TimeAboutStory() {
  return <StickyStory kicker="the idea" steps={[timeAbout]} caption={() => timeAboutCaption} figure={() => <ProfileFigure />} />;
}
