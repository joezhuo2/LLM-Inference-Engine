"use client";

import { spec, specFmt } from "@/content/kernels/speculative-decoding";
import { StickyStory, type StoryStep } from "../sticky-story";
import { StepsFigure } from "./figures";

const fmt = specFmt;

export const speculativeAbout: StoryStep = {
  id: "about",
  title: "guess ahead, then check",
  body: (
    <>
      <p>
        So far a decode step has meant one token. The model reads all of its weights, writes one token, and reads them all again for the next. This chapter is about getting several tokens out of one reading.
      </p>
      <p>
        The idea has two halves. First, guess the next few tokens cheaply, without the model. Second, run the model once on the real input and on all the guesses together. Where the model&apos;s own choice agrees with a guess, that token is done, and the weights were read once for all of them.
      </p>
      <p>
        Nothing is approximated. A guess is only kept when the model, given the right tokens before it, picks the same token. A wrong guess costs some wasted arithmetic and is thrown away. So the answer is the one plain decoding writes, in fewer steps.
      </p>
      <p>
        In the answer recorded for this chapter, {fmt(spec.outputTokens)} tokens took <span className="text-accent">{fmt(spec.decodeSteps + 1)} steps where plain decoding takes {fmt(spec.outputTokens)}</span>. The model agreed with {fmt(spec.kept)} of the {fmt(spec.drafted)} guessed tokens (measured).
      </p>
    </>
  ),
};

export const speculativeAboutCaption = `recorded ${spec.date} on the ${spec.example.gpu}: the prompt "${spec.example.name}" of scripts/prompts_repeat.json, greedy, up to ${spec.maxDraft} guesses per step; each block is one step and its width is the tokens that step wrote (measured)`;

export function SpeculativeAboutStory() {
  return <StickyStory kicker="the idea" steps={[speculativeAbout]} caption={() => speculativeAboutCaption} figure={() => <StepsFigure />} />;
}
