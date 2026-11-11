"use client";

import { hostTimes, journey, share, layersUs, stepProfile, tokenFmt } from "@/content/kernels/one-token";
import { StickyStory, type StoryStep } from "../sticky-story";
import { PipelineFigure } from "./figures";

const fmt = tokenFmt;

export const tokenAbout: StoryStep = {
  id: "about",
  title: "the path of one token",
  body: (
    <>
      <p>
        Ask the engine &quot;{journey.user}&quot; and it answers &quot;{journey.answer}&quot; This chapter follows one token of that answer, <span className="text-accent">&quot;{journey.chosenText.trim()}&quot;</span>, from the text of the question to the moment it is an ID, and gives the measured time of every stage on the way.
      </p>
      <p>
        A language model does not write a sentence. It writes one token, a piece of a word, and then it is run again for the next. Each run is called a decode step. The step that produces &quot;{journey.chosenText.trim()}&quot; takes in the token before it, &quot;{journey.inputText.trim()}&quot;, and gives back a score for each of the {fmt(journey.vocab)} tokens the model knows.
      </p>
      <p>
        There are seven stations. Two run on the host, at the ends: text becomes IDs, and an ID becomes text. Five run on the GPU: a lookup, {journey.layers} layers, a norm, the LM head and the sampler. The layers take {fmt(share(layersUs))}% of the {fmt(stepProfile.tpotMs, 2)} ms. The rest of the book is about them.
      </p>
      <p>Every later chapter opens one station. Read this one for the map.</p>
    </>
  ),
};

export const tokenAboutCaption = `the question, the answer and its tokens are the recorded greedy run in results/kv_demo.json; host times from docs/profiling.md, Tokenizer on the host (encode ${fmt(hostTimes.encode, 1)} µs); GPU times from results/decode_step.md, batch 1, 64 prompt and 64 output tokens, so they are the times of a typical decode step, not of this exact one`;

export function OneTokenAboutStory() {
  return <StickyStory kicker="the idea" steps={[tokenAbout]} caption={() => tokenAboutCaption} figure={() => <PipelineFigure />} />;
}
