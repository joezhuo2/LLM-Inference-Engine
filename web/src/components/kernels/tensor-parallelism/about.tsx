"use client";

import { mostRanks, tpShape } from "@/content/kernels/tensor-parallelism";
import { StickyStory, type StoryStep } from "../sticky-story";
import { SharesFigure } from "./figures";

export const tensorAbout: StoryStep = {
  id: "about",
  title: "one model, cut across GPUs",
  body: (
    <>
      <p>
        Every chapter so far ran the whole model on one GPU. A model whose weights do not fit in one GPU&apos;s memory, or that should read its weights faster than one GPU can, has to be spread over several. Tensor parallelism is one way to do it: every layer&apos;s weight matrices are cut into N pieces, and each GPU, called a rank, holds one piece of every layer.
      </p>
      <p>
        For a token, each rank multiplies the token&apos;s vector by its own pieces only. What comes out is a partial result: the right size, but only part of the answer. Adding the N partial results gives exactly what the whole matrices would have given. The ranks make that sum, an all-reduce, <span className="text-accent">twice in every layer</span>: after attention and after the MLP. TinyLlama has {tpShape.layers} layers, so one token needs {tpShape.sums} sums.
      </p>
      <p>This engine runs on a laptop with one GPU. It cuts TinyLlama the way up to {mostRanks} GPUs would need it, loads each rank onto the one GPU and checks the cut there.</p>
    </>
  ),
};

export const tensorAboutCaption = `calculated from TinyLlama's config.json: ${tpShape.layers} layers, ${tpShape.heads} query heads, ${tpShape.kv_heads} K/V heads, ${tpShape.intermediate.toLocaleString("en-US")} MLP channels, a vocabulary of ${tpShape.vocab.toLocaleString("en-US")}; 2 bytes per weight (BF16)`;

export function TensorAboutStory() {
  return <StickyStory kicker="the idea" steps={[tensorAbout]} caption={() => tensorAboutCaption} figure={() => <SharesFigure />} />;
}
