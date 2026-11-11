import { gemmUs, hostTimes, journey, layerUs, layersUs, share, stage, stepProfile, tokenFmt, tokenUs } from "@/content/kernels/one-token";
import type { ChapterReview } from "./types";

const fmt = tokenFmt;
const us = (v: number) => `${fmt(v, v < 100 ? 1 : 0)} µs`;

export const review: ChapterReview = {
  lesson: "A token's time is spent reading weights in the layers, not at the two ends people picture first.",
  why: "It gives the map every later chapter fills in, and one scale to judge each stage against.",
  takeaways: [
    `Text becomes IDs once per request on the host: ${journey.promptCount} tokens in ${us(hostTimes.encode)} (measured), ${fmt((hostTimes.encode / tokenUs) * 100, 1)}% of one decoded token.`,
    `A decode step takes one ID in and gives one ID out. Its ${journey.layers} layers take ${us(layersUs)} of the ${fmt(stepProfile.tpotMs, 2)} ms (measured), and the four matrix products are ${fmt((gemmUs / layersUs) * 100)}% of that.`,
    `The LM head turns ${fmt(journey.hidden)} numbers into ${fmt(journey.vocab)} logits in ${us(stage("lm-head").us)}, and the greedy sampler keeps one ID in ${us(stage("sample").us)} (measured).`,
  ],
  questions: [
    {
      question: `The prompt has ${journey.promptCount} tokens and the answer's text ${hostTimes.answerTokens}. How many times does the tokenizer encode, and how many of the ${hostTimes.answerTokens} come from a decode step?`,
      answer: `It encodes once. The prompt goes through in one prefill step, which gives the first token; each of the other ${hostTimes.answerTokens - 1} comes from a decode step.`,
      why: "Encoding is per request, not per token. After the prefill every step takes in the token the step before produced.",
    },
    {
      question: `One layer takes ${us(layerUs)}. Which would shorten a token more: a sampler that takes no time at all, or layers that are 1% faster?`,
      answer: `The layers: 1% of ${us(layersUs)} is ${us(layersUs / 100)}, and the whole sampler is ${us(stage("sample").us)}.`,
      why: `The layers are ${fmt(share(layersUs))}% of the token. A stage can give back at most its own share.`,
    },
    {
      question: `The step's kernels are done after the sampler writes ID ${journey.chosenId}. Why does the user wait ${us(stage("host").us)} more before the next step starts?`,
      answer: "The host reads the ID back, checks whether the answer has ended, and builds the next step.",
      why: "That work is between two steps, outside the span of the kernels, so no kernel change can remove it.",
    },
  ],
};
