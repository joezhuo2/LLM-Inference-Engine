"use client";

import Link from "next/link";
import { gemmUs, hostTimes, journey, layerKernels, layerOps, layerUs, layersUs, share, stage, stepProfile, tokenFmt, tokenUs, tripSum, widest } from "@/content/kernels/one-token";
import { StickyStory, type StoryStep } from "../sticky-story";
import { HeadFigure, LayerFigure, TokenizeFigure, TripFigure } from "./figures";

const fmt = tokenFmt;
const us = (v: number) => `${fmt(v, v < 100 ? 1 : 0)} µs`;
const chapter = (slug: string, text: string) => (
  <Link href={`/kernels/${slug}`} className="link-underline text-accent">
    {text}
  </Link>
);
const plain = (token: string) => token.replace(/▁/g, " ").trim();
const attention = layerOps.find((o) => o.slug === "paged-attention");
const head = stage("lm-head");
const sample = stage("sample");
const hostStage = stage("host");
const gaps = stage("gaps");

const steps: StoryStep[] = [
  {
    id: "tokenize",
    title: "text becomes IDs on the host, once",
    body: (
      <>
        <p>
          The model has never seen a letter. It knows {fmt(journey.vocab)} tokens, each with a number, and the tokenizer&apos;s job is to cut text into them. First the chat template wraps the question in the markers TinyLlama was trained with. Then SentencePiece cuts the result into <span className="text-accent">{journey.promptCount} pieces</span> and looks up the ID of each.
        </p>
        <p>
          This runs on the CPU and it is cheap: {us(hostTimes.encode)} for the whole prompt, measured. It also happens only once per request, before the first step. Against a decoded token of {fmt(stepProfile.tpotMs, 2)} ms it is {fmt((hostTimes.encode / tokenUs) * 100, 1)}%.
        </p>
        <p>
          The {journey.promptCount} IDs then go through the model together in one step, the prefill, which fills the cache that every later step reads. {chapter("flash-prefill", "FlashAttention prefill")} is about that step. After it the engine decodes: one token in, one token out. Five tokens into the answer the text so far is &quot;{journey.soFar}&quot;, and the token going in is &quot;{journey.inputText.trim()}&quot;, ID {journey.inputId}.
        </p>
      </>
    ),
  },
  {
    id: "layers",
    title: `one ID goes through ${journey.layers} layers, and that is the token's time`,
    body: (
      <>
        <p>
          On the GPU the ID first becomes a row of {fmt(journey.hidden)} numbers: the embedding kernel copies row {journey.inputId} of a table, in {us(stage("embedding").us)}. That row is the token&apos;s stream. Every layer reads it, adds something to it and passes it on.
        </p>
        <p>
          A layer is the same ten operations each time. Attention lets the token look at the tokens before it: a matrix product makes its query, key and value, RoPE marks them with the position, write_kv stores the key and the value, and {chapter("paged-attention", "paged attention")} scores the query against all {journey.inputPosition + 1} keys in the cache. Then an MLP works on the token alone: widen, gate, narrow. Before each half, {chapter("add-rmsnorm", "add + RMSNorm")} folds the last result into the stream.
        </p>
        <p>
          One layer takes <span className="text-accent">{us(layerUs)}</span>, and {journey.layers} of them {us(layersUs)} in {fmt(layerKernels)} kernels. The four matrix products hold {fmt((gemmUs / layersUs) * 100)}% of that, and the widest alone, the {widest.label}, {fmt((widest.step / layersUs) * 100)}%. They are slow for one reason: each reads its whole weight matrix from memory for a single token. {chapter("gemm", "The GEMM chapter")} explains it. Attention, the famous part, is {attention ? us(attention.layer) : ""} per layer.
        </p>
      </>
    ),
  },
  {
    id: "head",
    title: `${fmt(journey.hidden)} numbers become ${fmt(journey.vocab)} scores, then one ID`,
    body: (
      <>
        <p>
          After the last layer the stream is still {fmt(journey.hidden)} numbers. One more norm rescales it ({us(stage("final-norm").us)}). Then the LM head multiplies it by a matrix with one row per token of the vocabulary and gives {fmt(journey.vocab)} scores, the logits. It is one more matrix product: <span className="text-accent">{us(head.us)}</span>.
        </p>
        <p>
          Here the model has an opinion. After &quot;{journey.soFar}&quot; the largest logit is {journey.top[0].logit} for &quot;{plain(journey.top[0].token)}&quot;. The next is {journey.top[1].logit} for &quot;{plain(journey.top[1].token)}&quot;. A gap of {journey.lead.toFixed(2)} in logits is a factor of about {fmt(Math.exp(journey.lead))} between the two probabilities, so this step is not a close call.
        </p>
        <p>
          The sampler turns the scores into one choice. This run is greedy: one kernel finds the largest logit and writes its index, {journey.chosenId}, in {us(sample.us)}. With a temperature it would draw from the scores instead; {chapter("sampling", "the sampling chapter")} covers how. The host reads the ID back and turns it into text, &quot;{journey.chosenText}&quot;, in a few microseconds. The same ID is the input of the next step.
        </p>
      </>
    ),
  },
  {
    id: "trip",
    title: "the whole trip, timed",
    body: (
      <>
        <p>
          Put the stages on one line and to one scale. A token takes {fmt(stepProfile.tpotMs, 2)} ms. The {journey.layers} layers are <span className="text-accent">{fmt(share(layersUs))}%</span> of it and the LM head {fmt(share(head.us), 1)}%. The embedding, the final norm and the sampler together are {fmt(share(stage("embedding").us + stage("final-norm").us + sample.us), 1)}%: you cannot see them on the bar.
        </p>
        <p>
          Two parts are not kernels at all. The GPU waits a few microseconds before each kernel it is handed, and with {fmt(stepProfile.kernels)} kernels in a step those gaps add up to {us(gaps.us)}, {fmt(share(gaps.us))}%. {chapter("cuda-graphs", "CUDA Graphs")} removes most of them. And between two steps the host reads the token back, checks for the end and builds the next step: {us(hostStage.us)}, {fmt(share(hostStage.us))}%.
        </p>
        <p>
          So the tokenizer and the sampler, the two ends people picture first, are not where a token&apos;s time goes. The time is in reading weights, layer after layer. The next chapter, {chapter("foundations", "foundations")}, says why that read cannot be made much faster, and the last one, {chapter("where-the-time-went", "where the time went")}, comes back to this bar after every kernel in the book has changed.
        </p>
      </>
    ),
  },
];

const profile = `results/decode_step.md, the profile docs/results.md shows: batch 1, 64 prompt and 64 output tokens, ${fmt(stepProfile.kernels)} kernels per step launched one by one`;

const captions = [
  `prompt tokens from the recorded run in results/kv_demo.json; host times from docs/profiling.md, Tokenizer on the host: median of 2,001 calls on the CPU alone (scripts/tokenizer_bench.cpp); the bars are to the scale of one decoded token`,
  `${profile}; time per op is a mean over the steps, divided by ${journey.layers} for one layer; the first layer has a norm without an add and the last an add without a norm, counted with add + RMSNorm`,
  `logits: the Hugging Face reference in BF16 on the same context (scripts/dump_logits.py), top 5 of ${fmt(journey.vocab)}; IDs printed by scripts/tokenizer_bench.cpp; kernel times from ${profile}; the sampler is the greedy sample kernel, measured outside the step's span`,
  `${profile}; kernel times are means and the span and the time per token medians, so the parts add up to ${fmt(tripSum)} of the ${fmt(tokenUs)} µs; gaps are span minus busy time; host is the time per token minus span and sampler`,
];

export const tokenSteps = steps;
export const tokenCaptions = captions;

const figures = [TokenizeFigure, LayerFigure, HeadFigure, TripFigure];

export function OneTokenStory() {
  return (
    <StickyStory
      steps={steps}
      caption={(i) => captions[i]}
      figure={(i) => {
        const Figure = figures[i] ?? TripFigure;
        return <Figure />;
      }}
    />
  );
}
