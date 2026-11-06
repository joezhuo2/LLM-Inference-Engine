"use client";

import { motion } from "motion/react";
import trace from "@/content/kernels/cuda-graphs-trace.json";
import { kvDemo, launchGaps, modelShape } from "@/lib/data";
import { Frame, Label } from "../frame";
import { useLoop } from "../loop";
import { StickyStory, type StoryStep } from "../sticky-story";

const shape = modelShape();
const headDim = shape.hidden / shape.heads;
const kvDim = shape.kv_heads * headDim;
const perLayerWeights = shape.hidden * (shape.hidden + 2 * kvDim) + shape.hidden * shape.hidden + 3 * shape.hidden * shape.intermediate;
const params = shape.layers * perLayerWeights + 2 * shape.vocab * shape.hidden;
const graphs = launchGaps("graphs");
const demo = kvDemo().requests.find((r) => r.name === "capital") ?? kvDemo().requests[0];

const tokenText = (t: string) => (t === "</s>" ? "end" : t === "<0x0A>" ? "↵" : t.replace(/▁/g, ""));

export const aboutNumbers = {
  hidden: shape.hidden,
  vocab: shape.vocab,
  layers: shape.layers,
  kernels: graphs.kernels,
  perLayer: trace.eager.kernels_in_layer.length,
  stepMs: graphs.span_us / 1000,
  prompt: demo.prompt,
  answer: demo.answer,
  promptLast: tokenText(demo.prompt_tokens[demo.prompt_tokens.length - 1]),
  tokens: demo.output_tokens.map(tokenText),
  ops: trace.eager.kernels_in_layer.map((k) => k.op),
};

const n = aboutNumbers;
const fmt = (v: number) => v.toLocaleString("en-US");

export const foundationsAbout: StoryStep = {
  id: "about",
  title: "what an inference engine does",
  body: (
    <>
      <p>
        An inference engine is the program that runs a trained language model: it loads the weights onto the GPU, takes a prompt and writes the answer one token at a time. This one runs TinyLlama, whose {(params / 1e9).toFixed(1)} billion weights take {((params * 2) / 1e9).toFixed(1)} GB in BF16. A token is a word or a piece of one, numbered by its place in a vocabulary of {fmt(n.vocab)}: asked &ldquo;{n.prompt}&rdquo;, TinyLlama answered &ldquo;{n.answer}&rdquo; in {n.tokens.length - 1} tokens and a stop token.
      </p>
      <p>
        The prompt goes through the model once, all its tokens together. After that every new token costs one decode step. The id of the last token picks one row of the embedding table, {fmt(n.hidden)} numbers that stand for that token. The row passes through the {n.layers} layers in turn, and each layer adds to it what attention reads from the earlier tokens and what its MLP computes. At the end the LM head, a matrix of {fmt(n.vocab)} rows, turns the {fmt(n.hidden)} numbers into {fmt(n.vocab)} scores called logits, one for every token in the vocabulary. One token is picked from them, and its id is the input of the next step.
      </p>
      <p>
        Each of those pieces is one or more GPU kernels, programs the CPU hands to the GPU to run. In the profile a decode step is <span className="text-accent">{n.kernels} kernels</span>, {n.perLayer} in every layer and a few around them, and one more picks the token; the whole step takes about <span className="text-text">{n.stepMs.toFixed(1)} ms</span> on this GPU. The rest of the book takes those kernels one chapter at a time. This chapter starts underneath them all: how a GPU runs a kernel, where its data lives, and why a decode step waits on memory and not on arithmetic.
      </p>
    </>
  ),
};

export const foundationsAboutCaption = `illustrative: ${n.layers} plates for ${n.layers} layers, a few bars for ${fmt(n.vocab)} logits; the answer and its tokens from results/kv_demo.json, kernel count and step time from the CUDA Graphs profile`;

const shown = 6;
const pick = shown - 1;
const charW = 6.4;
const tokenY = 58;
const tokenH = 22;
const tokenBoxes = (() => {
  let x = 20;
  return n.tokens.slice(0, shown).map((t) => {
    const w = Math.max(22, t.length * charW + 12);
    const box = { x, w, text: t };
    x += w + 5;
    return box;
  });
})();

const top = 130;
const bottom = 260;
const embX = 20;
const embW = 56;
const layerX0 = 104;
const layerPitch = 12;
const layerW = 7;
const headX = layerX0 + n.layers * layerPitch + 14;
const headW = 40;
const barsX = headX + headW + 18;
const bars = [0.35, 0.55, 0.3, 0.7, 0.45, 0.25, 1, 0.5, 0.4, 0.6, 0.3, 0.45];
const winner = bars.indexOf(1);
const barPitch = 9;
const barMax = 100;
const rowY = top + 52;
const packetW = 22;
const travel: [number, number] = [0.12, 0.6];
const layerTime = (i: number) => travel[0] + ((i + 0.5) / n.layers) * (travel[1] - travel[0]);

export function FoundationsAboutFigure() {
  const { on, move, grow } = useLoop(14);
  const last = tokenBoxes[pick - 1];
  const next = tokenBoxes[pick];
  const winX = barsX + winner * barPitch + 3;
  return (
    <Frame label={`One decode step: the id of the last token picks a row of the embedding table, the row passes through ${n.layers} layers, the LM head gives ${n.vocab} logits, the highest is picked and the new token joins the answer`}>
      <Label x={20} y={24} className="fill-text">{"one decode step: the last token in, the next token out"}</Label>
      {tokenBoxes.slice(0, pick).map((b, i) => (
        <g key={i}>
          <rect x={b.x} y={tokenY} width={b.w} height={tokenH} rx={3} className={i === pick - 1 ? "fill-accent/20 stroke-accent" : "fill-surface stroke-border"} strokeWidth={0.75} />
          <Label x={b.x + b.w / 2} y={tokenY + 15} anchor="middle" className="fill-text">
            {b.text}
          </Label>
        </g>
      ))}
      <motion.g {...on(0.02, 0.83, false)}>
        <rect x={next.x} y={tokenY} width={next.w} height={tokenH} rx={3} className="fill-none stroke-border" strokeWidth={0.75} strokeDasharray="3 3" />
        <Label x={next.x + next.w / 2} y={tokenY + 15} anchor="middle">
          ?
        </Label>
      </motion.g>
      <motion.g {...on(0.84)}>
        <rect x={next.x} y={tokenY} width={next.w} height={tokenH} rx={3} className="fill-accent" />
        <Label x={next.x + next.w / 2} y={tokenY + 15} anchor="middle" className="fill-surface">
          {next.text}
        </Label>
      </motion.g>
      <Label x={20} y={tokenY - 10}>{`the answer so far; the step reads "${last.text}" and writes the next token`}</Label>
      <motion.path d={`M ${last.x + last.w / 2} ${tokenY + tokenH} C ${last.x + last.w / 2} ${tokenY + 90}, ${embX + embW / 2} ${rowY - 60}, ${embX + embW / 2} ${rowY - 6}`} className="fill-none stroke-accent" strokeWidth={1} strokeDasharray="3 3" {...on(0.03, 0.2, false)} />
      <rect x={embX} y={top} width={embW} height={bottom - top} rx={3} className="fill-surface stroke-border" strokeWidth={0.75} />
      {Array.from({ length: 14 }, (_, i) => (
        <line key={i} x1={embX + 4} x2={embX + embW - 4} y1={top + 8 + i * 9} y2={top + 8 + i * 9} className="stroke-border" strokeWidth={0.75} />
      ))}
      <motion.rect x={embX + 2} y={rowY - 3} width={embW - 4} height={6} rx={2} className="fill-accent" {...on(0.06, 0.9)} />
      {Array.from({ length: n.layers }, (_, i) => (
        <g key={i}>
          <rect x={layerX0 + i * layerPitch} y={top + 10} width={layerW} height={bottom - top - 20} rx={2} className="fill-baseline" />
          <motion.rect x={layerX0 + i * layerPitch} y={top + 10} width={layerW} height={bottom - top - 20} rx={2} className="fill-accent/40" {...on(layerTime(i), 0.9)} />
        </g>
      ))}
      <motion.rect x={embX + embW + 4} y={rowY - 3} width={packetW} height={6} rx={2} className="fill-accent" {...move(travel[0] - 0.02, travel[1], headX - embX - embW - packetW - 8, 0, false, 0.64)} />
      <rect x={headX} y={top} width={headW} height={bottom - top} rx={3} className="fill-baseline" />
      <motion.rect x={headX} y={top} width={headW} height={bottom - top} rx={3} className="fill-accent/40" {...on(0.62, 0.9)} />
      {bars.map((h, i) => (
        <motion.rect key={i} x={barsX + i * barPitch} y={bottom - h * barMax} width={6} height={h * barMax} rx={1} className={i === winner ? "fill-accent" : "fill-baseline"} {...grow(0.66, 0.76, "y")} />
      ))}
      <motion.path d={`M ${winX} ${bottom - barMax - 6} C ${winX} ${tokenY + 70}, ${next.x + next.w / 2} ${tokenY + 60}, ${next.x + next.w / 2} ${tokenY + tokenH + 2}`} className="fill-none stroke-accent" strokeWidth={1} strokeDasharray="3 3" {...on(0.78, 0.9, false)} />
      <Label x={embX} y={bottom + 18} className="fill-text">embedding</Label>
      <Label x={embX} y={bottom + 32}>{`${fmt(n.vocab)} rows`}</Label>
      <Label x={layerX0} y={bottom + 18} className="fill-text">{`${n.layers} layers`}</Label>
      <Label x={layerX0} y={bottom + 32}>{`${n.perLayer} kernels each`}</Label>
      <Label x={headX} y={bottom + 18} className="fill-text">LM head</Label>
      <Label x={headX} y={bottom + 32}>{`${fmt(n.hidden)} → ${fmt(n.vocab)}`}</Label>
      <Label x={barsX + bars.length * barPitch} y={bottom + 18} anchor="end" className="fill-text">logits</Label>
      <Label x={barsX + bars.length * barPitch} y={bottom + 32} anchor="end">highest wins</Label>
      <Label x={20} y={330}>{`a row of ${fmt(n.hidden)} numbers goes in, ${fmt(n.vocab)} scores come out, one token is picked`}</Label>
      <Label x={20} y={348} className="fill-accent">{`each box is GPU kernels: ${n.kernels} per step in the profile, then one to pick the token`}</Label>
      <Label x={20} y={366}>the new token&apos;s id is the input of the next step</Label>
    </Frame>
  );
}

export function FoundationsAboutStory() {
  return <StickyStory kicker="the idea" steps={[foundationsAbout]} caption={() => foundationsAboutCaption} figure={() => <FoundationsAboutFigure />} />;
}
