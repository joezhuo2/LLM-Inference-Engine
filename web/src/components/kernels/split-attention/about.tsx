"use client";

import { motion } from "motion/react";
import { Frame, Label } from "../frame";
import { useLoop } from "../loop";
import { StickyStory, type StoryStep } from "../sticky-story";
import { splitNumbers } from "./story";

export const splitAboutSizes = { context: 2048, tiles: 16, drawnSms: 8, heads: 32, sms: splitNumbers.sms };

const { context, tiles, drawnSms, heads, sms } = splitAboutSizes;
const fmt = (n: number) => n.toLocaleString("en-US");

export const splitAbout: StoryStep = {
  id: "about",
  title: "what splitting attention means",
  body: (
    <>
      <p>
        While a token is decoded, attention takes that token&apos;s query in one head, scores it against every key in the KV cache, turns the scores into weights with a softmax and adds up the cached values with those weights. TinyLlama does this in {heads} query heads, and each head&apos;s work is one walk along its whole context.
      </p>
      <p>
        The walking is done by the GPU&apos;s streaming multiprocessors, or SMs. An SM is one of the GPU&apos;s processors, with its own registers, its own fast shared memory and room for several groups of threads at a time; this GPU has {sms}. A kernel hands its work to the SMs as thread blocks: a block is a group of threads that starts on one SM, stays there until it is done and shares that SM&apos;s memory.
      </p>
      <p>
        At batch 1 the attention kernel gets one block per query head, {heads} in all, and each walks its head&apos;s context alone. That leaves most of the room on the SMs empty, and a longer context only makes every block take longer.
      </p>
      <p>
        Splitting cuts each context into pieces and gives every piece its own block, so more blocks run side by side. A block that has seen only a piece cannot write the answer, so it writes a partial result: the largest score it saw, its sum of exponentials and its weighted values. A second, small kernel combines the pieces into the same answer one walk over the whole context gives.
      </p>
    </>
  ),
};

export const splitAboutCaption = `illustrative: one head of ${heads}, its ${fmt(context)} tokens drawn as ${tiles} tiles, and ${drawnSms} SMs drawn of this GPU's ${sms}`;

const left = 20;
const width = 520;
const tile = width / tiles;
const stripY = 62;
const smY = 116;
const smW = 58;
const smPitch = 66;
const split = 4;
const rangeW = width / split;
const smX = (j: number) => left + j * smPitch;
const smFor = (i: number, s: number) => i * (drawnSms / s) + Math.floor((drawnSms / s - 1) / 2);

export function SplitAboutFigure() {
  const { on, grow } = useLoop(14);
  return (
    <Frame label={`One head's ${fmt(context)}-token context. Unsplit, one thread block walks all of it on one SM. Cut into ${split} pieces, ${split} blocks walk a piece each on ${split} SMs, each writes a partial result, and a merge kernel combines them`}>
      <Label x={left} y={24}>{`one head's query against its ${fmt(context)} cached keys and values`}</Label>
      <motion.g {...on(0.02, 0.48, false)}>
        <Label x={left} y={46} className="fill-text">{`no split: one thread block walks all ${fmt(context)} tokens`}</Label>
      </motion.g>
      <motion.g {...on(0.5, 0.97)}>
        <Label x={left} y={46} className="fill-text">{`${split} splits: ${split} blocks walk ${fmt(context / split)} tokens each, side by side`}</Label>
      </motion.g>
      {Array.from({ length: tiles }, (_, k) => (
        <rect key={k} x={left + k * tile} y={stripY} width={tile - 2} height={20} rx={2} className="fill-baseline/60" />
      ))}
      <motion.g {...on(0.02, 0.48, false)}>
        <motion.rect x={left} y={stripY} width={width - 2} height={20} rx={2} className="fill-accent/70" {...grow(0.06, 0.4)} />
      </motion.g>
      <motion.g {...on(0.5, 0.97)}>
        {Array.from({ length: split - 1 }, (_, s) => (
          <line key={s} x1={left + (s + 1) * rangeW - 1} y1={stripY - 6} x2={left + (s + 1) * rangeW - 1} y2={stripY + 26} className="stroke-accent" strokeWidth={2} />
        ))}
        {Array.from({ length: split }, (_, s) => (
          <motion.rect key={s} x={left + s * rangeW} y={stripY} width={rangeW - 4} height={20} rx={2} className="fill-accent/70" {...grow(0.56, 0.64)} />
        ))}
      </motion.g>
      {Array.from({ length: drawnSms }, (_, j) => (
        <g key={j}>
          <rect x={smX(j)} y={smY} width={smW} height={46} rx={3} className="fill-surface stroke-border" />
          <Label x={smX(j) + 6} y={smY + 14}>{`SM ${j}`}</Label>
        </g>
      ))}
      <motion.g {...on(0.02, 0.48, false)}>
        <rect x={smX(smFor(0, 1)) + 14} y={smY + 22} width={30} height={18} rx={2} className="fill-accent" />
        <Label x={left} y={184} className="fill-text">{`SMs busy: 1 of ${drawnSms}`}</Label>
        <Label x={left} y={226}>the block holds the whole softmax,</Label>
        <Label x={left} y={244}>so it writes the output itself</Label>
      </motion.g>
      <motion.g {...on(0.5, 0.97)}>
        {Array.from({ length: split }, (_, i) => (
          <rect key={i} x={smX(smFor(i, split)) + 14} y={smY + 22} width={30} height={18} rx={2} className="fill-accent" />
        ))}
        <Label x={left} y={184} className="fill-text">{`SMs busy: ${split} of ${drawnSms}`}</Label>
      </motion.g>
      {Array.from({ length: split }, (_, i) => {
        const x = smX(smFor(i, split)) + 4;
        return (
          <motion.g key={i} {...on(0.68 + i * 0.015, 0.97)}>
            <rect x={x} y={200} width={50} height={12} rx={2} className="fill-accent" />
            <Label x={x + 4} y={210} className="fill-bg">m</Label>
            <rect x={x} y={214} width={50} height={12} rx={2} className="fill-accent/60" />
            <Label x={x + 4} y={224} className="fill-bg">l</Label>
            {Array.from({ length: 4 }, (_, k) => (
              <rect key={k} x={x} y={229 + k * 5} width={50} height={3} rx={1} className="fill-baseline" />
            ))}
            <line x1={x + 25} y1={252} x2={300} y2={292} className="stroke-accent" />
          </motion.g>
        );
      })}
      <motion.g {...on(0.78, 0.97)}>
        <rect x={250} y={292} width={100} height={28} rx={3} className="fill-accent/10 stroke-accent" />
        <Label x={300} y={310} anchor="middle" className="fill-text">merge kernel</Label>
        <Label x={364} y={310}>rescales the partial results and adds them up</Label>
      </motion.g>
      <motion.g {...on(0.86, 0.97)}>
        <Label x={left} y={344} className="fill-accent">the head&apos;s output: the same answer as one walk</Label>
      </motion.g>
      <Label x={left} y={378}>a partial result: m, the largest score; l, the sum of exponentials; and the weighted values</Label>
    </Frame>
  );
}

export function SplitAboutStory() {
  return <StickyStory kicker="the idea" steps={[splitAbout]} caption={() => splitAboutCaption} figure={() => <SplitAboutFigure />} />;
}
