"use client";

import { motion } from "motion/react";
import { modelShape, samplingLogits } from "@/lib/data";
import { Frame, Label } from "../frame";
import { useLoop } from "../loop";
import { StickyStory, type StoryStep } from "../sticky-story";
import { mulberry } from "./figures";
import { buildGroups, softmaxMass } from "./sampler";

const dump = samplingLogits();
const shape = modelShape();
const context = dump.contexts.find((c) => c.name === "story") ?? dump.contexts[0];
const certain = dump.contexts.find((c) => c.name === "capital") ?? dump.contexts[0];
const SHOWN = 8;

const name = (t: string) => t.replaceAll("▁", "").replace("<0x0A>", "⏎");
const pct = (v: number) => `${Math.round(v * 100)}%`;
const fmt = (v: number) => v.toLocaleString("en-US");

function oddsOf(c: typeof context, temperature: number): number[] {
  const groups = buildGroups(c);
  if (temperature <= 0) return [1, ...Array<number>(SHOWN).fill(0)];
  const mass = softmaxMass(
    groups.map((g) => g.logit / temperature),
    groups.map((g) => g.count),
  );
  const top = mass.slice(0, SHOWN);
  return [...top, Math.max(0, 1 - top.reduce((a, b) => a + b, 0))];
}

export const aboutContext = {
  answer: context.answer_so_far,
  vocab: context.vocab,
  tokens: context.tokens.slice(0, SHOWN).map(name),
  logits: context.logits.slice(0, SHOWN),
  others: context.vocab - SHOWN,
  odds: (temperature: number) => oddsOf(context, temperature),
};

export function drawFrom(odds: number[], u: number): number {
  let run = 0;
  for (let i = 0; i < odds.length; i++) {
    run += odds[i];
    if (u < run) return i;
  }
  return odds.findIndex((p) => p > 0);
}

const atOne = aboutContext.odds(1);
const sure = oddsOf(certain, 1)[0];

export const samplingAbout: StoryStep = {
  id: "about",
  title: "what sampling is",
  body: (
    <>
      <p>
        The last layer of the model, the LM head, turns the token&apos;s final vector of {fmt(shape.hidden)} values into {fmt(context.vocab)} logits, one score for every token in the vocabulary. A logit can be any number, and only the gaps between them matter. Sampling turns those scores into the one token id the model writes next.
      </p>
      <p>
        Softmax makes them probabilities: divide each logit by a temperature T, raise e to it, and divide by the sum over all {fmt(context.vocab)}. Greedy decoding takes the largest logit, so a prompt always gets the same answer. Sampling draws one token at random with those probabilities.
      </p>
      <p>
        After &ldquo;{certain.answer_so_far}&rdquo; TinyLlama puts {pct(sure)} on {name(certain.tokens[0])}, the start of Ottawa, so both give the same word. After &ldquo;{context.answer_so_far}&rdquo; the likeliest name, {aboutContext.tokens[0]}, gets {pct(atOne[0])} at T = 1: greedy writes {aboutContext.tokens[0]} every time, sampling a different name most times.
      </p>
      <p>
        Temperature sets how bold the draw is. At T = 0.5 the gaps double and {aboutContext.tokens[0]} rises to {pct(aboutContext.odds(0.5)[0])}; at T = 1.5 it falls to {pct(aboutContext.odds(1.5)[0])}. Toward T = 0 sampling becomes greedy, and the engine treats T = 0 as greedy. The rest of the chapter is how the GPU picks without sorting the vocabulary.
      </p>
    </>
  ),
};

export const samplingAboutCaption = `TinyLlama's logits for “${context.answer_so_far}”, BF16 on the CPU: the ${SHOWN} likeliest of ${fmt(context.vocab)} tokens and the rest in one column; the draws are a simulation`;

const temps = [0.5, 1, 1.5];
const seeds = [3, 11, 5];
const phaseDraws = temps.map((t, k) => drawFrom(aboutContext.odds(t), mulberry(seeds[k])()));
const COLS = SHOWN + 1;
const x0 = 92;
const pitch = 50;
const bw = 34;
const logitBase = 100;
const logitScale = 4.5;
const probBase = 322;
const probScale = 190;
const span = 1 / temps.length;
const colX = (i: number) => x0 + i * pitch;

function heightKeys(i: number) {
  const hs = temps.map((t) => aboutContext.odds(t)[i] * probScale);
  const values: number[] = [];
  const times: number[] = [];
  temps.forEach((_, k) => {
    const start = k * span;
    values.push(hs[k], hs[k]);
    times.push(start + (k === 0 ? 0 : 0.08 * span), start + span * 0.88);
  });
  values.push(hs[0]);
  times.push(1);
  return { values: [hs[0], ...values], times: [0, ...times] };
}

export function SamplingAboutFigure() {
  const { on, move, reduced } = useLoop(18);
  const still = temps.indexOf(1);
  const label = (i: number) => (i < SHOWN ? aboutContext.tokens[i] : "others");
  return (
    <Frame label={`Sampling: ${fmt(aboutContext.vocab)} logits become probabilities by softmax of logit over T. Greedy always takes the largest; a draw picks a token at random by its probability. Lower T sharpens the odds, higher T flattens them`}>
      <Label x={20} y={24}>{`the last layer gives ${fmt(aboutContext.vocab)} logits, one per token (the ${SHOWN} largest shown)`}</Label>
      <Label x={20} y={logitBase - 4}>logits</Label>
      {aboutContext.logits.map((l, i) => (
        <g key={`l${i}`}>
          <rect x={colX(i)} y={logitBase - l * logitScale} width={bw} height={l * logitScale} rx={2} className="fill-baseline" />
          <Label x={colX(i) + bw / 2} y={logitBase - l * logitScale - 5} anchor="middle">
            {l.toFixed(1)}
          </Label>
        </g>
      ))}
      <Label x={colX(SHOWN) + bw / 2} y={logitBase - 4} anchor="middle">
        {"…"}
      </Label>
      <line x1={x0 - 4} y1={logitBase} x2={colX(COLS - 1) + bw + 4} y2={logitBase} className="stroke-border" />
      {temps.map((t, k) => (
        <motion.g key={`t${k}`} {...on(k * span, (k + 1) * span, k === still)}>
          <Label x={20} y={124} className="fill-text">{`p = softmax(logit / T) at T = ${t}${t < 1 ? ": sharper" : t > 1 ? ": flatter" : ": the model's own odds"}`}</Label>
        </motion.g>
      ))}
      <Label x={20} y={probBase - 4}>p</Label>
      {Array.from({ length: COLS }, (_, i) => {
        const keys = heightKeys(i);
        const final = aboutContext.odds(1)[i] * probScale;
        return (
          <motion.rect
            key={`p${i}`}
            x={colX(i)}
            width={bw}
            rx={2}
            className={i === SHOWN ? "fill-accent/50" : "fill-accent"}
            initial={false}
            animate={reduced ? { y: probBase - final, height: final } : { y: keys.values.map((h) => probBase - h), height: keys.values }}
            transition={reduced ? { duration: 0 } : { duration: 18, repeat: Infinity, ease: "easeInOut", times: keys.times }}
          />
        );
      })}
      <line x1={x0 - 4} y1={probBase} x2={colX(COLS - 1) + bw + 4} y2={probBase} className="stroke-border" />
      {Array.from({ length: COLS }, (_, i) => (
        <Label key={`n${i}`} x={colX(i) + bw / 2} y={probBase + 16} anchor="middle" className={i === 0 ? "fill-text" : "fill-muted"}>
          {label(i)}
        </Label>
      ))}
      {temps.map((t, k) => {
        const pick = phaseDraws[k];
        const top = probBase - aboutContext.odds(t)[pick] * probScale;
        return (
          <motion.circle key={`d${k}`} cx={colX(pick) + bw / 2} cy={top - 40} r={6} className="fill-warn" {...move(k * span + 0.25 * span, k * span + 0.55 * span, 0, 32, k === still, (k + 1) * span - 0.02)} />
        );
      })}
      <Label x={20} y={358}>{`greedy: ${aboutContext.tokens[0]} every time, the largest logit`}</Label>
      {temps.map((t, k) => {
        const odds = aboutContext.odds(t);
        return (
          <motion.g key={`r${k}`} {...on(k * span + 0.5 * span, (k + 1) * span, k === still)}>
            <Label x={20} y={376} className="fill-text">{`a draw at T = ${t}: ${label(phaseDraws[k])}  (${aboutContext.tokens[0]} ${pct(odds[0])}, the ${fmt(aboutContext.others)} others ${pct(odds[SHOWN])})`}</Label>
          </motion.g>
        );
      })}
      <Label x={20} y={394}>{`after “${aboutContext.answer}”`}</Label>
    </Frame>
  );
}

export function SamplingAboutStory() {
  return <StickyStory kicker="the idea" steps={[samplingAbout]} caption={() => samplingAboutCaption} figure={() => <SamplingAboutFigure />} />;
}
