"use client";

import { motion } from "motion/react";
import { Frame, Label } from "../frame";
import { useLoop } from "../loop";
import { StickyStory, type StoryStep } from "../sticky-story";

export type OnlineState = { score: number; max: number; sum: number; before: number; rose: boolean };

export const onlineScores = [0.8, 0.2, 1.6, 0.4, 1.1, 2.4, 0.6, 1.5];

export function onlineStates(scores: number[]): OnlineState[] {
  const states: OnlineState[] = [];
  scores.forEach((score, k) => {
    const last = states[k - 1];
    const before = last ? last.max : score;
    const max = Math.max(before, score);
    const kept = last ? last.sum * Math.exp(before - max) : 0;
    states.push({ score, max, sum: kept + Math.exp(score - max), before, rose: k > 0 && score > before });
  });
  return states;
}

export const attentionV2About: StoryStep = {
  id: "about",
  title: "what an online softmax is",
  body: (
    <>
      <p>
        Attention decides how much each earlier token matters to the one being generated. Each query head takes the dot product of its query with the key of every token in the context, 64 values each, and divides it by 8, the square root of 64. That gives one score per token, and a score can be any number.
      </p>
      <p>
        Softmax turns the scores into weights that are positive and add up to 1. Each weight is e to the power of its score minus m, divided by l: m is the largest score, and l is the sum of those powers over every score. Subtracting m does not change the weights, but it keeps every power between 0 and 1, so none can overflow. The weights then say how much of each token&apos;s value vector goes into the head&apos;s output.
      </p>
      <p>
        On a GPU this is awkward: no weight is known until m and l are, and both depend on every score, the last one included. The direct way keeps every score and reads them several times.
      </p>
      <p>
        An online softmax reads the scores once and keeps m and l as running values. A score below m just adds its term to l. A larger score becomes the new m, and l is first multiplied by e to the power of the old m minus the new one, which measures the old terms against the new maximum. After the last score, m and l are exactly right, and the same trick works for the weighted sum of the values.
      </p>
    </>
  ),
};

export const attentionV2AboutCaption = "illustrative: 8 example scores; a real query head has one score for every token of the context";

const states = onlineStates(onlineScores);
const N = onlineScores.length;
const top = Math.max(...onlineScores);
const left = 110;
const pitch = 54;
const bw = 30;
const scoreBase = 170;
const scoreH = 40;
const termBase = 290;
const termH = 70;
const start = (k: number) => 0.04 + k * 0.1;
const ramp = 0.035;
const hold = 0.94;

function keyframes(values: (number | null)[], rest: number) {
  const times = [0];
  const out = [rest];
  values.forEach((v, k) => {
    const value = v ?? rest;
    times.push(start(k), start(k) + ramp);
    out.push(out[out.length - 1], value);
  });
  times.push(hold, 1);
  out.push(out[out.length - 1], rest);
  return { times, values: out };
}

export function AttentionV2AboutFigure() {
  const { on, reduced } = useLoop(18);
  const final = states[N - 1];
  const track = (values: (number | null)[], rest: number, key: "scaleY" | "y" | "opacity", end: number) => {
    if (reduced) return { initial: false as const, animate: { [key]: end } };
    const k = keyframes(values, rest);
    return { initial: { [key]: rest }, animate: { [key]: k.values }, transition: { duration: 18, repeat: Infinity, times: k.times, ease: "easeInOut" as const } };
  };
  const fmt = (v: number) => v.toFixed(2);
  return (
    <Frame label="An online softmax reads the scores one by one, keeping a running maximum m and a running sum l of e to the power of each score minus m; when a larger score arrives, the old terms shrink by e to the power of the old m minus the new one">
      <Label x={20} y={24} className="fill-text">{"softmax: weight = e^(s − m) / l, m = the largest score, l = Σ e^(s − m)"}</Label>
      <Label x={20} y={44}>online: one pass, with m and l kept as running values</Label>
      <Label x={20} y={scoreBase - 40} className="fill-text">
        scores s
      </Label>
      <Label x={20} y={scoreBase - 24}>one per token</Label>
      <Label x={20} y={termBase - 40} className="fill-accent">
        e^(s − m)
      </Label>
      <Label x={20} y={termBase - 24}>the terms of l</Label>
      <line x1={left - 8} y1={scoreBase} x2={left + N * pitch - 16} y2={scoreBase} className="stroke-border" />
      <line x1={left - 8} y1={termBase} x2={left + N * pitch - 16} y2={termBase} className="stroke-border" />
      {onlineScores.map((s, i) => (
        <g key={i}>
          <rect x={left + i * pitch} y={scoreBase - s * scoreH} width={bw} height={s * scoreH} rx={2} className="fill-surface stroke-border" />
          <motion.rect x={left + i * pitch} y={scoreBase - s * scoreH} width={bw} height={s * scoreH} rx={2} className="fill-baseline" {...on(start(i), hold)} />
          <Label x={left + i * pitch + bw / 2} y={scoreBase + 14} anchor="middle">
            {s.toFixed(1)}
          </Label>
          <rect x={left + i * pitch} y={termBase - termH} width={bw} height={termH} rx={2} className="fill-none stroke-border" strokeDasharray="2 3" />
          <motion.rect
            x={left + i * pitch}
            y={termBase - termH}
            width={bw}
            height={termH}
            rx={2}
            className="fill-accent"
            style={{ transformBox: "fill-box", originX: 0.5, originY: 1 }}
            {...track(
              states.map((st, k) => (k < i ? 0 : Math.exp(s - st.max))),
              0,
              "scaleY",
              Math.exp(s - final.max),
            )}
          />
        </g>
      ))}
      <motion.g {...track(states.map((st) => -st.max * scoreH), -onlineScores[0] * scoreH, "y", -final.max * scoreH)}>
        <motion.g {...track(states.map(() => 1), 0, "opacity", 1)}>
          <line x1={left - 8} y1={scoreBase} x2={left + N * pitch - 16} y2={scoreBase} className="stroke-accent" strokeWidth={1.5} strokeDasharray="5 3" />
          <Label x={left + N * pitch - 10} y={scoreBase + 4} className="fill-accent">
            m
          </Label>
        </motion.g>
      </motion.g>
      <Label x={left + N * pitch - 10} y={scoreBase - top * scoreH - 12} anchor="end" className="fill-accent">
        running max
      </Label>
      {states.map((st, k) => {
        const head = `score ${k + 1} = ${st.score.toFixed(1)}`;
        const what = k === 0 ? "the first score: m = s, l = e^0 = 1" : st.rose ? `beats m: old terms × e^(${st.before.toFixed(1)} − ${st.max.toFixed(1)}) = × ${fmt(Math.exp(st.before - st.max))}` : `below m: add e^(${st.score.toFixed(1)} − ${st.max.toFixed(1)}) = ${fmt(Math.exp(st.score - st.max))}`;
        return (
          <motion.g key={k} {...on(start(k), k === N - 1 ? hold : start(k + 1), k === N - 1)}>
            <Label x={20} y={326} className={st.rose ? "fill-warn" : "fill-text"}>{`${head}, ${what}`}</Label>
            <Label x={20} y={346}>{`m = ${st.max.toFixed(1)}, l = ${fmt(st.sum)}, ${k + 1} of ${N} scores seen`}</Label>
          </motion.g>
        );
      })}
      <Label x={20} y={380}>after the last score m and l are exact: each weight is its blue term ÷ l</Label>
    </Frame>
  );
}

export function AttentionV2AboutStory() {
  return <StickyStory kicker="the idea" steps={[attentionV2About]} caption={() => attentionV2AboutCaption} figure={() => <AttentionV2AboutFigure />} />;
}
