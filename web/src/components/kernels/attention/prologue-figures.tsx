"use client";

import { motion } from "motion/react";
import { Frame, Label } from "../frame";
import { useLoop } from "../loop";

const words = ["The", "cat", "sat", "on", "the", "mat", "and", "purrs", "at", "dawn", "then", "slept"];
const pitch = 66;
const wide = 43.5;

function Cells({ x, y, w, h, count, pitchX, className }: { x: number; y: number; w: number; h: number; count: number; pitchX: number; className: string }) {
  return (
    <>
      {Array.from({ length: count }, (_, i) => (
        <rect key={i} x={x + i * pitchX} y={y} width={w} height={h} rx={2} className={className} />
      ))}
    </>
  );
}

export function PrefillFigure() {
  const { on, move, grow } = useLoop(6);
  const prompt = words.slice(0, 8);
  return (
    <Frame label="Prefill: all eight prompt tokens go through the model in one wide pass, and the keys and values of every position are written to the KV cache at the same time">
      <Label x={20} y={24}>the prompt: 8 tokens, all known up front (illustrative)</Label>
      {prompt.map((w, i) => (
        <g key={w + i}>
          <rect x={20 + i * pitch} y={34} width={60} height={24} rx={3} className="fill-surface stroke-border" />
          <Label x={50 + i * pitch} y={50} anchor="middle" className="fill-text">{w}</Label>
          <motion.rect x={20 + i * pitch} y={34} width={60} height={24} rx={3} className="fill-accent/15 stroke-accent" {...move(0.04, 0.24, 0, 56, false, 0.5)} />
        </g>
      ))}
      <rect x={20} y={104} width={520} height={44} rx={3} className="fill-surface stroke-border" />
      <motion.rect x={20} y={104} width={520} height={44} rx={3} className="fill-accent/20 stroke-accent" {...grow(0.24, 0.4)} />
      <Label x={280} y={131} anchor="middle" className="fill-text">model: one wide pass, every position side by side</Label>
      {prompt.map((_, i) => (
        <motion.line key={i} x1={50 + i * pitch} y1={148} x2={50 + i * pitch} y2={196} className="stroke-accent" strokeWidth={1.5} {...on(0.4, 0.94)} />
      ))}
      <Label x={20} y={192}>KV cache</Label>
      <Label x={20} y={222}>K</Label>
      <Label x={20} y={250}>V</Label>
      {prompt.map((_, i) => (
        <g key={i}>
          <rect x={34 + i * pitch} y={208} width={46} height={20} rx={2} className="fill-surface stroke-border" />
          <rect x={34 + i * pitch} y={234} width={46} height={20} rx={2} className="fill-surface stroke-border" />
          <motion.rect x={34 + i * pitch} y={208} width={46} height={20} rx={2} className="fill-accent/60" {...on(0.5, 0.94)} />
          <motion.rect x={34 + i * pitch} y={234} width={46} height={20} rx={2} className="fill-accent/60" {...on(0.5, 0.94)} />
        </g>
      ))}
      <Label x={20} y={290} className="fill-text">8 positions in, 8 keys and 8 values out, in one pass</Label>
      <Label x={20} y={378}>no position waits for another: the whole prompt is known</Label>
    </Frame>
  );
}

export function DecodeFigure({ percent }: { percent: number }) {
  const { on, grow, reduced } = useLoop(8);
  const steps = 4;
  const prompt = 8;
  const col = (i: number) => 20 + i * wide;
  const at = (s: number) => 0.04 + s * 0.23;
  const bar = 420;
  return (
    <Frame label="Decode: one new token per step, each step reads every cached key and value and adds one more pair">
      <Label x={20} y={24}>8 prompt tokens, then one new token per step (illustrative)</Label>
      {Array.from({ length: prompt }, (_, i) => (
        <g key={i}>
          <rect x={col(i)} y={34} width={39} height={22} rx={3} className="fill-surface stroke-border" />
          <Label x={col(i) + 19.5} y={49} anchor="middle" className="fill-text">{words[i].slice(0, 5)}</Label>
        </g>
      ))}
      <Label x={20} y={84}>model: one position per step</Label>
      <Label x={20} y={150}>KV cache</Label>
      <Cells x={20} y={158} w={39} h={18} count={prompt} pitchX={wide} className="fill-accent/30 stroke-border" />
      <Cells x={20} y={180} w={39} h={18} count={prompt} pitchX={wide} className="fill-accent/30 stroke-border" />
      {Array.from({ length: steps }, (_, s) => {
        const t = at(s);
        const next = prompt + s;
        return (
          <g key={s}>
            <rect x={col(next)} y={34} width={39} height={22} rx={3} className="fill-surface stroke-border" strokeDasharray="3 3" />
            <rect x={col(next)} y={158} width={39} height={18} rx={2} className="fill-surface stroke-border" strokeDasharray="3 3" />
            <rect x={col(next)} y={180} width={39} height={18} rx={2} className="fill-surface stroke-border" strokeDasharray="3 3" />
            <motion.rect x={col(prompt - 1 + s)} y={92} width={39} height={22} rx={3} className="fill-accent/30 stroke-accent" {...on(t, t + 0.12, false)} />
            <motion.rect x={16} y={154} width={wide * next - 1} height={48} rx={3} className="fill-none stroke-accent" strokeWidth={1.5} {...on(t, t + 0.12, false)} />
            <motion.g {...on(t + 0.12, 0.96)}>
              <rect x={col(next)} y={34} width={39} height={22} rx={3} className="fill-accent/15 stroke-accent" />
              <Label x={col(next) + 19.5} y={49} anchor="middle" className="fill-text">{words[next].slice(0, 5)}</Label>
              <rect x={col(next)} y={158} width={39} height={18} rx={2} className="fill-accent/60" />
              <rect x={col(next)} y={180} width={39} height={18} rx={2} className="fill-accent/60" />
            </motion.g>
            <motion.text x={20} y={238} className="fill-accent text-[11px]" {...on(t, t + 0.2, false)}>
              {`step ${s + 1}: reads ${next} cached K/V pairs, adds 1`}
            </motion.text>
          </g>
        );
      })}
      {reduced && <Label x={20} y={238} className="fill-accent">each step reads every cached K/V pair and adds one</Label>}
      <Label x={20} y={284}>bandwidth used at batch 1, of the GPU&apos;s measured peak</Label>
      <rect x={20} y={294} width={bar} height={12} rx={2} className="fill-border" />
      <motion.rect x={20} y={294} width={(bar * percent) / 100} height={12} rx={2} className="fill-accent" {...grow(0.04, 0.2)} />
      <Label x={20 + bar + 10} y={304} className="fill-accent">{`${percent}%`}</Label>
      <Label x={20} y={378}>a step needs the token the last one produced, so steps cannot overlap</Label>
    </Frame>
  );
}
