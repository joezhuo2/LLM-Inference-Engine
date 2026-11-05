"use client";

import { motion } from "motion/react";
import { Frame, Label } from "../frame";
import { useLoop } from "../loop";
import { others, table } from "./figures";

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

export function SideBySideFigure({ prefillMs, stepMs }: { prefillMs: number; stepMs: number }) {
  const { on, grow } = useLoop(7);
  const full = 520;
  const ratio = Math.round(prefillMs / stepMs);
  const boxes = [
    { x: 20, title: "prefill", sub: "all positions, once", from: 0.04, to: 0.3 },
    { x: 205, title: "KV cache", sub: "K and V per position", from: 0.3, to: 0.56 },
    { x: 390, title: "decode", sub: "one token per step", from: 0.56, to: 0.9 },
  ];
  return (
    <Frame label={`Prefill writes the KV cache once and decode reads it every step; without the cache each new token would rerun the whole prefill, which takes about ${ratio} times as long as a decode step`}>
      <Label x={20} y={24}>prefill writes the cache, decode reads it</Label>
      {boxes.map((b) => (
        <g key={b.title}>
          <rect x={b.x} y={36} width={150} height={44} rx={3} className="fill-surface stroke-border" />
          <motion.rect x={b.x} y={36} width={150} height={44} rx={3} className="fill-accent/15 stroke-accent" {...on(b.from, b.to, false)} />
          <Label x={b.x + 75} y={55} anchor="middle" className="fill-text">{b.title}</Label>
          <Label x={b.x + 75} y={71} anchor="middle">{b.sub}</Label>
        </g>
      ))}
      <line x1={172} y1={58} x2={203} y2={58} className="stroke-accent" strokeWidth={1.5} />
      <line x1={357} y1={58} x2={388} y2={58} className="stroke-accent" strokeWidth={1.5} />
      <Label x={187} y={98} anchor="middle">write</Label>
      <Label x={372} y={98} anchor="middle">read</Label>
      <Label x={20} y={132} className="fill-text">the KV cache exists so decode does not recompute the prefill</Label>
      <Label x={20} y={176}>one new token at a context of about 2,000 tokens:</Label>
      <Label x={20} y={206}>without a cache, rerun the prefill</Label>
      <rect x={20} y={214} width={full} height={14} rx={2} className="fill-border" />
      <motion.rect x={20} y={214} width={full} height={14} rx={2} className="fill-baseline" {...grow(0.08, 0.5)} />
      <Label x={540} y={246} anchor="end" className="fill-text">{`${prefillMs.toLocaleString("en-US")} ms`}</Label>
      <Label x={20} y={284}>with the cache, one decode step</Label>
      <rect x={20} y={292} width={full} height={14} rx={2} className="fill-border" />
      <motion.rect x={20} y={292} width={Math.max((full * stepMs) / prefillMs, 2)} height={14} rx={1} className="fill-accent" {...grow(0.5, 0.52)} />
      <Label x={20} y={324} className="fill-accent">{`${stepMs} ms, about ${ratio}x less`}</Label>
      <Label x={20} y={378}>a 2,048-token prefill against a decode step at 1,984 tokens, batch 1</Label>
    </Frame>
  );
}

export function CacheHomeFigure() {
  const { grow, move } = useLoop(8);
  const slice = 0.4 / table.length;
  return (
    <Frame label="A sequence's cache grows a token per step; it is cut into 16-token blocks and placed in free blocks of a pool shared by every request">
      <Label x={20} y={24}>one sequence&apos;s cache, growing a token per step</Label>
      {Array.from({ length: 16 }, (_, n) => (
        <rect
          key={n}
          x={20 + (n % 4) * 132}
          y={170 + Math.floor(n / 4) * 40}
          width={126}
          height={34}
          rx={3}
          className={others.has(n) ? "fill-baseline/25 stroke-border" : "fill-surface stroke-border"}
        />
      ))}
      {table.map((phys, k) => {
        const t = 0.02 + k * slice;
        const dx = 20 + (phys % 4) * 132 + 22 - (20 + k * 88);
        const dy = 170 + Math.floor(phys / 4) * 40 + 4 - 60;
        return (
          <g key={k}>
            <rect x={20 + k * 88} y={60} width={82} height={26} rx={3} className="fill-surface stroke-border" />
            <motion.rect x={20 + k * 88} y={60} width={82} height={26} rx={3} className="fill-accent/60" {...grow(t, t + slice)} />
            <motion.rect x={20 + k * 88} y={60} width={82} height={26} rx={3} className="fill-accent/60 stroke-accent" {...move(0.5 + k * 0.03, 0.78, dx, dy, true, 0.94)} />
          </g>
        );
      })}
      <Label x={20} y={108}>16 tokens per block, 6 blocks = 96 tokens</Label>
      <Label x={20} y={160}>KV pool on the GPU, shared by every request</Label>
      <Label x={20} y={350}>placed in whichever blocks are free, in any order</Label>
      <Label x={20} y={378}>paged attention is how the kernel reads them back</Label>
    </Frame>
  );
}
