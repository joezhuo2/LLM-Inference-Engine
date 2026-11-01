"use client";

import { motion } from "motion/react";
import { Frame, Label } from "../frame";
import { useLoop } from "../loop";

const scores = [0.42, 0.7, 0.25, 0.9, 0.55, 0.33, 0.78, 0.5, 0.62, 0.2];
const probs = (() => {
  const e = scores.map((s) => Math.exp(4 * s));
  const sum = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / sum);
})();

export function OverviewFigure() {
  const { on, grow } = useLoop(6);
  const step = 0.05;
  const maxP = Math.max(...probs);
  return (
    <Frame label="A new query scores every cached key, softmax turns the scores into weights, and the weighted values become the output">
      <Label x={20} y={28}>q, the new token</Label>
      {Array.from({ length: 8 }, (_, i) => (
        <rect key={i} x={150 + i * 18} y={16} width={16} height={16} rx={2} className="fill-accent" />
      ))}
      <Label x={20} y={70}>K, cached keys</Label>
      <Label x={330} y={50}>q · k</Label>
      <Label x={440} y={50}>softmax</Label>
      {scores.map((s, r) => {
        const y = 60 + r * 24;
        const t = 0.04 + r * step;
        return (
          <g key={r}>
            {Array.from({ length: 8 }, (_, i) => (
              <rect key={i} x={150 + i * 18} y={y} width={16} height={16} rx={2} className="fill-surface stroke-border" />
            ))}
            <motion.rect x={146} y={y - 3} width={150} height={22} rx={3} className="fill-none stroke-accent" {...on(t, t + step, false)} />
            <motion.rect x={330} y={y + 3} width={s * 90} height={10} rx={2} className="fill-baseline" {...grow(t, t + step)} />
            <motion.rect x={440} y={y + 3} width={Math.round((probs[r] / maxP) * 10000) / 100} height={10} rx={2} className="fill-accent" {...grow(0.6, 0.7)} />
          </g>
        );
      })}
      <Label x={20} y={330}>out = Σ p · v</Label>
      {Array.from({ length: 8 }, (_, i) => (
        <motion.rect key={i} x={150 + i * 18} y={318} width={16} height={16} rx={2} className="fill-accent" {...on(0.76 + i * 0.015)} />
      ))}
      <Label x={20} y={378}>one query per sequence per step; the context grows by one token every step</Label>
    </Frame>
  );
}

const table = [7, 2, 11, 4, 9, 0];
const others = new Set([1, 5, 6, 12, 14]);

export function BlocksFigure() {
  const { on } = useLoop(6);
  const slice = 0.94 / table.length;
  return (
    <Frame label="A sequence's logical blocks map through its block table to scattered physical blocks in a shared KV pool">
      <Label x={20} y={24}>this sequence: 96 tokens, 6 blocks of 16</Label>
      {table.map((phys, k) => {
        const t = k * slice + 0.01;
        return (
          <g key={k}>
            <rect x={20 + k * 88} y={34} width={82} height={26} rx={3} className="fill-surface stroke-border" />
            <Label x={61 + k * 88} y={51} anchor="middle" className="fill-text">
              {`${k * 16}–${k * 16 + 15}`}
            </Label>
            <rect x={20 + k * 88} y={96} width={82} height={24} rx={3} className="fill-bg stroke-border" />
            <Label x={61 + k * 88} y={112} anchor="middle" className="fill-text">
              {phys}
            </Label>
            <motion.g {...on(t, t + slice, false)}>
              <rect x={20 + k * 88} y={34} width={82} height={26} rx={3} className="fill-accent/10 stroke-accent" />
              <rect x={20 + k * 88} y={96} width={82} height={24} rx={3} className="fill-accent/10 stroke-accent" />
            </motion.g>
            <motion.line
              x1={61 + k * 88}
              y1={120}
              x2={20 + (phys % 4) * 132 + 63}
              y2={170 + Math.floor(phys / 4) * 40}
              className="stroke-accent"
              strokeWidth={1.5}
              {...on(t, t + slice, false)}
            />
          </g>
        );
      })}
      <Label x={20} y={88}>block table</Label>
      <Label x={20} y={160}>KV pool on the GPU, shared by every request</Label>
      {Array.from({ length: 16 }, (_, n) => {
        const mine = table.includes(n);
        return (
          <g key={n}>
            <rect
              x={20 + (n % 4) * 132}
              y={170 + Math.floor(n / 4) * 40}
              width={126}
              height={34}
              rx={3}
              className={mine ? "fill-accent/15 stroke-accent" : others.has(n) ? "fill-baseline/25 stroke-border" : "fill-surface stroke-border"}
            />
            <Label x={30 + (n % 4) * 132} y={191 + Math.floor(n / 4) * 40} className={mine ? "fill-text" : "fill-muted"}>
              {`block ${n}`}
            </Label>
          </g>
        );
      })}
      {[
        { x: 20, label: "this sequence", cls: "fill-accent/15 stroke-accent" },
        { x: 170, label: "other requests", cls: "fill-baseline/25 stroke-border" },
        { x: 330, label: "free", cls: "fill-surface stroke-border" },
      ].map((l) => (
        <g key={l.label}>
          <rect x={l.x} y={333} width={12} height={12} rx={2} className={l.cls} />
          <Label x={l.x + 18} y={343}>{l.label}</Label>
        </g>
      ))}
      <Label x={20} y={370} className="fill-text">
        key of token pos = pool[table[pos / 16]], row pos % 16
      </Label>
      <Label x={20} y={390}>blocks are handed out as the sequence grows, nothing reserved up front</Label>
    </Frame>
  );
}

const sms = 26;
const slotsPerSm = 8;

function SmGrid({ lit, y }: { lit: number; y: number }) {
  const { pulse } = useLoop(5);
  const total = sms * slotsPerSm;
  const litSet = new Set(Array.from({ length: Math.round(lit * total) }, (_, i) => Math.floor((i * total) / Math.round(lit * total))));
  return (
    <g>
      {Array.from({ length: sms }, (_, s) => {
        const x = 20 + (s % 13) * 40;
        const yy = y + Math.floor(s / 13) * 58;
        return (
          <g key={s}>
            <rect x={x} y={yy} width={36} height={52} rx={3} className="fill-surface stroke-border" />
            {Array.from({ length: slotsPerSm }, (_, k) => {
              const n = s * slotsPerSm + k;
              const cxv = x + 11 + (k % 2) * 14;
              const cyv = yy + 10 + Math.floor(k / 2) * 11;
              return litSet.has(n) ? (
                <motion.circle key={k} cx={cxv} cy={cyv} r={3.5} className="fill-accent" {...pulse((n % 7) * 0.2)} />
              ) : (
                <circle key={k} cx={cxv} cy={cyv} r={3.5} className="fill-border" />
              );
            })}
          </g>
        );
      })}
    </g>
  );
}

function TimeBars({ v2, split }: { v2: number; split?: number }) {
  const { grow } = useLoop(5);
  const full = 420;
  return (
    <g>
      <Label x={20} y={268}>v2</Label>
      <rect x={100} y={258} width={full} height={12} rx={2} className="fill-border" />
      <motion.rect x={100} y={258} width={full} height={12} rx={2} className="fill-baseline" {...grow(0.02, 0.86)} />
      <Label x={520} y={288} anchor="end">{`${v2} µs`}</Label>
      {split !== undefined && (
        <>
          <Label x={20} y={318} className="fill-accent">
            8 splits
          </Label>
          <rect x={100} y={308} width={full} height={12} rx={2} className="fill-border" />
          <motion.rect x={100} y={308} width={(full * split) / v2} height={12} rx={2} className="fill-accent" {...grow(0.02, 0.02 + (0.84 * split) / v2)} />
          <Label x={100 + (full * split) / v2} y={338} anchor="end" className="fill-accent">
            {`${split} µs, merge included`}
          </Label>
        </>
      )}
    </g>
  );
}

export function SplitFigure({ v2, split }: { v2: number; split: number }) {
  const { on } = useLoop(5);
  const per = 4;
  return (
    <Frame label="The split kernel cuts the context into 8 ranges processed by separate blocks at the same time, filling the GPU">
      <Label x={20} y={24}>8 splits × 32 heads = 256 blocks, each on its own range</Label>
      {Array.from({ length: 32 }, (_, i) => {
        const s = Math.floor(i / per);
        const k = i % per;
        return (
          <g key={i}>
            <rect x={20 + i * 16.5 + s * 1.5} y={34} width={14.5} height={20} rx={2} className="fill-surface stroke-border" />
            <motion.rect x={20 + i * 16.5 + s * 1.5} y={34} width={14.5} height={20} rx={2} className="fill-accent/60" {...on(0.02 + k * 0.08, 0.94)} />
          </g>
        );
      })}
      <SmGrid lit={0.85} y={70} />
      <Label x={20} y={222}>all 8 ranges at once, then a small merge kernel</Label>
      <TimeBars v2={v2} split={split} />
      <Label x={20} y={372}>one row at 2,048 tokens; the busy SMs are an illustration</Label>
    </Frame>
  );
}
