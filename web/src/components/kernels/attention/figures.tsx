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

export function V1Figure() {
  const { on, grow } = useLoop(6);
  const cells = 32;
  const cw = 16.5;
  const scan = 0.42;
  return (
    <Frame label="v1 writes one score per token into shared memory, which fills up at 2,048 tokens, then makes two more passes">
      <Label x={20} y={24}>one thread block = one row × one query head</Label>
      {Array.from({ length: cells }, (_, i) => (
        <g key={i}>
          <rect x={20 + i * cw} y={34} width={cw - 2} height={20} rx={2} className="fill-surface stroke-border" />
          <motion.rect x={20 + i * cw} y={34} width={cw - 2} height={20} rx={2} className="fill-accent/50" {...on(0.02 + (i * scan) / cells, 0.94)} />
        </g>
      ))}
      <Label x={20} y={72}>context, 2,048 tokens (one cell = 64)</Label>
      <Label x={20} y={104}>shared memory: one float per token</Label>
      <rect x={18} y={112} width={cells * cw + 2} height={30} rx={3} className="fill-none stroke-warn" />
      {Array.from({ length: cells }, (_, i) => (
        <motion.rect key={i} x={21 + i * cw} y={116} width={cw - 3} height={22} rx={1} className="fill-warn/60" {...grow(0.02 + (i * scan) / cells, 0.02 + ((i + 1) * scan) / cells, "y")} />
      ))}
      <Label x={20 + cells * cw} y={160} anchor="end" className="fill-warn">
        full: 2,048 scores = 8 KB
      </Label>
      {[
        { y: 200, text: "1  every score into shared memory", from: 0.02, to: 0.44 },
        { y: 228, text: "2  max, then exp and sum over all of them", from: 0.46, to: 0.66 },
        { y: 256, text: "3  probabilities × values, 64 threads busy", from: 0.68, to: 0.92 },
      ].map((p) => (
        <g key={p.y}>
          <Label x={20} y={p.y}>{p.text}</Label>
          <motion.text x={20} y={p.y} className="fill-accent text-[11px]" {...on(p.from, p.to, false)}>
            {p.text}
          </motion.text>
        </g>
      ))}
      <Label x={20} y={312}>8,192 tokens</Label>
      {Array.from({ length: 6 }, (_, i) => (
        <rect key={i} x={110 + i * 22} y={300} width={18} height={16} rx={2} className="fill-warn/30 stroke-warn" />
      ))}
      <Label x={250} y={312} className="fill-warn">
        does not fit: v1 rejects it
      </Label>
      <Label x={20} y={372}>three passes over the context, and a cap set by shared memory</Label>
    </Frame>
  );
}

const chunkScores = [0.3, 0.45, 0.2, 0.5, 0.35, 0.4, 0.25, 0.38, 0.55, 0.7, 0.4, 0.62, 0.5, 0.3, 0.66, 0.45, 0.35, 0.5, 0.6, 0.42, 0.3, 0.55, 0.48, 0.36, 0.82, 0.6, 0.7, 0.5, 0.65, 0.4, 0.58, 0.45];

const chunkMaxes = Array.from({ length: 4 }, (_, c) => ({
  prev: c === 0 ? 0 : Math.max(...chunkScores.slice(0, c * 8)),
  now: Math.max(...chunkScores.slice(0, (c + 1) * 8)),
}));

export function OnlineSoftmaxFigure() {
  const { on } = useLoop(7);
  const chunks = 4;
  const per = 8;
  const cw = 16.5;
  const slice = 0.92 / chunks;
  const base = 170;
  const h = 110;
  return (
    <Frame label="v2 reads the context one 256-token chunk at a time, keeping a running maximum and rescaling its sums when the maximum rises">
      <Label x={20} y={24}>chunks of 256 tokens, one after another (one bar = 32 tokens)</Label>
      {chunkScores.map((s, i) => {
        const c = Math.floor(i / per);
        return (
          <motion.rect
            key={i}
            x={20 + i * cw}
            y={base - s * h}
            width={cw - 3}
            height={s * h}
            rx={1}
            className="fill-baseline"
            {...on(0.02 + c * slice, 0.94)}
          />
        );
      })}
      <line x1={18} y1={base} x2={20 + 32 * cw} y2={base} className="stroke-border" />
      {chunkMaxes.map((m, c) => {
        const t = 0.02 + c * slice;
        const rose = c > 0 && m.now > m.prev;
        return (
          <g key={c}>
            <motion.rect x={18 + c * per * cw} y={40} width={per * cw} height={base - 36} rx={3} className="fill-accent/10 stroke-accent" {...on(t, t + slice, false)} />
            <motion.line
              x1={18}
              x2={20 + (c + 1) * per * cw}
              y1={base - m.now * h}
              y2={base - m.now * h}
              className="stroke-accent"
              strokeDasharray="4 3"
              {...on(t + 0.03, c === chunks - 1 ? 0.94 : t + slice, c === chunks - 1)}
            />
            {rose && (
              <motion.text x={20} y={204} className="fill-warn text-[11px]" {...on(t + 0.03, t + slice - 0.02, false)}>
                max rose: scale the old sum and acc by exp(m_old − m_new)
              </motion.text>
            )}
          </g>
        );
      })}
      <Label x={20 + 32 * cw} y={36} anchor="end" className="fill-accent">
        running max m
      </Label>
      <rect x={20} y={232} width={250} height={84} rx={3} className="fill-none stroke-accent" />
      <Label x={32} y={254} className="fill-text">
        kept between chunks
      </Label>
      <Label x={32} y={276}>m, running max</Label>
      <Label x={32} y={294}>l, running sum of exp</Label>
      <Label x={32} y={312}>acc[64], running Σ p · v</Label>
      <rect x={290} y={232} width={250} height={84} rx={3} className="fill-none stroke-border" />
      <Label x={302} y={254} className="fill-text">
        shared memory
      </Label>
      <Label x={302} y={276}>one chunk of 256 scores</Label>
      <Label x={302} y={294}>the same at 128 or 8,192 tokens</Label>
      <Label x={20} y={360}>one pass over the keys and values; nothing grows with the context</Label>
    </Frame>
  );
}

export function LanesFigure() {
  const { on } = useLoop(6);
  const cw = 26;
  const x0 = 40;
  const y0 = 64;
  const rowH = 18;
  const steps = 8;
  const slice = 0.56 / steps;
  const levels = [
    { y: 270, n: 8, t: 0.62 },
    { y: 302, n: 4, t: 0.7 },
    { y: 334, n: 2, t: 0.78 },
    { y: 366, n: 1, t: 0.86 },
  ];
  const cx = (n: number, j: number) => x0 + ((j + 0.5) * (8 * cw)) / n;
  return (
    <Frame label="Eight lanes share one key: each step they read eight neighbouring dimensions together, then three shuffles add their partial dot products">
      <Label x={x0} y={24}>one key, 64 dims, read by a group of 8 lanes</Label>
      {Array.from({ length: 8 }, (_, j) => (
        <Label key={j} x={x0 + j * cw + cw / 2} y={52} anchor="middle">
          {String(j)}
        </Label>
      ))}
      <Label x={x0 + 8 * cw + 10} y={52}>lane</Label>
      {Array.from({ length: steps }, (_, s) => (
        <g key={s}>
          <Label x={x0 - 8} y={y0 + s * rowH + 12} anchor="end">
            {String(s * 8)}
          </Label>
          {Array.from({ length: 8 }, (_, j) => (
            <g key={j}>
              <rect x={x0 + j * cw} y={y0 + s * rowH} width={cw - 3} height={rowH - 3} rx={2} className="fill-surface stroke-border" />
              <motion.rect x={x0 + j * cw} y={y0 + s * rowH} width={cw - 3} height={rowH - 3} rx={2} className="fill-accent/60" {...on(0.02 + s * slice, 0.94)} />
            </g>
          ))}
          <motion.text x={x0 + 8 * cw + 10} y={y0 + s * rowH + 12} className="fill-accent text-[11px]" {...on(0.02 + s * slice, 0.02 + (s + 1) * slice, false)}>
            {`dims ${s * 8}–${s * 8 + 7}: 16 bytes in one go`}
          </motion.text>
        </g>
      ))}
      <Label x={x0} y={252}>partial dot products, added by shuffles (xor 4, 2, 1)</Label>
      {levels.map((lv, li) => (
        <g key={lv.y}>
          {Array.from({ length: lv.n }, (_, j) => (
            <g key={j}>
              {li > 0 &&
                [0, 1].map((k) => (
                  <motion.line
                    key={k}
                    x1={cx(levels[li - 1].n, j * 2 + k)}
                    y1={levels[li - 1].y}
                    x2={cx(lv.n, j)}
                    y2={lv.y}
                    className="stroke-accent"
                    {...on(lv.t, 0.94)}
                  />
                ))}
              <motion.circle cx={cx(lv.n, j)} cy={lv.y} r={li === 3 ? 6 : 4} className={li === 3 ? "fill-accent" : "fill-baseline"} {...on(lv.t, 0.94)} />
            </g>
          ))}
        </g>
      ))}
      <Label x={x0 + 4 * cw + 12} y={370} className="fill-text">
        one score
      </Label>
      <Label x={300} y={296} className="fill-text">
        a warp = 4 groups = 4 keys at once
      </Label>
      <Label x={300} y={314}>a 256-thread block: 32 keys at once</Label>
      <Label x={300} y={346}>v1: one thread walks all 64</Label>
      <Label x={300} y={364}>dims of its key on its own</Label>
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
