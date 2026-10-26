"use client";

import { motion } from "motion/react";
import { Frame, Label } from "../frame";
import { useLoop } from "../loop";

const sums = [0.6, -0.9, 0.3, 1.2, -0.4, 0.8, -1.1, 0.5, 0.2, -0.7, 1.0, -0.3, 0.9, -0.6, 0.4, -1.0];

export function JobFigure() {
  const { on, grow } = useLoop(7);
  const x0 = 130;
  const cw = 25;
  const n = sums.length;
  const slice = 0.3 / n;
  const sq = Math.max(...sums.map((s) => s * s));
  return (
    <Frame label="The layer output is added to the residual stream, the sum is kept as the next residual, and its RMS norm scaled by a weight vector feeds the next matrix multiply">
      <Label x={20} y={24}>one row, one token: 2,048 values (one cell = 128)</Label>
      {[
        { y: 40, label: "x, layer output" },
        { y: 70, label: "r, residual" },
      ].map((row) => (
        <g key={row.y}>
          <Label x={20} y={row.y + 13}>{row.label}</Label>
          {sums.map((_, i) => (
            <g key={i}>
              <rect x={x0 + i * cw} y={row.y} width={cw - 3} height={18} rx={2} className="fill-surface stroke-border" />
              <motion.rect x={x0 + i * cw} y={row.y} width={cw - 3} height={18} rx={2} className="fill-baseline/60" {...on(0.03 + i * slice, 0.03 + (i + 2) * slice, false)} />
            </g>
          ))}
        </g>
      ))}
      <Label x={20} y={123} className="fill-text">
        sum = x + r
      </Label>
      {sums.map((_, i) => (
        <g key={i}>
          <rect x={x0 + i * cw} y={110} width={cw - 3} height={18} rx={2} className="fill-surface stroke-border" />
          <motion.rect x={x0 + i * cw} y={110} width={cw - 3} height={18} rx={2} className="fill-accent/60" {...on(0.03 + i * slice)} />
        </g>
      ))}
      <motion.g {...on(0.34)}>
        <Label x={x0} y={146}>
          kept: it is the next layer&apos;s residual
        </Label>
      </motion.g>
      <Label x={20} y={196}>sum²</Label>
      {sums.map((s, i) => (
        <motion.rect key={i} x={x0 + i * cw + 4} y={210 - (s * s * 40) / sq} width={cw - 11} height={(s * s * 40) / sq} rx={1} className="fill-baseline" {...grow(0.36 + i * 0.01, 0.4 + i * 0.01, "y")} />
      ))}
      <line x1={x0} y1={210} x2={x0 + n * cw - 3} y2={210} className="stroke-border" />
      <motion.g {...on(0.56)}>
        <Label x={x0} y={236} className="fill-text">
          rstd = 1 / √(mean of sum² + ε), one number per row
        </Label>
      </motion.g>
      <Label x={20} y={273} className="fill-text">
        norm
      </Label>
      {sums.map((_, i) => (
        <g key={i}>
          <rect x={x0 + i * cw} y={260} width={cw - 3} height={18} rx={2} className="fill-surface stroke-border" />
          <motion.rect x={x0 + i * cw} y={260} width={cw - 3} height={18} rx={2} className="fill-accent" {...on(0.62 + i * slice)} />
        </g>
      ))}
      <motion.g {...on(0.66)}>
        <Label x={x0} y={298}>
          norm = w · sum · rstd → the next GEMM
        </Label>
      </motion.g>
      <Label x={20} y={350}>twice per layer: after attention, and after the MLP into the next</Label>
      <Label x={20} y={370}>layer&apos;s input norm; 22 layers, 43 fused launches per decode step</Label>
    </Frame>
  );
}

function Packet({ x, y, text, accent, motionProps }: { x: number; y: number; text: string; accent?: boolean; motionProps: object }) {
  return (
    <motion.g {...motionProps}>
      <rect x={x} y={y} width={44} height={16} rx={3} className={accent ? "fill-accent" : "fill-baseline"} />
      <text x={x + 22} y={y + 12} textAnchor="middle" className="fill-bg text-[10px]">
        {text}
      </text>
    </motion.g>
  );
}

export function RoundTripFigure({ naiveKb, fusedKb }: { naiveKb: number; fusedKb: number }) {
  const { move, grow, on } = useLoop(8);
  const memX = 20;
  const memW = 90;
  const kx = 240;
  const kw = 130;
  const barX = 400;
  const barW = 140;
  const into = kx - memX - memW - 50;
  return (
    <Frame label="The naive pair writes the sum to memory and reads it back in a second kernel; the fused kernel reads its inputs once and writes both outputs">
      <Label x={20} y={24} className="fill-text">
        two kernels
      </Label>
      <rect x={memX} y={36} width={memW} height={130} rx={3} className="fill-surface stroke-border" />
      <Label x={memX + memW / 2} y={106} anchor="middle">
        DRAM
      </Label>
      <rect x={kx} y={44} width={kw} height={34} rx={3} className="fill-surface stroke-border" />
      <Label x={kx + kw / 2} y={65} anchor="middle" className="fill-text">
        add
      </Label>
      <rect x={kx} y={120} width={kw} height={34} rx={3} className="fill-surface stroke-border" />
      <Label x={kx + kw / 2} y={141} anchor="middle" className="fill-text">
        rmsnorm
      </Label>
      <Packet x={memX + memW + 6} y={44} text="x, r" motionProps={move(0.02, 0.16, into, 0)} />
      <Packet x={kx - 50} y={60} text="sum" accent motionProps={move(0.18, 0.32, -into, 0)} />
      <Packet x={memX + memW + 6} y={120} text="sum" accent motionProps={move(0.34, 0.48, into, 0)} />
      <Packet x={memX + memW + 6} y={138} text="w" motionProps={move(0.36, 0.5, into, 0)} />
      <Packet x={kx - 50} y={138} text="norm" motionProps={move(0.52, 0.66, -into, 0)} />
      <motion.g {...on(0.34, 0.5)}>
        <Label x={kx} y={108} className="fill-warn">
          the sum makes a round trip
        </Label>
      </motion.g>

      <Label x={20} y={214} className="fill-text">
        one kernel
      </Label>
      <rect x={memX} y={226} width={memW} height={100} rx={3} className="fill-surface stroke-border" />
      <Label x={memX + memW / 2} y={281} anchor="middle">
        DRAM
      </Label>
      <rect x={kx} y={250} width={kw} height={52} rx={3} className="fill-accent/10 stroke-accent" />
      <Label x={kx + kw / 2} y={280} anchor="middle" className="fill-text">
        add_rmsnorm
      </Label>
      <Packet x={memX + memW + 6} y={256} text="x, r" motionProps={move(0.02, 0.16, into, 0)} />
      <Packet x={memX + memW + 6} y={278} text="w" motionProps={move(0.04, 0.18, into, 0)} />
      <Packet x={kx - 50} y={256} text="sum" accent motionProps={move(0.3, 0.44, -into, 0)} />
      <Packet x={kx - 50} y={278} text="norm" motionProps={move(0.32, 0.46, -into, 0)} />
      <Label x={kx} y={322}>the sum stays on the chip</Label>

      <Label x={barX} y={24}>DRAM bytes, 64 tokens</Label>
      <rect x={barX} y={36} width={barW} height={12} rx={2} className="fill-border" />
      <motion.rect x={barX} y={36} width={barW} height={12} rx={2} className="fill-baseline" {...grow(0.02, 0.66)} />
      <Label x={barX + barW} y={66} anchor="end">{`${naiveKb} KB, 2 launches`}</Label>
      <rect x={barX} y={226} width={barW} height={12} rx={2} className="fill-border" />
      <motion.rect x={barX} y={226} width={(barW * fusedKb) / naiveKb} height={12} rx={2} className="fill-accent" {...grow(0.02, 0.46)} />
      <Label x={barX + barW} y={256} anchor="end" className="fill-accent">{`${fusedKb} KB, 1 launch`}</Label>
      <Label x={20} y={372}>bytes measured with Nsight Compute; the moving boxes are an illustration</Label>
    </Frame>
  );
}

const phases = [
  { text: "load x, r: 8 scalar loads of 2 bytes", w: 0.22 },
  { text: "add, write the sum, square it", w: 0.12 },
  { text: "reduce: shuffles, shared memory", w: 0.14, barrier: true },
  { text: "thread 0 writes rstd to shared", w: 0.06, barrier: true },
  { text: "read every sum back from memory", w: 0.2 },
  { text: "scale by w, write the norm", w: 0.14 },
].map((p, i, all) => ({ ...p, start: all.slice(0, i).reduce((a, q) => a + q.w, 0) }));

function MeasuredBars({ rows, y, max }: { rows: { label: string; us: number; accent?: boolean }[]; y: number; max: number }) {
  const { grow } = useLoop(7);
  const x = 130;
  const full = 340;
  return (
    <g>
      {rows.map((r, i) => {
        const yy = y + i * 30;
        const w = (full * r.us) / max;
        return (
          <g key={r.label}>
            <Label x={20} y={yy + 10} className={r.accent ? "fill-accent" : "fill-muted"}>
              {r.label}
            </Label>
            <rect x={x} y={yy} width={full} height={12} rx={2} className="fill-border" />
            <motion.rect x={x} y={yy} width={w} height={12} rx={2} className={r.accent ? "fill-accent" : "fill-baseline"} {...grow(0.02, 0.02 + (0.8 * r.us) / max)} />
            <Label x={x + full + 8} y={yy + 10} className={r.accent ? "fill-accent" : "fill-muted"}>{`${r.us.toFixed(1)} µs`}</Label>
          </g>
        );
      })}
    </g>
  );
}

export function FirstFusedFigure({ pair, first, flat }: { pair: number; first: number; flat: string }) {
  const { grow, on } = useLoop(7);
  const gx = 300;
  const gw = 240;
  return (
    <Frame label="The first fused kernel runs six phases one after another with two barriers, and takes about the same time as the two naive kernels">
      <Label x={20} y={24}>one block of 256 threads per row, in order</Label>
      {phases.map((p, i) => {
        const y = 44 + i * 28;
        const end = p.start + p.w;
        const t0 = 0.02 + p.start * 0.82;
        const t1 = 0.02 + end * 0.82;
        return (
          <g key={p.text}>
            <Label x={20} y={y + 10}>{p.text}</Label>
            <motion.text x={20} y={y + 10} className="fill-accent text-[11px]" {...on(t0, t1, false)}>
              {p.text}
            </motion.text>
            <motion.rect x={gx + p.start * gw} y={y} width={p.w * gw} height={12} rx={2} className={i === 4 ? "fill-warn/70" : "fill-accent/70"} {...grow(t0, t1)} />
            {p.barrier && <line x1={gx + end * gw} y1={36} x2={gx + end * gw} y2={210} className="stroke-warn" strokeDasharray="3 3" />}
          </g>
        );
      })}
      <Label x={gx + 0.48 * gw} y={226} anchor="middle" className="fill-warn">
        2 barriers
      </Label>
      <MeasuredBars
        y={258}
        max={pair}
        rows={[
          { label: "naive pair", us: pair },
          { label: "first fused", us: first, accent: true },
        ]}
      />
      <Label x={20} y={340}>{`one token, µs per launch; at 1, 16 and 64 tokens: ${flat}`}</Label>
      <Label x={20} y={362}>the time does not grow with the rows: latency bound, not bandwidth</Label>
    </Frame>
  );
}

const stages = [
  { text: "x, r", type: "bf16" },
  { text: "x + r", type: "float" },
  { text: "sum = round(x + r), written out", type: "bf16", round: true },
  { text: "sum², from the rounded sum", type: "float", fix: true },
  { text: "Σ sum² over the row, then rstd", type: "float" },
  { text: "round(sum · rstd)", type: "bf16", round: true, fix: true },
  { text: "norm = round(w · that)", type: "bf16", round: true },
];

export function RoundingFigure() {
  const { on } = useLoop(8);
  const slice = 0.84 / stages.length;
  return (
    <Frame label="The order of operations and roundings that makes the fused kernel's sum bit-identical to the naive add and its norm match the naive RMSNorm">
      <Label x={20} y={24}>every value, in order; ● marks a rounding to BF16</Label>
      {stages.map((s, i) => {
        const y = 52 + i * 34;
        const t = 0.02 + i * slice;
        return (
          <g key={s.text}>
            <rect x={20} y={y - 16} width={520} height={26} rx={3} className="fill-surface stroke-border" />
            <motion.rect x={20} y={y - 16} width={520} height={26} rx={3} className="fill-accent/10 stroke-accent" {...on(t, t + slice, false)} />
            {s.round && <circle cx={36} cy={y - 3} r={4} className="fill-warn" />}
            <Label x={52} y={y + 1} className="fill-text">
              {s.text}
            </Label>
            <Label x={528} y={y + 1} anchor="end">
              {s.type}
            </Label>
            {s.fix && (
              <motion.g {...on(t, 0.94)}>
                <Label x={380} y={y + 1} className="fill-warn">
                  review fix
                </Label>
              </motion.g>
            )}
          </g>
        );
      })}
      <motion.g {...on(0.12 + 3 * slice)}>
        <Label x={20} y={312} className="fill-warn">
          fix: the first draft squared the float sum, before rounding
        </Label>
      </motion.g>
      <motion.g {...on(0.12 + 5 * slice)}>
        <Label x={20} y={332} className="fill-warn">
          fix: it also rounded once where the naive kernel rounds twice
        </Label>
      </motion.g>
      <Label x={20} y={366}>sum: the same bits as the naive add, always</Label>
      <Label x={20} y={386}>norm: equal up to the order the squares are added</Label>
    </Frame>
  );
}

export function VectorFigure() {
  const { on } = useLoop(7);
  const cw = 6.5;
  const group = 8 * cw + 6;
  const rounds = 8;
  const slice = 0.48 / rounds;
  return (
    <Frame label="The first kernel's threads make eight 2-byte loads each, one per round; the tuned kernel's threads load 8 neighbouring values in one 16-byte load and keep them in registers">
      <Label x={20} y={24}>first fused: thread t reads t, t + 256, t + 512 …</Label>
      {Array.from({ length: rounds }, (_, k) => (
        <g key={k}>
          {Array.from({ length: 8 }, (_, t) => (
            <g key={t}>
              <rect x={20 + k * group + t * cw} y={36} width={cw - 1} height={20} rx={1} className="fill-surface stroke-border" strokeWidth={0.5} />
              <motion.rect x={20 + k * group + t * cw} y={36} width={cw - 1} height={20} rx={1} className="fill-baseline" {...on(0.03 + k * slice, 0.94)} />
            </g>
          ))}
          <Label x={20 + k * group + 4 * cw} y={72} anchor="middle">
            {String(k)}
          </Label>
        </g>
      ))}
      <Label x={20} y={92}>each box: threads 0–7 in rounds 0 to 7; later the sums are read back</Label>

      <Label x={20} y={136}>tuned: thread t reads 8t … 8t + 7 in one 16-byte load</Label>
      {Array.from({ length: 8 }, (_, t) => (
        <g key={t}>
          {Array.from({ length: 8 }, (_, e) => (
            <g key={e}>
              <rect x={20 + t * group + e * cw} y={148} width={cw - 1} height={20} rx={1} className="fill-surface stroke-border" strokeWidth={0.5} />
              <motion.rect x={20 + t * group + e * cw} y={148} width={cw - 1} height={20} rx={1} className="fill-accent" {...on(0.03, 0.94)} />
            </g>
          ))}
          <Label x={20 + t * group + 4 * cw} y={184} anchor="middle">
            {`t${t}`}
          </Label>
        </g>
      ))}
      <Label x={20} y={204}>2,048 values = 256 threads × 8: one load each</Label>
      <Label x={20} y={238} className="fill-text">
        registers of thread 0
      </Label>
      {Array.from({ length: 8 }, (_, e) => (
        <g key={e}>
          <rect x={190 + e * 20} y={226} width={17} height={17} rx={2} className="fill-surface stroke-border" />
          <motion.rect x={190 + e * 20} y={226} width={17} height={17} rx={2} className="fill-accent/70" {...on(0.1 + e * 0.01, 0.94)} />
        </g>
      ))}
      <motion.g {...on(0.56)}>
        <Label x={360} y={238} className="fill-accent">
          pass 2 reads these
        </Label>
      </motion.g>
      <Label x={20} y={290}>global loads per thread at hidden 2048 (x, r, sum, w)</Label>
      {[
        { y: 302, label: "first", n: 32, text: "32 × 2 bytes", cls: "fill-baseline" },
        { y: 326, label: "tuned", n: 3, text: "3 × 16 bytes", cls: "fill-accent" },
      ].map((b) => (
        <g key={b.label}>
          <Label x={20} y={b.y + 10} className={b.label === "tuned" ? "fill-accent" : "fill-muted"}>
            {b.label}
          </Label>
          {Array.from({ length: b.n }, (_, i) => (
            <motion.rect key={i} x={80 + i * 10} y={b.y} width={8} height={12} rx={1} className={b.cls} {...on(0.6 + i * 0.008, 0.94)} />
          ))}
          <Label x={b.n === 3 ? 120 : 410} y={b.y + 10}>
            {b.text}
          </Label>
        </g>
      ))}
      <Label x={20} y={372}>tuned path: hidden a multiple of 8, at most 8,192, 16-byte aligned</Label>
    </Frame>
  );
}

const lanesStart = [3, 1, 4, 1, 5, 9, 2, 6];

const shuffleLevels = (() => {
  const levels = [lanesStart];
  for (const m of [4, 2, 1]) {
    const prev = levels[levels.length - 1];
    levels.push(prev.map((v, j) => v + prev[j ^ m]));
  }
  return levels;
})();

export function ShuffleFigure() {
  const { on } = useLoop(8);
  const lx = (j: number) => 70 + j * 60;
  const ly = (k: number) => 52 + k * 66;
  const masks = [4, 2, 1];
  return (
    <Frame label="A butterfly of xor shuffles: after three exchanges every one of 8 lanes holds the total, illustrating the 32-lane warp's five">
      {shuffleLevels.map((vals, k) => (
        <g key={k}>
          {k > 0 && (
            <motion.g {...on(0.04 + (k - 1) * 0.2, 0.94)}>
              <Label x={20} y={ly(k) - 30} className="fill-accent">{`xor ${masks[k - 1]}`}</Label>
              {vals.map((_, j) => (
                <g key={j}>
                  <line x1={lx(j)} y1={ly(k - 1) + 14} x2={lx(j)} y2={ly(k) - 14} className="stroke-border" />
                  <line x1={lx(j ^ masks[k - 1])} y1={ly(k - 1) + 14} x2={lx(j)} y2={ly(k) - 14} className="stroke-accent" />
                </g>
              ))}
            </motion.g>
          )}
          {vals.map((v, j) => {
            const last = k === shuffleLevels.length - 1;
            const node = (
              <g>
                <circle cx={lx(j)} cy={ly(k)} r={14} className={last ? "fill-accent" : k === 0 ? "fill-surface stroke-border" : "fill-accent/15 stroke-accent"} />
                <text x={lx(j)} y={ly(k) + 4} textAnchor="middle" className={last ? "fill-bg text-[11px]" : "fill-text text-[11px]"}>
                  {v}
                </text>
              </g>
            );
            return k === 0 ? (
              <g key={j}>{node}</g>
            ) : (
              <motion.g key={j} {...on(0.1 + (k - 1) * 0.2, 0.94)}>
                {node}
              </motion.g>
            );
          })}
        </g>
      ))}
      <Label x={20} y={28}>lane</Label>
      {lanesStart.map((_, j) => (
        <Label key={j} x={lx(j)} y={28} anchor="middle">
          {String(j)}
        </Label>
      ))}
      <motion.g {...on(0.66)}>
        <Label x={20} y={300} className="fill-text">
          a real warp: 32 lanes, xor 16, 8, 4, 2, 1
        </Label>
        <Label x={20} y={320}>8 warp totals → shared memory → one barrier → xor again</Label>
      </motion.g>
      <motion.g {...on(0.78)}>
        <Label x={20} y={350} className="fill-accent">
          every thread now holds the total and computes rstd itself:
        </Label>
        <Label x={20} y={370}>no thread 0, no shared rstd, no second barrier</Label>
      </motion.g>
    </Frame>
  );
}
