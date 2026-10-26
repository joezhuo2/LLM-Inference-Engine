"use client";

import { motion } from "motion/react";
import { Frame, Label } from "../frame";
import { useLoop } from "../loop";

export type Projection = { op: string; label: string; out: number; in: number; per: number; us: number };

export const bytesOf = (p: Projection) => p.out * p.in * 2 * p.per;

const mb = (bytes: number) => `${Math.round(bytes / 1e6).toLocaleString("en-US")} MB`;

export function ProjectionsFigure({ projections, share }: { projections: Projection[]; share: number }) {
  const { grow } = useLoop(9);
  const x = 196;
  const full = 256;
  const maxBytes = Math.max(...projections.map(bytesOf));
  const maxUs = Math.max(...projections.map((p) => p.us));
  const total = projections.reduce((s, p) => s + bytesOf(p), 0);
  return (
    <Frame label="The five projections of a decode step: QKV, output, gate/up and down in each of 22 layers and the LM head once; the weights each one reads per step and its GPU time per step are drawn side by side and have the same shape">
      <Label x={20} y={24}>{"one decode step: 4 projections × 22 layers, then the LM head"}</Label>
      <Label x={x} y={44}>weights read per step</Label>
      <Label x={x} y={60} className="fill-accent">GPU time per step, batch 1</Label>
      {projections.map((p, i) => {
        const y = 78 + i * 52;
        const delay = 0.04 + i * 0.08;
        return (
          <g key={p.op}>
            <Label x={20} y={y + 9} className="fill-text">
              {p.label}
            </Label>
            <Label x={20} y={y + 25}>{`[${p.out.toLocaleString("en-US")} × ${p.in.toLocaleString("en-US")}]${p.per > 1 ? ` × ${p.per}` : ""}`}</Label>
            <rect x={x} y={y} width={full} height={10} rx={5} className="fill-border" />
            <motion.rect x={x} y={y} width={(full * bytesOf(p)) / maxBytes} height={10} rx={5} className="fill-baseline" {...grow(delay, delay + 0.3)} />
            <Label x={x + full + 8} y={y + 9}>{mb(bytesOf(p))}</Label>
            <rect x={x} y={y + 16} width={full} height={10} rx={5} className="fill-border" />
            <motion.rect x={x} y={y + 16} width={(full * p.us) / maxUs} height={10} rx={5} className="fill-accent" {...grow(delay + 0.04, delay + 0.34)} />
            <Label x={x + full + 8} y={y + 25} className="fill-accent">{`${Math.round(p.us).toLocaleString("en-US")} µs`}</Label>
          </g>
        );
      })}
      <Label x={20} y={350} className="fill-text">{`${(total / 1e9).toFixed(2)} GB of weights per token; these five own ${share.toFixed(1)}% of the GPU time`}</Label>
      <Label x={20} y={370}>the two bars have the same shape: each projection takes as long</Label>
      <Label x={20} y={386}>as its weights take to come out of DRAM</Label>
    </Frame>
  );
}

const out = 3;
const inn = 4;
const tone = ["fill-accent", "fill-baseline", "fill-accent/40"];

export function TransposeFigure() {
  const { on } = useLoop(10);
  const cw = 34;
  const ch = 22;
  const win = (r: number) => [0.04 + r * 0.29, 0.04 + (r + 1) * 0.29] as const;
  const cell = (key: string, x: number, y: number, r: number, c: number) => (
    <g key={key}>
      <rect x={x} y={y} width={cw - 4} height={ch - 4} rx={2} className={tone[r]} opacity={0.35} />
      <motion.rect x={x} y={y} width={cw - 4} height={ch - 4} rx={2} className={tone[r]} {...on(...win(r), r === 0)} />
      <text x={x + (cw - 4) / 2} y={y + 13} textAnchor="middle" className="fill-text text-[9px]">{`${r},${c}`}</text>
    </g>
  );
  return (
    <Frame label="W is stored row-major as [out, in]; the same bytes read column-major with a leading dimension of in are Wᵀ, so asking cuBLAS for the transposed product reads the model's weights as they are">
      <Label x={20} y={24}>{`W in memory: [out = ${out}, in = ${inn}], one row after the other`}</Label>
      {Array.from({ length: out * inn }, (_, k) => cell(`m${k}`, 20 + k * (cw + 4), 36, Math.floor(k / inn), k % inn))}
      <Label x={20} y={84}>address →</Label>
      <Label x={20} y={118} className="fill-text">row-major [out, in]</Label>
      <Label x={20} y={134}>what the model stores</Label>
      {Array.from({ length: out * inn }, (_, k) => cell(`a${k}`, 20 + (k % inn) * cw, 148 + Math.floor(k / inn) * ch, Math.floor(k / inn), k % inn))}
      <Label x={300} y={118} className="fill-text">column-major, ld = in</Label>
      <Label x={300} y={134}>what cuBLAS sees: [in, out] = Wᵀ</Label>
      {Array.from({ length: out * inn }, (_, k) => cell(`b${k}`, 300 + Math.floor(k / inn) * cw, 148 + (k % inn) * ch, Math.floor(k / inn), k % inn))}
      <Label x={20} y={256}>row r of W is column r of what cuBLAS sees: the same 4 addresses</Label>
      <Label x={20} y={296} className="fill-text">cublasGemmEx(OP_T, OP_N, m = out, n = B, k = in,</Label>
      <Label x={112} y={312} className="fill-text">W, lda = in, x, ldb = in, y, ldc = out)</Label>
      <motion.g {...on(0.3)}>
        <Label x={20} y={340}>{"C = op(A) · B = (Wᵀ)ᵀ · Xᵀ = W · Xᵀ = Yᵀ, written column-major"}</Label>
        <Label x={20} y={356} className="fill-accent">
          Yᵀ column-major is Y row-major: no copy, no transpose kernel
        </Label>
      </motion.g>
    </Frame>
  );
}

const drawnRows = 12;

export function GemvFigure({ weightBytes, peakGbs, tpotUs }: { weightBytes: number; peakGbs: number; tpotUs: number }) {
  const { on } = useLoop(10);
  const rowY = (r: number) => 80 + r * 15;
  const slice = 0.8 / drawnRows;
  const floorMs = weightBytes / (peakGbs * 1e9) / 1e-3;
  return (
    <Frame label="At batch 1 each projection is a matrix-vector product: every weight row streams from DRAM once and meets the one input row in a single dot product, 2 FLOPs for every 2 bytes">
      <Label x={20} y={24}>{"batch 1: x is one row, so y = x · Wᵀ is a matrix-vector product"}</Label>
      <Label x={20} y={50}>x</Label>
      {Array.from({ length: 16 }, (_, k) => (
        <rect key={k} x={40 + k * 20} y={40} width={17} height={14} rx={2} className="fill-accent/70" />
      ))}
      <Label x={370} y={46}>2,048 values, 4 KB:</Label>
      <Label x={370} y={60}>stays in cache</Label>
      <Label x={20} y={72}>W, streamed from DRAM, {drawnRows} of 2,560 QKV rows drawn</Label>
      {Array.from({ length: drawnRows }, (_, r) => {
        const start = 0.04 + r * slice;
        return (
          <g key={r}>
            <rect x={40} y={rowY(r)} width={316} height={10} rx={2} className="fill-surface stroke-border" strokeWidth={0.5} />
            <motion.rect x={40} y={rowY(r)} width={316} height={10} rx={2} className="fill-baseline" {...on(start, start + 1.5 * slice, false)} />
            <rect x={400} y={rowY(r)} width={18} height={10} rx={2} className="fill-surface stroke-border" strokeWidth={0.5} />
            <motion.rect x={400} y={rowY(r)} width={18} height={10} rx={2} className="fill-accent" {...on(start + slice, 0.94, true)} />
          </g>
        );
      })}
      <Label x={428} y={rowY(0) + 9}>y: one value per row</Label>
      <Label x={428} y={rowY(1) + 9}>a dot product of</Label>
      <Label x={428} y={rowY(2) + 9}>2,048 pairs</Label>
      <Label x={20} y={276} className="fill-text">each weight: 2 bytes read once, 1 multiply-add = 2 FLOPs</Label>
      <Label x={20} y={292} className="fill-accent">1 FLOP per byte: the speed of DRAM is the speed of the GEMV</Label>
      <motion.g {...on(0.45)}>
        <Label x={20} y={326}>{`all five projections: ${(weightBytes / 1e9).toFixed(2)} GB per token`}</Label>
        <Label x={20} y={342}>{`at the measured ${peakGbs} GB/s that is ${floorMs.toFixed(2)} ms per token at best;`}</Label>
        <Label x={20} y={358}>{`the engine with CUDA Graphs takes ${(tpotUs / 1000).toFixed(2)} ms`}</Label>
        <Label x={20} y={382}>cuBLAS sees B = 1 and runs a gemv kernel, not a GEMM</Label>
      </motion.g>
    </Frame>
  );
}

export type RoofRow = { label: string; gbs: number; pct: number; warn?: boolean };

export function RoofBarsFigure({ rows, peakGbs }: { rows: RoofRow[]; peakGbs: number }) {
  const { grow, on } = useLoop(9);
  const x = 130;
  const full = 320;
  const max = Math.max(peakGbs, ...rows.map((r) => r.gbs)) * 1.06;
  const px = x + (full * peakGbs) / max;
  const top = 66;
  const step = 38;
  return (
    <Frame label="Achieved DRAM bandwidth of each projection at batch 1 against the measured 323.6 GB/s: the five GEMVs run at 91 to 102% of it; attention, at 2%, is limited by something else">
      <Label x={20} y={24}>{"batch 1: bytes each op moves ÷ its time, per decode step"}</Label>
      {rows.map((r, i) => {
        const y = top + i * step;
        const cls = r.warn ? "fill-warn" : "fill-accent";
        return (
          <g key={r.label}>
            <Label x={20} y={y + 10} className={r.warn ? "fill-warn" : "fill-text"}>
              {r.label}
            </Label>
            <rect x={x} y={y} width={full} height={12} rx={6} className="fill-border" />
            <motion.rect x={x} y={y} width={Math.max(4, (full * r.gbs) / max)} height={12} rx={6} className={cls} {...grow(0.04 + i * 0.05, 0.3 + i * 0.05)} />
            <Label x={x + full + 10} y={y + 10} className={r.warn ? "fill-warn" : "fill-accent"}>{`${r.gbs} GB/s`}</Label>
            <Label x={x + full + 10} y={y + 24}>{`${r.pct}% of roof`}</Label>
          </g>
        );
      })}
      <line x1={px} x2={px} y1={top - 8} y2={top + rows.length * step - 18} className="stroke-text" strokeDasharray="3 3" />
      <Label x={px} y={top - 14} anchor="middle" className="fill-text">{`measured peak ${peakGbs}`}</Label>
      <motion.g {...on(0.45)}>
        <Label x={20} y={330}>the roof is a measured copy kernel, not a hard limit: 102% reads as</Label>
        <Label x={20} y={346}>&quot;at the roof&quot;. A faster GEMV would have to read fewer bytes:</Label>
        <Label x={20} y={362}>smaller weights (8-bit, 4-bit) or more rows per read</Label>
      </motion.g>
    </Frame>
  );
}

export function ReuseFigure({ us1, us16, tf1, tf16 }: { us1: number; us16: number; tf1: number; tf16: number }) {
  const { on, grow } = useLoop(10);
  const rows = 16;
  return (
    <Frame label="With 16 rows each weight row is read once and used by 16 dot products: the projections take the same time per step while doing 16 times the FLOPs">
      <Label x={20} y={24}>{"16 rows: one weight row read once, used 16 times"}</Label>
      <rect x={40} y={40} width={316} height={10} rx={2} className="fill-baseline" />
      <Label x={366} y={49}>one row of W from DRAM</Label>
      {Array.from({ length: rows }, (_, r) => {
        const y = 62 + r * 9;
        return (
          <g key={r}>
            <rect x={40} y={y} width={316} height={5} rx={1.5} className="fill-accent/30" />
            <motion.rect x={400} y={y - 1} width={18} height={7} rx={1.5} className="fill-accent" {...on(0.05 + r * 0.025, 0.94, true)} />
          </g>
        );
      })}
      <Label x={428} y={74}>16 outputs</Label>
      <Label x={20} y={226}>16 tokens&apos; rows</Label>
      <Label x={20} y={242} className="fill-accent">2 FLOPs × 16 per 2 bytes: 16 FLOP/byte, still left of the 107 ridge</Label>
      {[
        { title: "five projections, µs per decode step", a: us1, b: us16, fmt: (v: number) => `${Math.round(v).toLocaleString("en-US")} µs`, y: 274 },
        { title: "gate/up, TFLOP/s", a: tf1, b: tf16, fmt: (v: number) => `${v.toFixed(2)}`, y: 334 },
      ].map((g, gi) => {
        const max = Math.max(g.a, g.b);
        return (
          <g key={g.title}>
            <Label x={20} y={g.y}>{g.title}</Label>
            {[
              { label: "1 row", v: g.a },
              { label: "16 rows", v: g.b, accent: true },
            ].map((r, i) => {
              const y = g.y + 8 + i * 20;
              const start = 0.3 + gi * 0.2;
              return (
                <g key={r.label}>
                  <Label x={20} y={y + 10} className={r.accent ? "fill-accent" : "fill-muted"}>
                    {r.label}
                  </Label>
                  <rect x={90} y={y} width={340} height={12} rx={6} className="fill-border" />
                  <motion.rect x={90} y={y} width={Math.max(4, (340 * r.v) / max)} height={12} rx={6} className={r.accent ? "fill-accent" : "fill-baseline"} {...grow(start, start + 0.2)} />
                  <Label x={440} y={y + 10} className={r.accent ? "fill-accent" : "fill-muted"}>
                    {g.fmt(r.v)}
                  </Label>
                </g>
              );
            })}
          </g>
        );
      })}
    </Frame>
  );
}
