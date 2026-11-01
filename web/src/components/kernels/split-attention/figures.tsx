"use client";

import { motion } from "motion/react";
import { Frame, Label } from "../frame";
import { useLoop } from "../loop";

export type Occupancy = { splits: number; blocks: number; waves: number; warps: number; us: number };

const sms = 26;
const smW = 38;
const smH = 62;

function slot(i: number, perSm: number) {
  const sm = i % sms;
  const k = Math.floor(i / sms) % perSm;
  const x = 20 + (sm % 13) * 40 + 4 + (k % 2) * 16;
  const y = Math.floor(sm / 13) * (smH + 6) + 8 + Math.floor(k / 2) * 18;
  return { x, y, second: i >= sms * perSm };
}

function SmBoxes({ y, perSm }: { y: number; perSm: number }) {
  return (
    <g transform={`translate(0 ${y})`}>
      {Array.from({ length: sms }, (_, s) => (
        <g key={s}>
          <rect x={20 + (s % 13) * 40} y={Math.floor(s / 13) * (smH + 6)} width={smW} height={smH} rx={3} className="fill-surface stroke-border" />
          {Array.from({ length: perSm }, (_, k) => {
            const p = slot(k * sms + s, perSm);
            return <rect key={k} x={p.x} y={p.y} width={14} height={14} rx={2} className="fill-none stroke-border" strokeDasharray="2 2" />;
          })}
        </g>
      ))}
    </g>
  );
}

export function OccupancyFigure({ perSm, v2 }: { perSm: number; v2: Occupancy }) {
  const { on, grow } = useLoop(6);
  const capacity = sms * perSm;
  return (
    <Frame label={`One row launches ${v2.blocks} blocks, one per query head, into ${capacity} slots: ${v2.waves} waves, ${v2.warps}% of warps active`}>
      <Label x={20} y={22} className="fill-text">{`1 row × 32 heads = ${v2.blocks} blocks · ${sms} SMs × ${perSm} slots = ${capacity}`}</Label>
      <SmBoxes y={34} perSm={perSm} />
      <g transform="translate(0 34)">
        {Array.from({ length: v2.blocks }, (_, i) => {
          const p = slot(i, perSm);
          return <motion.rect key={i} x={p.x} y={p.y} width={14} height={14} rx={2} className="fill-accent" {...on(0.03 + i * 0.006)} />;
        })}
      </g>
      <Label x={20} y={196}>{`${v2.waves} waves · warps active ${v2.warps}% · ${capacity - v2.blocks} slots never used`}</Label>
      <Label x={20} y={232}>each block walks all 2,048 tokens of its head alone</Label>
      {Array.from({ length: 8 }, (_, i) => (
        <g key={i}>
          <rect x={20} y={246 + i * 13} width={520} height={8} rx={2} className="fill-border" />
          <motion.rect x={20} y={246 + i * 13} width={520} height={8} rx={2} className="fill-accent/70" {...grow(0.2, 0.9)} />
        </g>
      ))}
      <Label x={20} y={372}>{`8 of the 32 heads shown · ${v2.us} µs in Nsight Compute`}</Label>
    </Frame>
  );
}

export function WaitingFigure({ warps, sm, dram }: { warps: number; sm: number; dram: number }) {
  const { grow, on } = useLoop(6);
  const gauges = [
    { label: "warps active", v: warps },
    { label: "SM throughput", v: sm },
    { label: "DRAM throughput", v: dram },
  ];
  const chain = [0, 1, 2, 3];
  return (
    <Frame label={`Nsight Compute on v2 at one row: warps active ${warps}%, SM throughput ${sm}%, DRAM ${dram}%`}>
      <Label x={20} y={22} className="fill-text">
        v2, one row at 2,048 tokens (Nsight Compute)
      </Label>
      {gauges.map((g, i) => (
        <g key={g.label}>
          <Label x={20} y={58 + i * 40}>{g.label}</Label>
          <rect x={160} y={47 + i * 40} width={320} height={14} rx={3} className="fill-border" />
          <motion.rect x={160} y={47 + i * 40} width={(320 * g.v) / 100} height={14} rx={3} className="fill-accent" {...grow(0.05 + i * 0.08, 0.3 + i * 0.08)} />
          <Label x={490} y={58 + i * 40} className="fill-text">{`${g.v}%`}</Label>
        </g>
      ))}
      <Label x={20} y={206} className="fill-text">
        one warp&apos;s loop over its keys
      </Label>
      {chain.map((c) => {
        const x = 20 + c * 132;
        return (
          <motion.g key={c} {...on(0.35 + c * 0.13)}>
            <rect x={x} y={220} width={20} height={22} rx={2} className="fill-baseline" />
            <rect x={x + 22} y={220} width={84} height={22} rx={2} className="fill-none stroke-border" strokeDasharray="3 3" />
            <rect x={x + 108} y={220} width={16} height={22} rx={2} className="fill-accent" />
          </motion.g>
        );
      })}
      <rect x={20} y={262} width={12} height={10} rx={2} className="fill-baseline" />
      <Label x={38} y={271}>block table and key load issued</Label>
      <rect x={250} y={262} width={12} height={10} rx={2} className="fill-none stroke-border" strokeDasharray="3 3" />
      <Label x={268} y={271}>waiting for memory</Label>
      <rect x={410} y={262} width={12} height={10} rx={2} className="fill-accent" />
      <Label x={428} y={271}>math</Label>
      <Label x={20} y={312}>the GPU hides that wait by switching to other warps,</Label>
      <Label x={20} y={330}>and with 32 blocks there are too few to switch to</Label>
      <Label x={20} y={372}>bar widths are measured; the loop below them is a sketch</Label>
    </Frame>
  );
}

export function CutFigure({ splits, contextBlocks }: { splits: number; contextBlocks: number }) {
  const { on, move } = useLoop(7);
  const per = Math.ceil(contextBlocks / splits);
  const cw = 520 / contextBlocks;
  const laneH = 22;
  return (
    <Frame label={`A ${contextBlocks}-block context cut into ${splits} ranges of ${per} KV blocks, each handed to its own thread block`}>
      <Label x={20} y={22} className="fill-text">{`one head's context: ${contextBlocks} KV blocks of 16 tokens`}</Label>
      {Array.from({ length: contextBlocks }, (_, b) => (
        <rect key={b} x={20 + b * cw} y={34} width={cw - 0.8} height={18} className="fill-baseline/60" />
      ))}
      {Array.from({ length: splits - 1 }, (_, s) => (
        <motion.line key={s} x1={20 + (s + 1) * per * cw} y1={28} x2={20 + (s + 1) * per * cw} y2={58} className="stroke-accent" strokeWidth={2} {...on(0.08 + s * 0.03)} />
      ))}
      <motion.g {...on(0.3)}>
        <Label x={20} y={78}>{`blocks per split = ⌈${contextBlocks} / ${splits}⌉ = ${per}, so a cut never lands inside a block`}</Label>
      </motion.g>
      {Array.from({ length: splits }, (_, s) => {
        const y0 = 34;
        const y1 = 100 + s * (laneH + 6);
        return (
          <motion.g key={s} {...move(0.38 + s * 0.02, 0.55 + s * 0.02, -s * per * cw, y1 - y0, true, 0.94)}>
            <rect x={20 + s * per * cw} y={y0} width={per * cw - 0.8} height={laneH} rx={2} className="fill-accent/60" />
          </motion.g>
        );
      })}
      {Array.from({ length: splits }, (_, s) => (
        <motion.g key={s} {...on(0.6 + s * 0.02)}>
          <Label x={20 + per * cw + 12} y={115 + s * (laneH + 6)}>{`block (split ${s}, head, row): tokens ${s * per * 16} to ${(s + 1) * per * 16 - 1}`}</Label>
        </motion.g>
      ))}
      <Label x={20} y={372}>every range runs the same online-softmax loop as v2, on its own</Label>
    </Frame>
  );
}

export function WaveFigure({ perSm, rows }: { perSm: number; rows: Occupancy[] }) {
  const { on } = useLoop(12);
  const capacity = sms * perSm;
  const span = 1 / rows.length;
  const slowest = Math.max(...rows.map((r) => r.us));
  return (
    <Frame label={`Doubling the split count doubles the grid; up to ${capacity} blocks fit in one wave, beyond that a second, partly empty wave runs`}>
      <SmBoxes y={34} perSm={perSm} />
      {rows.map((r, j) => {
        const t0 = j * span + 0.01;
        const t1 = (j + 1) * span - 0.01;
        return (
          <motion.g key={r.splits} {...on(t0, t1, j === rows.length - 1)}>
            <Label x={20} y={22} className="fill-text">{`${r.splits} split${r.splits > 1 ? "s" : ""} × 32 heads = ${r.blocks} blocks · ${r.waves} waves`}</Label>
            <g transform="translate(0 34)">
              {Array.from({ length: Math.min(r.blocks, capacity) }, (_, i) => {
                const p = slot(i, perSm);
                return <rect key={i} x={p.x} y={p.y} width={14} height={14} rx={2} className="fill-accent" />;
              })}
            </g>
            {r.blocks > capacity && (
              <>
                {Array.from({ length: r.blocks - capacity }, (_, i) => (
                  <rect key={i} x={20 + i * 5.2} y={186} width={3.6} height={9} rx={1} className="fill-none stroke-accent" strokeWidth={0.8} />
                ))}
                <Label x={20} y={212} className="fill-accent">{`${r.blocks - capacity} more blocks queue for a second, mostly empty wave`}</Label>
              </>
            )}
          </motion.g>
        );
      })}
      {rows.map((r, i) => (
        <g key={r.splits}>
          <Label x={20} y={238 + i * 30}>{`${r.splits} split${r.splits > 1 ? "s" : ""}`}</Label>
          <rect x={90} y={228 + i * 30} width={300} height={12} rx={2} className="fill-border" />
          <rect x={90} y={228 + i * 30} width={(300 * r.us) / slowest} height={12} rx={2} className="fill-baseline" />
          <motion.rect x={90} y={228 + i * 30} width={(300 * r.us) / slowest} height={12} rx={2} className="fill-accent" {...on(i * span + 0.01, (i + 1) * span - 0.01, i === rows.length - 1)} />
          <Label x={400} y={238 + i * 30}>{`${r.us.toFixed(1)} µs · ${r.warps}% warps`}</Label>
        </g>
      ))}
      <Label x={20} y={372}>partial kernel only, Nsight Compute, one row at 2,048 tokens</Label>
    </Frame>
  );
}

export function ScratchFigure({ splits, bytes }: { splits: number; bytes: number }) {
  const { on } = useLoop(6);
  const w = 520 / splits;
  return (
    <Frame label={`Each split writes its running maximum, its sum and 64 weighted values to a float scratch buffer of ${bytes} bytes`}>
      <Label x={20} y={22} className="fill-text">
        scratch: float [rows, heads, splits, 64 + 2]
      </Label>
      <Label x={20} y={44}>one head, one row: a record per split</Label>
      {Array.from({ length: splits }, (_, s) => {
        const x = 20 + s * w;
        return (
          <motion.g key={s} {...on(0.08 + ((s * 5) % splits) * 0.05)}>
            <rect x={x} y={58} width={w - 4} height={120} rx={3} className="fill-accent/10 stroke-accent" />
            <Label x={x + 6} y={76} className="fill-text">{`split ${s}`}</Label>
            <rect x={x + 6} y={86} width={w - 16} height={14} rx={2} className="fill-accent" />
            <Label x={x + 9} y={97} className="fill-bg">m</Label>
            <rect x={x + 6} y={104} width={w - 16} height={14} rx={2} className="fill-accent/60" />
            <Label x={x + 9} y={115} className="fill-bg">l</Label>
            {Array.from({ length: 8 }, (_, k) => (
              <rect key={k} x={x + 6} y={124 + k * 6} width={w - 16} height={4} rx={1} className="fill-baseline" />
            ))}
          </motion.g>
        );
      })}
      <Label x={20} y={206}>m: the largest score in the range</Label>
      <Label x={20} y={224}>l: the sum of exp(score − m) over the range</Label>
      <Label x={20} y={242}>acc: 64 values weighted by exp(score − m), not yet divided by l</Label>
      <motion.g {...on(0.55)}>
        <Label x={20} y={290} className="fill-text">{`1 row × 32 heads × ${splits} splits × 66 floats = ${bytes.toLocaleString("en-US")} bytes`}</Label>
        <Label x={20} y={308}>the runner allocates it once: 256 records, the most the rule ever asks for</Label>
      </motion.g>
      <Label x={20} y={372}>splits finish in any order; the merge reads them in a fixed one</Label>
    </Frame>
  );
}

const partials = [
  { m: 2.1, l: 41.7 },
  { m: 3.4, l: 18.2 },
  { m: 1.7, l: 63.0 },
  { m: 2.9, l: 25.4 },
];

export function MergeFigure() {
  const { on } = useLoop(7);
  const M = Math.max(...partials.map((p) => p.m));
  return (
    <Frame label="The merge rescales every split's partial result to the largest running maximum and adds them in split order">
      <Label x={20} y={24}>partial results in the scratch buffer (example values)</Label>
      {partials.map((p, i) => {
        const x = 20 + i * 132;
        const top = p.m === M;
        return (
          <g key={i}>
            <motion.g {...on(0.02 + i * 0.05)}>
              <rect x={x} y={36} width={124} height={78} rx={3} className={top ? "fill-accent/10 stroke-accent" : "fill-surface stroke-border"} />
              <Label x={x + 10} y={56} className="fill-text">{`split ${i}`}</Label>
              <Label x={x + 10} y={76}>{`m = ${p.m.toFixed(1)}`}</Label>
              <Label x={x + 10} y={92}>{`l = ${p.l.toFixed(1)}`}</Label>
              <Label x={x + 10} y={108}>acc[64]</Label>
            </motion.g>
            <motion.g {...on(0.42 + i * 0.05)}>
              <Label x={x + 62} y={148} anchor="middle" className="fill-accent">
                {`× ${Math.exp(p.m - M).toFixed(2)}`}
              </Label>
              <line x1={x + 62} y1={158} x2={280} y2={250} className="stroke-accent" />
            </motion.g>
          </g>
        );
      })}
      <motion.g {...on(0.28)}>
        <Label x={20} y={186} className="fill-text">
          {`M = max of the m values = ${M.toFixed(1)}`}
        </Label>
        <Label x={20} y={204}>each split is scaled by exp(m − M)</Label>
      </motion.g>
      <motion.g {...on(0.66)}>
        <circle cx={280} cy={262} r={12} className="fill-accent" />
        <Label x={300} y={266} className="fill-text">
          added in split order
        </Label>
      </motion.g>
      <motion.g {...on(0.76)}>
        <Label x={20} y={312} className="fill-text">
          out = Σ exp(mᵢ − M) accᵢ / Σ exp(mᵢ − M) lᵢ
        </Label>
        <Label x={20} y={334}>the same answer as one pass over the whole context,</Label>
        <Label x={20} y={352}>and the same bits every run, because the order is fixed</Label>
      </motion.g>
    </Frame>
  );
}

export type GraphKey = { bucket: number; splits: number[] };

export function GraphsFigure({ bars, keys }: { bars: { label: string; ms: number; accent?: boolean }[]; keys: GraphKey[] }) {
  const { on, grow } = useLoop(7);
  const max = Math.max(...bars.map((b) => b.ms));
  const total = keys.reduce((a, k) => a + k.splits.length, 0);
  return (
    <Frame label={`Graphs captured on an empty batch always replayed v2; one graph per batch size and split count makes ${total} graphs`}>
      <Label x={20} y={22} className="fill-text">
        ms per token, 1,984-token prompt, batch 1
      </Label>
      {bars.map((b, i) => (
        <g key={b.label}>
          <Label x={20} y={54 + i * 30} className={b.accent ? "fill-accent" : "fill-muted"}>
            {b.label}
          </Label>
          <rect x={200} y={44 + i * 30} width={250} height={12} rx={2} className="fill-border" />
          <motion.rect x={200} y={44 + i * 30} width={(250 * b.ms) / max} height={12} rx={2} className={b.accent ? "fill-accent" : "fill-baseline"} {...grow(0.03 + i * 0.05, 0.25 + i * 0.05)} />
          <Label x={460} y={54 + i * 30}>{`${b.ms.toFixed(2)} ms`}</Label>
        </g>
      ))}
      <motion.g {...on(0.35)}>
        <Label x={20} y={196}>before: each batch size captured once, on a dummy batch with no context,</Label>
        <Label x={20} y={214}>so the split count was 1 and every replay ran v2</Label>
      </motion.g>
      <motion.g {...on(0.55)}>
        <Label x={20} y={246} className="fill-text">{`after: one graph per (batch size, split count), ${total} in all`}</Label>
        {keys.map((k, i) => (
          <g key={k.bucket}>
            <Label x={20} y={272 + i * 18}>{`batch ${k.bucket}`}</Label>
            {k.splits.map((s, j) => (
              <g key={s}>
                <rect x={100 + j * 46} y={262 + i * 18} width={40} height={13} rx={2} className="fill-accent/15 stroke-accent" />
                <Label x={120 + j * 46} y={272 + i * 18} anchor="middle" className="fill-text">{`${s}`}</Label>
              </g>
            ))}
          </g>
        ))}
      </motion.g>
    </Frame>
  );
}
