"use client";

import { motion } from "motion/react";
import { batchRows, budget, fits, gb, memFmt, parts, tokenFactors } from "@/content/kernels/memory-budget";
import { Frame, Label } from "../frame";
import { useLoop } from "../loop";

const fmt = memFmt;
const X = 20;
const W = 520;
const before = (values: number[]) => values.map((_, i) => values.slice(0, i).reduce((a, b) => a + b, 0));
const cls = { weights: "fill-text/70", pool: "fill-accent", spare: "fill-accent/25", rest: "fill-muted/50" } as Record<string, string>;

export function TankFigure() {
  const { on, grow } = useLoop(14);
  const px = W / budget.vram;
  const starts = before(parts.map((p) => p.bytes * px));
  return (
    <Frame label={`The GPU's ${gb(budget.vram)}: ${parts.map((p) => `${p.label} ${gb(p.bytes)}`).join(", ")}`}>
      <Label x={X} y={22} className="fill-text">{`the GPU's memory, ${gb(budget.vram)}, to scale`}</Label>
      <rect x={X} y={34} width={W} height={34} rx={3} className="fill-border" />
      <motion.g {...grow(0.05, 0.3)}>
        {parts.map((p, i) => (
          <rect key={p.id} x={X + starts[i]} y={34} width={Math.max(1, p.bytes * px - 1)} height={34} rx={2} className={cls[p.id]} />
        ))}
      </motion.g>
      {parts.map((p, i) => (
        <motion.g key={p.id} {...on(0.3 + i * 0.08)}>
          <rect x={X} y={98 + i * 62} width={10} height={10} rx={1} className={cls[p.id]} />
          <Label x={X + 18} y={107 + i * 62} className="fill-text">
            {p.label}
          </Label>
          <Label x={X + W} y={107 + i * 62} anchor="end" className="fill-accent">{`${gb(p.bytes)} · ${fmt((p.bytes / budget.vram) * 100)}%`}</Label>
          <Label x={X + 18} y={123 + i * 62}>{p.does}</Label>
          <Label x={X + 18} y={138 + i * 62}>{p.grows}</Label>
        </motion.g>
      ))}
      <Label x={X} y={384}>{`pool as allocated in the recorded run of ${budget.poolDate} (measured); the last two parts are calculated`}</Label>
    </Frame>
  );
}

export function BytesFigure() {
  const { on } = useLoop(14);
  const chipW = 96;
  const ladder = [
    { label: "one cached token", bytes: budget.tokenBytes, text: `${fmt(budget.tokenBytes)} bytes` },
    { label: `one block of ${budget.blockTokens} tokens`, bytes: budget.blockBytes, text: `${fmt(budget.blockBytes / 1e3)} KB` },
    { label: `one full context, ${fmt(budget.fullContext)} tokens`, bytes: budget.fullContext * budget.tokenBytes, text: `${fmt(budget.fullMb, 1)} MB` },
    { label: `the pool, ${fmt(budget.poolTokens)} tokens`, bytes: budget.pool, text: gb(budget.pool) },
    { label: "the weights, for scale", bytes: budget.weights, text: gb(budget.weights) },
  ];
  return (
    <Frame label={`One cached token is ${tokenFactors.map((f) => `${f.value} ${f.label}`).join(" × ")} = ${fmt(budget.tokenBytes)} bytes; a block is ${fmt(budget.blockBytes / 1e3)} KB, a full context ${fmt(budget.fullMb, 1)} MB, the pool ${gb(budget.pool)}`}>
      <Label x={X} y={22} className="fill-text">
        what one token leaves in the cache
      </Label>
      {tokenFactors.map((f, i) => (
        <motion.g key={i} {...on(0.05 + i * 0.07)}>
          <rect x={X + i * (chipW + 10)} y={36} width={chipW} height={58} rx={3} className="fill-accent/20" />
          <text x={X + i * (chipW + 10) + chipW / 2} y={62} textAnchor="middle" className="fill-text text-[18px]">
            {f.value}
          </text>
          <text x={X + i * (chipW + 10) + chipW / 2} y={82} textAnchor="middle" className="fill-muted text-[10px]">
            {f.label}
          </text>
          {i > 0 && (
            <text x={X + i * (chipW + 10) - 5} y={68} textAnchor="middle" className="fill-muted text-[12px]">
              ×
            </text>
          )}
        </motion.g>
      ))}
      <motion.g {...on(0.45)}>
        <Label x={X} y={122} className="fill-accent">{`= ${fmt(budget.tokenBytes)} bytes per token, whatever the token says`}</Label>
      </motion.g>
      {ladder.map((r, i) => (
        <motion.g key={r.label} {...on(0.52 + i * 0.06)}>
          <Label x={X} y={160 + i * 42}>{r.label}</Label>
          <Label x={X + W} y={160 + i * 42} anchor="end" className={i === 4 ? "fill-text" : "fill-accent"}>
            {r.text}
          </Label>
          <rect x={X} y={167 + i * 42} width={W} height={7} rx={2} className="fill-border" />
          <rect x={X} y={167 + i * 42} width={Math.max(1.5, (r.bytes / budget.pool) * W)} height={7} rx={2} className={i === 4 ? "fill-text/70" : "fill-accent"} />
        </motion.g>
      ))}
      <Label x={X} y={384}>bars to the scale of the pool; calculated from the model&apos;s shape and the block count</Label>
    </Frame>
  );
}

export function BudgetFigure() {
  const { on } = useLoop(14);
  const top = 40;
  const h = 200;
  const maxBatch = batchRows[batchRows.length - 1].batch;
  const x = (b: number) => X + 30 + (b / maxBatch) * (W - 30);
  const y = (c: number) => top + h - (c / budget.fullContext) * h;
  const curve = Array.from({ length: 128 }, (_, i) => 1 + (i * (maxBatch - 1)) / 127).map((b) => `${x(b)},${y(Math.min(budget.fullContext, budget.poolTokens / b))}`).join(" ");
  return (
    <Frame label={`Tokens each sequence can reach against the number of sequences: ${batchRows.map((r) => `${r.batch} sequences ${fmt(r.context)} tokens`).join(", ")}`}>
      <Label x={X} y={22} className="fill-text">
        tokens each sequence can reach, against how many share the pool
      </Label>
      <line x1={x(0)} y1={top + h} x2={x(maxBatch)} y2={top + h} className="stroke-border" />
      <line x1={x(0)} y1={top} x2={x(0)} y2={top + h} className="stroke-border" />
      <Label x={x(0) - 4} y={y(budget.fullContext) + 4} anchor="end">
        {fmt(budget.fullContext)}
      </Label>
      <Label x={x(0) - 4} y={top + h + 4} anchor="end">
        0
      </Label>
      <motion.polyline points={curve} fill="none" strokeWidth={2} className="stroke-accent" {...on(0.08)} />
      {batchRows.map((r, i) => (
        <motion.g key={r.batch} {...on(0.15 + i * 0.06)}>
          <circle cx={x(r.batch)} cy={y(r.context)} r={4} className={r.bound === "pool" ? "fill-warn" : "fill-accent"} />
          <Label x={x(r.batch)} y={top + h + 16} anchor="middle">
            {r.batch}
          </Label>
        </motion.g>
      ))}
      <Label x={x(maxBatch)} y={top + h + 32} anchor="end">
        sequences sharing the pool
      </Label>
      <motion.g {...on(0.6)}>
        <Label x={X} y={300} className="fill-accent">{`up to ${budget.reservedFit} sequences: the model's ${fmt(budget.fullContext)} positions are the limit, and pool is left over`}</Label>
        <Label x={X} y={320} className="fill-warn">{`past ${budget.reservedFit}: the pool is the limit, and batch × context stays at ${fmt(budget.poolTokens)} tokens`}</Label>
        <Label x={X} y={340}>{`the engine was run with at most ${budget.maxBatch} to ${maxBatch} sequences`}</Label>
      </motion.g>
      <Label x={X} y={384}>{`calculated: ${fmt(budget.poolTokens)} tokens in the pool, shared equally, rounded down to whole blocks`}</Label>
    </Frame>
  );
}

export function FitFigure() {
  const { on, grow } = useLoop(14);
  const px = (W - 40) / budget.vram;
  return (
    <Frame label={`On this GPU: ${fits.map((f) => `${f.name}: weights ${gb(f.weights)}, ${f.over > 0 ? `${gb(f.over)} too large` : `${fmt(f.contexts)} contexts of ${fmt(budget.fullContext)} tokens`}`).join("; ")}`}>
      <Label x={X} y={22} className="fill-text">{`four models in the same ${gb(budget.vram)}`}</Label>
      {fits.map((f, i) => {
        const yy = 44 + i * 78;
        const segs = [
          { w: budget.rest * px, c: cls.rest },
          { w: Math.min(f.weights, budget.vram - budget.rest) * px, c: cls.weights },
          { w: f.pool * px, c: cls.pool },
          { w: f.spare * px, c: cls.spare },
        ];
        const starts = before(segs.map((s) => s.w));
        return (
          <motion.g key={f.name} {...on(0.05 + i * 0.12)}>
            <Label x={X} y={yy} className="fill-text">
              {f.name}
            </Label>
            <Label x={X + W} y={yy} anchor="end" className={f.over > 0 ? "fill-warn" : "fill-accent"}>
              {f.over > 0 ? `${gb(f.over, 1)} too large` : `${fmt(f.contexts)} contexts of ${fmt(budget.fullContext)}`}
            </Label>
            <rect x={X} y={yy + 8} width={budget.vram * px} height={20} rx={2} className="fill-border" />
            <motion.g {...grow(0.08 + i * 0.12, 0.25 + i * 0.12)}>
              {segs.map((s, k) => (
                <rect key={k} x={X + starts[k]} y={yy + 8} width={Math.max(0, s.w - 1)} height={20} rx={2} className={s.c} />
              ))}
              {f.over > 0 && <rect x={X + budget.vram * px + 2} y={yy + 8} width={36} height={20} rx={2} className="fill-warn" />}
            </motion.g>
            <Label x={X} y={yy + 44}>{`weights ${gb(f.weights, 1)} · cache ${fmt(f.perToken)} bytes per token${f.over > 0 ? "" : ` · pool ${gb(f.pool, 1)} = ${fmt(f.tokens)} tokens`}`}</Label>
          </motion.g>
        );
      })}
      <Label x={X} y={372}>grey: everything else; dark: weights; blue: KV pool; pale: kept free</Label>
      <Label x={X} y={388}>TinyLlama measured; the others calculated from published shapes, cache in BF16</Label>
    </Frame>
  );
}
