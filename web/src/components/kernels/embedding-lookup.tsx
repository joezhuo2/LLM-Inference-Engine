"use client";

import clsx from "clsx";
import { motion } from "motion/react";
import { embeddingLaunch, modelShape } from "@/lib/data";
import { EasedBar } from "./eased-bar";
import { Label } from "./frame";
import { useLoop } from "./loop";

const shape = modelShape();
const launch = embeddingLaunch();
const naiveThreads = 256;
const naiveLoads = shape.hidden / naiveThreads;
const valueBytes = launch.bytesPerThread / launch.valuesPerLoad;
const whole = (n: number) => n.toLocaleString("en-US");
const range = ([lo, hi]: [number, number]) => `${lo} to ${hi}`;

export const embeddingNumbers = {
  ...launch,
  hidden: shape.hidden,
  vocab: shape.vocab,
  naiveThreads,
  naiveLoads,
  valueBytes,
  threadsPerRow: shape.hidden / launch.valuesPerLoad,
  speedup: launch.decodeUs.naive / launch.decodeUs.fast,
  run: `nsys, max batch ${launch.maxBatch}, ${whole(launch.promptTokens)}-token prompts, ${launch.date}`,
};

const n = embeddingNumbers;

const ids = [
  { id: 1, row: 0 },
  { id: 15043, row: 4 },
  { id: 3186, row: 2 },
];

export function GatherFigure() {
  const { on, move } = useLoop(8);
  const rows = 6;
  const rowH = 16;
  const tx = 150;
  const ox = 400;
  return (
    <svg viewBox="0 0 560 160" className="w-full tabular-nums" role="img" aria-label="Each token id picks one row of the embedding table, copied into the hidden state">
      <Label x={20} y={18}>token ids</Label>
      {ids.map((t, i) => (
        <motion.g key={t.id} {...on(0.04 + i * 0.28, 0.94)}>
          <rect x={20} y={30 + i * 26} width={70} height={18} rx={3} className="fill-accent/15 stroke-accent" />
          <Label x={55} y={43 + i * 26} anchor="middle" className="fill-text">
            {whole(t.id)}
          </Label>
        </motion.g>
      ))}
      <Label x={tx} y={18}>{`table E: ${whole(n.vocab)} × ${whole(n.hidden)}`}</Label>
      {Array.from({ length: rows }, (_, r) => (
        <rect key={r} x={tx} y={30 + r * (rowH + 2)} width={200} height={rowH} rx={2} className="fill-baseline/40" />
      ))}
      <Label x={tx + 100} y={30 + rows * (rowH + 2) + 12} anchor="middle">
        ⋮
      </Label>
      <Label x={ox} y={18}>{`x: tokens × ${whole(n.hidden)}`}</Label>
      {ids.map((t, i) => {
        const y = 30 + t.row * (rowH + 2);
        const yo = 30 + i * 26;
        const t0 = 0.08 + i * 0.28;
        return (
          <g key={t.id}>
            <motion.rect x={tx} y={y} width={200} height={rowH} rx={2} className="fill-accent/50" {...on(t0, t0 + 0.2, false)} />
            <motion.rect x={tx} y={y} width={140} height={rowH} rx={2} className="fill-accent" {...move(t0 + 0.04, t0 + 0.2, ox - tx, yo - y + 1, true, 0.94)} />
          </g>
        );
      })}
    </svg>
  );
}

const cell = 100 / 34;

export function CopyLanes({ compact = false }: { compact?: boolean }) {
  const { on } = useLoop(8);
  const lanes = [
    { name: "naive", how: `one block per token, ${n.naiveThreads} threads`, note: `${n.naiveLoads} loads of ${n.valueBytes} bytes`, cells: Array.from({ length: n.naiveLoads }, (_, i) => ({ left: (1 + i * 4.2) * cell, width: cell, start: 0.1 + i * 0.09 })), accent: false },
    { name: "fast", how: `${n.naiveThreads}-thread blocks over every row`, note: `1 load of ${n.bytesPerThread} bytes`, cells: [{ left: cell, width: n.valuesPerLoad * cell, start: 0.1 }], accent: true },
  ];
  return (
    <div className="space-y-2 tabular-nums text-xs" role="img" aria-label={`One thread of the naive kernel copies ${n.naiveLoads} values of a row in ${n.naiveLoads} loads of ${n.valueBytes} bytes; one thread of the fast kernel copies ${n.valuesPerLoad} values in one load of ${n.bytesPerThread} bytes`}>
      <div className="text-muted">{`what one thread copies: ${n.valuesPerLoad} BF16 values of a row (calculated from the row width, not to scale)`}</div>
      {lanes.map((l) => (
        <div key={l.name} className="grid grid-cols-[2.75rem_1fr] items-center gap-x-3 gap-y-0.5 sm:grid-cols-[2.75rem_1fr_13rem]">
          <span className={l.accent ? "text-accent" : "text-text"}>{l.name}</span>
          <div className="relative h-3 min-w-0 overflow-hidden rounded-sm bg-baseline/30">
            {l.cells.map((c, i) => (
              <motion.div key={i} className={l.accent ? "absolute inset-y-0 rounded-sm bg-accent" : "absolute inset-y-0 rounded-sm bg-baseline"} style={{ left: `${c.left}%`, width: `${c.width}%` }} {...on(c.start, 0.94)} />
            ))}
          </div>
          <span className="col-start-2 text-muted sm:col-start-3">
            <span className={l.accent ? "text-accent" : "text-text"}>{l.note}</span>
            <span className={compact ? "max-md:hidden" : undefined}>{` · ${l.how}`}</span>
          </span>
        </div>
      ))}
    </div>
  );
}

export function LaunchBars({ compact = false, brief = false }: { compact?: boolean; brief?: boolean }) {
  const groups = [
    { label: "median launch in decode steps, µs, measured", max: n.decodeUs.naive, bars: [{ name: "naive", us: n.decodeUs.naive, text: `${n.decodeUs.naive} µs` }, { name: "fast", us: n.decodeUs.fast, text: `${n.decodeUs.fast} µs` }] },
    { label: `largest launch in ${whole(n.prefillTokens)}-token prefill steps, µs, measured`, max: n.prefillUs.naive[1], bars: [{ name: "naive", us: n.prefillUs.naive[1], text: `${range(n.prefillUs.naive)} µs` }, { name: "fast", us: n.prefillUs.fast[1], text: `${range(n.prefillUs.fast)} µs` }] },
  ];
  return (
    <div className="space-y-3 tabular-nums text-xs">
      {groups.slice(0, brief ? 1 : groups.length).map((g, i) => (
        <div key={g.label} className={clsx("space-y-1.5", compact && i > 0 && "max-md:hidden")}>
          <div className="text-muted">{g.label}</div>
          {g.bars.map((b) => (
            <div key={b.name} className="grid grid-cols-[2.75rem_1fr_6.5rem] items-center gap-3">
              <span className={b.name === "fast" ? "text-accent" : "text-text"}>{b.name}</span>
              <div className="h-2 min-w-0">
                <EasedBar fraction={b.us / g.max} className={b.name === "fast" ? "bg-accent" : "bg-baseline"} />
              </div>
              <span className={b.name === "fast" ? "text-right text-accent" : "text-right text-muted"}>{b.text}</span>
            </div>
          ))}
        </div>
      ))}
      {!brief && (
        <div className="border-t border-border pt-2 leading-relaxed text-muted">
          {`${n.run}. `}
          <span className="text-text">{`${n.speedup.toFixed(1)}x faster per launch (calculated)`}</span>
          {`, and under ${n.underPct}% of a step either way, so no end-to-end number moved.`}
        </div>
      )}
    </div>
  );
}

export function EmbeddingPanel({ compact = false }: { compact?: boolean }) {
  return (
    <div className="flex h-full flex-col justify-center gap-4">
      <div className={compact ? "max-md:hidden" : undefined}>
        <GatherFigure />
      </div>
      <CopyLanes compact={compact} />
      <LaunchBars compact={compact} />
    </div>
  );
}

export function LookupScene() {
  return (
    <div className="flex h-full flex-col justify-center gap-4 px-2">
      <GatherFigure />
      <LaunchBars brief />
    </div>
  );
}
