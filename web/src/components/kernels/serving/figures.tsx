"use client";

import NumberFlow from "@number-flow/react";
import { motion } from "motion/react";
import { useEffect, useState } from "react";
import { Legend, LineChart, Stats, type Axis } from "@/components/results/line-chart";
import { SmoothRange } from "@/components/smooth-range";
import { blockBytes, blockTokens, budgetSystem, engine, fmt, fullBlocks, fullContext, fullMb, heldBy, kvValues, loadRates, maxBatch, peak, poolBlocks, poolFraction, poolGb, reservedFit, sweepRows, tokenBytes, tpot, usableBlocks, vllm, worstCaseBlocks } from "@/content/kernels/serving";
import { goodputSeries, tpotLoadSeries } from "@/content/results";
import { kvReplay, modelShape } from "@/lib/data";
import { usePrefersReducedMotion } from "@/lib/reduced-motion";
import { Frame, Label } from "../frame";
import { useLoop } from "../loop";
import { continuousBatching, type Grid, lengths, slots, staticBatching } from "./batching";
import { firstEviction, queueBlocks, queueConfig, queueRequests, queueRun } from "./queue";

const layers = modelShape().layers;
const sample = kvReplay().requests[0];

export function BudgetFigure() {
  const { on } = useLoop(12);
  const pair = 520 / layers;
  const slot = 520 / blockTokens;
  const promptBlocks = Math.ceil(sample.prompt / blockTokens);
  const blocks = Math.ceil(sample.final_length / blockTokens);
  const bw = Math.min(40, 520 / blocks);
  const fullW = Math.max(3, (fullBlocks / poolBlocks) * 520);
  return (
    <Frame label={`One token costs ${fmt(tokenBytes)} bytes of KV cache: a key and a value of ${kvValues} numbers in each of ${layers} layers, 2 bytes a number. A block holds ${blockTokens} tokens, ${fmt(blockBytes)} bytes. A request holds one block for every ${blockTokens} tokens it has, and the pool is ${fmt(poolBlocks)} blocks, ${fmt(poolGb, 2)} GB`}>
      <Label x={20} y={22} className="fill-text">{`one token: a key and a value in each of the ${layers} layers`}</Label>
      {Array.from({ length: layers }, (_, l) => (
        <g key={l}>
          <rect x={20 + l * pair} y={32} width={pair / 2 - 1} height={20} rx={1.5} className="fill-accent" />
          <rect x={20 + l * pair + pair / 2} y={32} width={pair / 2 - 2} height={20} rx={1.5} className="fill-accent/45" />
        </g>
      ))}
      <Label x={20} y={70} className="fill-accent">{`2 × ${layers} layers × ${kvValues} numbers × 2 bytes = ${fmt(tokenBytes)} bytes`}</Label>
      <Label x={20} y={104} className="fill-text">{`one block: ${blockTokens} tokens, filled one decode step at a time`}</Label>
      {Array.from({ length: blockTokens }, (_, s) => (
        <g key={s}>
          <rect x={20 + s * slot} y={114} width={slot - 3} height={20} rx={2} className="fill-surface stroke-border" />
          <motion.rect x={20 + s * slot} y={114} width={slot - 3} height={20} rx={2} className="fill-accent" {...on(0.04 + s * 0.022, 0.94)} />
        </g>
      ))}
      <Label x={20} y={152} className="fill-accent">{`${blockTokens} × ${fmt(tokenBytes)} = ${fmt(blockBytes)} bytes`}</Label>
      <Label x={20} y={186} className="fill-text">{`one request: ceil(length / ${blockTokens}) blocks, one more every ${blockTokens} tokens`}</Label>
      {Array.from({ length: blocks }, (_, j) => (
        <g key={j}>
          <rect x={20 + j * bw} y={196} width={bw - 3} height={20} rx={2} className="fill-surface stroke-border" strokeDasharray="3 2" />
          {j < promptBlocks ? <rect x={20 + j * bw} y={196} width={bw - 3} height={20} rx={2} className="fill-accent/45" /> : <motion.rect x={20 + j * bw} y={196} width={bw - 3} height={20} rx={2} className="fill-accent" {...on(0.42 + ((j - promptBlocks) / Math.max(1, blocks - promptBlocks)) * 0.4, 0.94)} />}
        </g>
      ))}
      <Label x={20} y={234}>{`a recorded chat request: ${promptBlocks} blocks for its ${sample.prompt}-token prompt, ${blocks} when it stops at ${sample.final_length} tokens`}</Label>
      <Label x={20} y={270} className="fill-text">{`the pool: ${fmt(poolBlocks)} blocks, ${fmt(poolGb, 2)} GB, ${poolFraction}% of the memory free after the weights`}</Label>
      <rect x={20} y={280} width={520} height={26} rx={3} className="fill-surface stroke-border" />
      <rect x={20} y={280} width={fullW} height={26} rx={2} className="fill-warn/70" />
      <motion.rect x={20} y={280} width={Math.max(1.5, (blocks / poolBlocks) * 520)} height={26} rx={1} className="fill-accent" {...on(0.8, 0.94)} />
      <Label x={20} y={324} className="fill-warn">{`a request at the full ${fmt(fullContext)} tokens: ${fullBlocks} blocks, ${fmt(fullMb)} MB, the pink sliver`}</Label>
      <Label x={20} y={342} className="fill-accent">{`the chat request above: ${blocks} blocks, drawn inside it`}</Label>
      <Label x={20} y={376}>{`${fmt(usableBlocks)} blocks can be handed out; block 0 is kept as the null block`}</Label>
    </Frame>
  );
}

const trayCols = 20;
const trayW = 520 / trayCols;
const trayH = 17;
const trayRows = Math.ceil(reservedFit / trayCols);

function Trays({ y, filled, cls, title, titleCls }: { y: number; filled: number; cls: string; title: string; titleCls: string }) {
  return (
    <g>
      <Label x={20} y={y - 8} className={titleCls}>{title}</Label>
      {Array.from({ length: trayRows * trayCols }, (_, i) => {
        if (i >= reservedFit) return null;
        const x = 20 + (i % trayCols) * trayW;
        const ty = y + Math.floor(i / trayCols) * (trayH + 3);
        const part = Math.max(0, Math.min(1, filled - i));
        return (
          <g key={i}>
            <rect x={x} y={ty} width={trayW - 3} height={trayH} rx={2} className="fill-surface stroke-border" strokeWidth={0.75} />
            {part > 0 && <rect x={x} y={ty} width={(trayW - 3) * part} height={trayH} rx={2} className={cls} />}
          </g>
        );
      })}
    </g>
  );
}

const fitMax = 256;

export function FitFigure() {
  const [n, setN] = useState(reservedFit);
  const held = heldBy(n);
  const reserved = n * fullBlocks;
  const turned = Math.max(0, n - reservedFit);
  const stats = [
    { label: "requests", value: n, cls: "text-text", suffix: "" },
    { label: "blocks if reserved", value: reserved, cls: reserved > usableBlocks ? "text-warn" : "text-text", suffix: "" },
    { label: "blocks actually held", value: held, cls: "text-accent", suffix: "" },
    { label: "of the pool, held", value: (100 * held) / poolBlocks, cls: "text-accent", suffix: "%" },
  ];
  return (
    <div className="flex h-full flex-col gap-2 tabular-nums text-xs">
      <div className="flex items-center gap-3">
        <span className="shrink-0 text-muted">requests</span>
        <SmoothRange min={1} max={fitMax} step={1} value={n} onValue={(v) => setN(Math.round(v))} aria-label="requests running at once" className="w-full accent-[var(--accent)]" />
        <span className="w-10 shrink-0 text-right text-text">{n}</span>
      </div>
      <svg viewBox="0 0 560 300" className="w-full" role="img" aria-label={`${n} requests in a pool of ${fmt(poolBlocks)} blocks: reserving the full ${fmt(fullContext)}-token context for each needs ${fmt(reserved)} blocks${turned ? `, so ${turned} do not fit` : ""}; holding only the blocks their tokens fill needs ${fmt(held)}`}>
        <Trays y={26} filled={Math.min(n, reservedFit)} cls="fill-warn/70" titleCls={turned ? "fill-warn" : "fill-text"} title={turned ? `reserving ${fmt(fullContext)} tokens each: ${reservedFit} fit, ${turned} wait outside` : `reserving ${fmt(fullContext)} tokens each: one tray per request`} />
        <Trays y={170} filled={held / fullBlocks} cls="fill-accent" titleCls="fill-text" title={`blocks taken as the requests grow: ${fmt(held)} blocks, packed into the same trays`} />
        <Label x={20} y={290}>{`a tray is ${fullBlocks} blocks, one full context; ${reservedFit} trays are the pool; lengths are the recorded chat run's, repeated`}</Label>
      </svg>
      <div className="grid grid-cols-2 gap-x-4 gap-y-2 border-t border-border pt-2 sm:grid-cols-4">
        {stats.map((r) => (
          <div key={r.label}>
            <div className="text-muted">{r.label}</div>
            <div className={r.cls}>
              <NumberFlow value={r.value} suffix={r.suffix} format={{ maximumFractionDigits: 1 }} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export function BatchLimitFigure() {
  const { grow, on } = useLoop(10);
  const x0 = 96;
  const span = 300;
  const top = sweepRows[sweepRows.length - 1].batch * fullBlocks;
  const bx = (blocks: number) => (blocks / top) * span;
  const rowY = (i: number) => 62 + i * 46;
  const poolX = x0 + bx(usableBlocks);
  return (
    <Frame label={`For each max batch from ${sweepRows[0].batch} to ${sweepRows[sweepRows.length - 1].batch}: the blocks held if every request reached the full context against the pool of ${fmt(poolBlocks)} blocks, the most blocks a measured run held, and the measured step time. At max batch ${maxBatch} the worst case is ${fmt(worstCaseBlocks)} blocks`}>
      <Label x={20} y={22} className="fill-text">what each max_batch could hold, and what a step costs</Label>
      <rect x={x0} y={32} width={10} height={10} rx={2} className="fill-none stroke-warn" />
      <Label x={x0 + 16} y={41}>every request at the full context</Label>
      <rect x={x0 + 216} y={32} width={10} height={10} rx={2} className="fill-accent" />
      <Label x={x0 + 232} y={41}>most blocks held, measured</Label>
      {sweepRows.map((r, i) => {
        const worst = r.batch * fullBlocks;
        const over = worst > usableBlocks;
        return (
          <g key={r.batch}>
            <Label x={20} y={rowY(i) + 14} className="fill-text">{`max_batch ${r.batch}`}</Label>
            <motion.rect x={x0} y={rowY(i)} width={bx(worst)} height={20} rx={2} className={over ? "fill-warn/20 stroke-warn" : "fill-warn/10 stroke-warn/70"} strokeWidth={1} {...grow(0.04 + i * 0.06, 0.2 + i * 0.06)} />
            <motion.rect x={x0} y={rowY(i) + 4} width={Math.max(2, bx(r.peak))} height={12} rx={1.5} className="fill-accent" {...grow(0.36 + i * 0.05, 0.5 + i * 0.05)} />
            <Label x={x0 + span + 12} y={rowY(i) + 9} className="fill-text">{`${r.tpot} ms a step`}</Label>
            <Label x={x0 + span + 12} y={rowY(i) + 23}>{`${fmt(r.tokens)} tokens/s`}</Label>
            <motion.g {...on(0.22 + i * 0.06)}>
              <Label x={x0 + bx(worst) - 6} y={rowY(i) + 36} anchor="end" className={over ? "fill-warn" : "fill-muted"}>{`${fmt(worst)} blocks${over ? ", more than the pool" : ""}`}</Label>
            </motion.g>
          </g>
        );
      })}
      <line x1={poolX} x2={poolX} y1={52} y2={rowY(sweepRows.length) - 8} className="stroke-text" strokeDasharray="4 3" />
      <Label x={poolX + 5} y={60} className="fill-text">{`the pool, ${fmt(poolBlocks)}`}</Label>
      <motion.g {...on(0.7)}>
        <Label x={20} y={rowY(sweepRows.length) + 22} className="fill-accent">{`max_batch ${maxBatch}: at most ${fmt(worstCaseBlocks)} blocks, so the pool cannot run out`}</Label>
        <Label x={20} y={rowY(sweepRows.length) + 40}>the measured runs used 64 prompt and 64 output tokens, twice max_batch requests at once</Label>
      </motion.g>
    </Frame>
  );
}

const reqTint = ["fill-accent", "fill-accent/55", "fill-baseline"];
const chip = { w: 44, h: 24 };
const noteWidth = 88;

function wrap(text: string, width: number) {
  const lines: string[] = [];
  let line = "";
  for (const word of text.split(" ")) {
    if (line && line.length + word.length + 1 > width) {
      lines.push(line);
      line = word;
    } else line = line ? `${line} ${word}` : word;
  }
  if (line) lines.push(line);
  return lines;
}

export function QueueFigure({ from, to, still }: { from: number; to: number; still: number }) {
  const reduced = usePrefersReducedMotion();
  const [at, setAt] = useState(from);
  const [paused, setPaused] = useState(false);
  useEffect(() => {
    if (reduced || paused) return;
    const timer = setInterval(() => setAt((i) => (i >= to ? from : i + 1)), 2600);
    return () => clearInterval(timer);
  }, [reduced, paused, from, to]);
  const i = reduced && !paused ? still : Math.min(Math.max(at, from), to);
  const snap = queueRun[i];
  const ease = { duration: reduced ? 0 : 0.7, ease: "easeInOut" as const };
  const place = (id: number) => {
    const w = snap.waiting.indexOf(id);
    if (w >= 0) return { x: 20 + w * (chip.w + 8), y: 34, o: 1 };
    const r = snap.running.indexOf(id);
    if (r >= 0) return { x: 20 + r * 130, y: 104, o: 1 };
    const d = snap.done.indexOf(id);
    return { x: 330 + d * (chip.w - 8), y: 34, o: 0.35 };
  };
  const cell = 520 / queueBlocks;
  const lines = wrap(`step ${snap.step}: ${snap.note}`, noteWidth);
  return (
    <div className="flex h-full flex-col gap-2 tabular-nums text-xs">
      <div className="flex items-center gap-3">
        <button type="button" onClick={() => setPaused((p) => !p)} aria-pressed={!paused} className="pressable rounded border border-border px-2.5 py-1 text-accent">
          {paused || reduced ? "play" : "pause"}
        </button>
        <SmoothRange
          min={from}
          max={to}
          step={1}
          value={i}
          onValue={(v) => {
            setPaused(true);
            setAt(Math.round(v));
          }}
          aria-label="step of the example run"
          className="w-full accent-[var(--accent)]"
        />
        <span className="w-14 shrink-0 text-right text-muted">{`step ${snap.step}`}</span>
      </div>
      <svg viewBox="0 0 560 330" className="w-full [font-variant-ligatures:none]" role="img" aria-label={`The scheduler example at step ${snap.step}: ${snap.note}. Running: ${snap.running.map((id) => queueRequests[id].name).join(", ") || "none"}; waiting: ${snap.waiting.map((id) => queueRequests[id].name).join(", ") || "none"}; ${snap.free} of ${queueBlocks} blocks free`}>
        <Label x={20} y={22}>waiting queue, front first</Label>
        <Label x={330} y={22}>finished</Label>
        <Label x={20} y={92}>{`running list, max_batch ${queueConfig.maxBatch}, newest last`}</Label>
        {Array.from({ length: queueConfig.maxBatch }, (_, k) => {
          const id = snap.running[k];
          return (
            <g key={k}>
              <rect x={16} y={100} width={122} height={32} rx={3} transform={`translate(${k * 130} 0)`} className="fill-none stroke-border" strokeDasharray="3 2" />
              {id !== undefined && (
                <text x={20 + k * 130 + chip.w + 6} y={120} className="fill-muted text-[10px]">{`${snap.cached[id]} tok, ${snap.held[id]} blk`}</text>
              )}
            </g>
          );
        })}
        {queueRequests.map((r, id) => {
          const p = place(id);
          const out = snap.evicted.includes(id);
          return (
            <motion.g key={id} initial={false} animate={{ x: p.x, y: p.y, opacity: p.o }} transition={ease}>
              <rect width={chip.w} height={chip.h} rx={3} className={out ? "fill-warn" : reqTint[id % 3]} />
              <text x={chip.w / 2} y={16} textAnchor="middle" className="fill-bg text-[11px]">
                {r.name}
              </text>
            </motion.g>
          );
        })}
        <Label x={20} y={164}>{`the pool: ${queueBlocks} blocks of ${queueConfig.blockSize} tokens, ${snap.free} free`}</Label>
        {Array.from({ length: queueBlocks }, (_, k) => {
          const owner = snap.owner[k + 1];
          return (
            <g key={k}>
              <rect x={20 + k * cell} y={172} width={cell - 4} height={30} rx={3} className="fill-surface stroke-border" />
              <motion.rect x={20 + k * cell} y={172} width={cell - 4} height={30} rx={3} className={owner >= 0 ? reqTint[owner % 3] : "fill-surface"} initial={false} animate={{ opacity: owner >= 0 ? 1 : 0 }} transition={ease} />
              {owner >= 0 && (
                <text x={20 + k * cell + (cell - 4) / 2} y={191} textAnchor="middle" className="fill-bg text-[11px]">
                  {queueRequests[owner].name}
                </text>
              )}
            </g>
          );
        })}
        {lines.map((l, k) => (
          <Label key={k} x={20} y={232 + k * 16} className={snap.evicted.length ? "fill-warn" : "fill-text"}>
            {l}
          </Label>
        ))}
        <Label x={20} y={322}>{`illustrative: blocks of ${queueConfig.blockSize} tokens and max_batch ${queueConfig.maxBatch}, run through a port of Scheduler::schedule()`}</Label>
      </svg>
    </div>
  );
}

export const queueStill = firstEviction;

const qpsAxis: Axis = { kind: "linear", min: 0.5, max: Number(loadRates[loadRates.length - 1]) + 0.5, ticks: loadRates.map(Number), format: String, title: "offered load (requests per second, Poisson)" };
const goodputTop = Math.ceil(Math.max(...goodputSeries.flatMap((s) => s.values.filter((v): v is number => v !== undefined))) / 2) * 2;
const goodputAxis: Axis = { kind: "linear", min: 0, max: goodputTop, ticks: Array.from({ length: goodputTop / 2 + 1 }, (_, i) => i * 2), format: String, title: "goodput (requests/s that meet the SLO)" };
const tpotTop = Math.ceil(Math.max(...tpotLoadSeries.flatMap((s) => s.values.filter((v): v is number => v !== undefined))) / 4) * 4;
const tpotAxis: Axis = { kind: "linear", min: 0, max: tpotTop, ticks: Array.from({ length: tpotTop / 4 + 1 }, (_, i) => i * 4), format: String, title: "ms per output token (TPOT p50)" };
const xs = loadRates.map(Number);
const high = loadRates[loadRates.length - 1];

export function LoadFigure({ kind }: { kind: "goodput" | "tpot" }) {
  if (kind === "goodput") {
    return (
      <div className="flex h-full flex-col gap-2">
        <Legend series={goodputSeries} />
        <LineChart view="serving-goodput" label="Goodput against offered load for the engine with and without a token budget and for vLLM" xs={xs} xAxis={qpsAxis} yAxis={goodputAxis} series={goodputSeries} heading={(i) => `${xs[i]} requests/s offered`} value={(v) => `${v.toFixed(2)} req/s`} />
        <Stats
          stats={[
            { label: "engine, best", value: peak(engine).value, digits: 2, suffix: " req/s" },
            { label: "with a token budget", value: peak(budgetSystem).value, digits: 2, suffix: " req/s" },
            { label: "vLLM, best", value: peak(vllm).value, digits: 2, suffix: " req/s" },
          ]}
        />
      </div>
    );
  }
  return (
    <div className="flex h-full flex-col gap-2">
      <Legend series={tpotLoadSeries} />
      <LineChart view="serving-tpot" label="Median time per output token against offered load for the engine with and without a token budget and for vLLM" xs={xs} xAxis={qpsAxis} yAxis={tpotAxis} series={tpotLoadSeries} heading={(i) => `${xs[i]} requests/s offered`} value={(v) => `${v.toFixed(1)} ms`} />
      <Stats
        stats={[
          { label: `engine at ${loadRates[0]}/s`, value: tpot(engine, loadRates[0]), digits: 1, suffix: " ms" },
          { label: `engine at ${high}/s`, value: tpot(engine, high), digits: 1, suffix: " ms" },
          { label: `vLLM at ${high}/s`, value: tpot(vllm, high), digits: 1, suffix: " ms" },
        ]}
      />
    </div>
  );
}

const staticGrid = staticBatching();
export const roomyGrid = continuousBatching(100);
export const tightBlocks = 8;
export const tightGrid = continuousBatching(tightBlocks);
export const batchGrids = { static: staticGrid, roomy: roomyGrid, tight: tightGrid };

const gx = 28;
const gw = 540 - gx;
const cellW = gw / staticGrid.steps;
const rowH = 30;
const rowPitch = 34;
const gridTop = 52;
const chipW = 520 / lengths.length;
const columnMs = 600;
const holdColumns = 5;
const chipTint = ["fill-accent", "fill-accent/55", "fill-baseline"];

function useColumns(total: number) {
  const reduced = usePrefersReducedMotion();
  const [col, setCol] = useState(0);
  useEffect(() => {
    if (reduced) return;
    const timer = setInterval(() => setCol((c) => (c >= total + holdColumns ? 0 : c + 1)), columnMs);
    return () => clearInterval(timer);
  }, [reduced, total]);
  return { col: reduced ? total : Math.min(col, total), reduced };
}

function idleBefore(g: Grid, col: number) {
  let n = 0;
  for (let s = 0; s < g.slots; s++) for (let t = 0; t < col; t++) if (g.cells[s][t] === null || g.redo[s][t]) n++;
  return n;
}

export type BatchKind = "static" | "roomy" | "tight";

export function BatchingFigure({ kind, focus = false }: { kind: BatchKind; focus?: boolean }) {
  const g = batchGrids[kind];
  const { col: raw, reduced } = useColumns(g.steps + holdColumns);
  const col = Math.min(raw, g.steps);
  const ease = { duration: reduced ? 0 : 0.5, ease: "easeInOut" as const };
  const waiting = new Set(col === 0 ? lengths.map((_, id) => id) : g.waiting[col - 1]);
  const evictions = g.evictions.slice(0, col).reduce((n, e) => n + e, 0);
  const wasted = idleBefore(g, col);
  const stats =
    kind === "static"
      ? [
          { label: "steps so far", value: col, cls: "text-text" },
          { label: "slots in use", value: col ? g.running[col - 1] : 0, cls: "text-text" },
          { label: "wasted slot-steps", value: wasted, cls: "text-warn" },
          { label: "steps to finish", value: g.steps, cls: "text-muted" },
        ]
      : [
          { label: "requests running", value: col ? g.running[col - 1] : 0, cls: focus ? "text-accent" : "text-text" },
          { label: "preemptions", value: evictions, cls: focus ? "text-accent" : "text-warn" },
          { label: "wasted slot-steps", value: wasted, cls: "text-warn" },
          { label: "static batching wasted", value: staticGrid.wasted, cls: "text-muted" },
        ];
  const height = gridTop + slots * rowPitch + 18;
  return (
    <div className="flex h-full flex-col gap-2 tabular-nums text-xs">
      <svg
        viewBox={`0 0 560 ${height}`}
        className="w-full [font-variant-ligatures:none]"
        role="img"
        aria-label={`${kind === "static" ? "Static batching" : "Continuous batching"} of ${lengths.length} requests in ${slots} slots: ${g.steps} steps, ${g.wasted} wasted slot-steps${kind === "tight" ? `, ${g.evictions.reduce((n, e) => n + e, 0)} preemptions` : ""}`}
      >
        <Label x={20} y={12}>{`queue, request:steps (illustrative lengths)`}</Label>
        {lengths.map((len, id) => (
          <motion.g key={id} initial={false} animate={{ opacity: waiting.has(id) ? 1 : 0.2 }} transition={ease}>
            <rect x={20 + id * chipW} y={20} width={chipW - 3} height={18} rx={2} className={chipTint[id % 3]} />
            <text x={20 + id * chipW + (chipW - 3) / 2} y={33} textAnchor="middle" className="fill-bg text-[10px]">{`${id}:${len}`}</text>
          </motion.g>
        ))}
        {Array.from({ length: slots }, (_, s) => (
          <g key={s}>
            <Label x={8} y={gridTop + s * rowPitch + 19} anchor="middle">{s + 1}</Label>
            {Array.from({ length: staticGrid.steps }, (_, t) => {
              const id = g.cells[s]?.[t] ?? null;
              const redo = g.redo[s]?.[t] ?? false;
              const gone = t < g.steps && g.preempted[s].includes(t);
              const shown = t < col;
              const x = gx + t * cellW;
              const y = gridTop + s * rowPitch;
              const idle = id === null || redo;
              return (
                <g key={t}>
                  <rect x={x} y={y} width={cellW - 2} height={rowH} rx={2} className="fill-bg stroke-border" strokeWidth={0.5} />
                  {t < g.steps && (
                    <motion.g
                      style={{ transformBox: "fill-box", transformOrigin: "center" }}
                      initial={false}
                      animate={{ opacity: shown ? 1 : 0, transform: shown ? "scale(1)" : "scale(0.6)" }}
                      transition={ease}
                    >
                      {idle && !redo && <rect x={x} y={y} width={cellW - 2} height={rowH} rx={2} className="fill-warn/10 stroke-warn/50" strokeDasharray="3 2" strokeWidth={0.75} />}
                      {redo && <rect x={x} y={y} width={cellW - 2} height={rowH} rx={2} className="fill-warn/45" />}
                      {id !== null && !redo && <rect x={x} y={y} width={cellW - 2} height={rowH} rx={2} className={chipTint[id % 3]} />}
                      {id !== null && (
                        <text x={x + (cellW - 2) / 2} y={y + 19} textAnchor="middle" className={`${redo ? "fill-text" : "fill-bg"} text-[10px]`}>
                          {id}
                        </text>
                      )}
                      {gone && (
                        <text x={x + (cellW - 2) / 2} y={y + 20} textAnchor="middle" className="fill-warn text-[13px]">
                          ×
                        </text>
                      )}
                    </motion.g>
                  )}
                </g>
              );
            })}
          </g>
        ))}
        <Label x={gx} y={height - 4}>{`step 1 to ${staticGrid.steps}, one column per step`}</Label>
      </svg>
      <div className="grid grid-cols-2 gap-x-4 gap-y-2 border-t border-border pt-2 tabular-nums sm:grid-cols-4">
        {stats.map((r) => (
          <div key={r.label}>
            <div className="text-muted">{r.label}</div>
            <div className={r.cls}>
              <NumberFlow value={r.value} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
