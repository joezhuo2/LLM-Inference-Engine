"use client";

import NumberFlow from "@/components/number-flow";
import clsx from "clsx";
import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { SmoothRange } from "@/components/smooth-range";
import { glide as tween, type Tween } from "@/lib/glide";
import { usePrefersReducedMotion } from "@/lib/reduced-motion";
import { ceilings, environment, modelShape, profilingTable } from "@/lib/data";
import { EasedBar } from "../eased-bar";
import { Frame, Label } from "../frame";
import { type Op, analyse, crossoverRows, opCost, ops, rowSteps, rowsAt, rowsBetween, stepOf } from "./roofline";

const shape = modelShape();
const peak = ceilings();
const table = profilingTable("roofline", 0, "results");
const column = (name: string) => table.header.indexOf(name);
const measuredRows = [...new Set(table.rows.filter((r) => r[column("op")] === "qkv").map((r) => r[column("rows")]))];

const presets = [1, 16, 64, 512];
const W = 560;
const H = 400;
const pad = { l: 56, r: 20, t: 26, b: 56 };
const xr = [0.5, 1000];
const yr = [0.02, 100];
const lx = (v: number) => pad.l + ((Math.log10(v) - Math.log10(xr[0])) / (Math.log10(xr[1]) - Math.log10(xr[0]))) * (W - pad.l - pad.r);
const ly = (v: number) => H - pad.b - ((Math.log10(v) - Math.log10(yr[0])) / (Math.log10(yr[1]) - Math.log10(yr[0]))) * (H - pad.t - pad.b);
const roof = (intensity: number) => Math.min((intensity * peak.gbs) / 1000, peak.tflops);
const ridge = (peak.tflops * 1e3) / peak.gbs;

const headDim = shape.hidden / shape.heads;
const groupRatio = shape.heads / shape.kv_heads;
const fmt = (v: number) => v.toLocaleString("en-US");
const sig = { maximumSignificantDigits: 3 };
const one = { minimumFractionDigits: 1, maximumFractionDigits: 1 };

const labels: Record<Op, string> = {
  qkv: "QKV projection",
  o_proj: "o_proj",
  gate_up: "gate/up",
  down: "down",
  attention: "attention",
  rmsnorm: "RMSNorm",
  swiglu: "SwiGLU",
  rope: "RoPE",
};

const counted: Record<Op, string> = {
  qkv: `one [${fmt(shape.hidden)} × ${fmt(shape.heads * headDim + 2 * shape.kv_heads * headDim)}] weight matrix per layer, 2 FLOPs per weight per row`,
  o_proj: `one [${fmt(shape.heads * headDim)} × ${fmt(shape.hidden)}] weight matrix per layer, 2 FLOPs per weight per row`,
  gate_up: `one [${fmt(shape.hidden)} × ${fmt(2 * shape.intermediate)}] weight matrix per layer, 2 FLOPs per weight per row`,
  down: `one [${fmt(shape.intermediate)} × ${fmt(shape.hidden)}] weight matrix per layer, 2 FLOPs per weight per row`,
  attention: `each row reads its own K and V cache: ${shape.kv_heads} KV heads × ${headDim} dims × context, 4 FLOPs per query value per cached token`,
  rmsnorm: `two per layer, 5 FLOPs per element (square, add, scale, weight), the row read and written once`,
  swiglu: `5 FLOPs per element of the ${fmt(shape.intermediate)}-wide intermediate (sigmoid, two multiplies), gate and up read, one half-width output written`,
  rope: `3 FLOPs per element of the ${fmt(shape.heads * headDim + shape.kv_heads * headDim)} query and key values, read and written once`,
};

const lessons: Record<Op, string> = {
  qkv: "Every row reuses the same weights, so FLOPs grow with rows while the bytes barely move.",
  o_proj: "Every row reuses the same weights, so FLOPs grow with rows while the bytes barely move.",
  gate_up: "Every row reuses the same weights, so FLOPs grow with rows while the bytes barely move.",
  down: "Every row reuses the same weights, so FLOPs grow with rows while the bytes barely move.",
  attention: `Every row brings its own KV cache, so FLOPs and bytes grow together and batching cannot help. Only the ${groupRatio} query heads sharing each KV head lift it above 1.`,
  rmsnorm: "There is no big weight matrix to share, so more rows add bytes as fast as FLOPs.",
  swiglu: "There is no weight matrix at all, so more rows add bytes as fast as FLOPs.",
  rope: "There is no weight matrix at all, so more rows add bytes as fast as FLOPs.",
};

const flopUnits: [string, number][] = [["FLOP", 1], ["MFLOP", 1e6], ["GFLOP", 1e9], ["TFLOP", 1e12]];
const byteUnits: [string, number][] = [["B", 1], ["KB", 1e3], ["MB", 1e6], ["GB", 1e9]];

function scaled(v: number, units: [string, number][]) {
  const [unit, factor] = [...units].reverse().find(([, f]) => v >= f) ?? units[0];
  return { value: v / factor, unit };
}

function measuredPoint(op: Op, rows: number, context: number) {
  const row = table.rows.find((r) => r[column("op")] === op && Number(r[column("rows")]) === rows);
  if (!row || (op === "attention" && context !== shape.roofline_context)) return null;
  const n = (name: string) => Number(row[column(name)]);
  return { intensity: n("FLOP/byte"), us: n("us/step"), tflops: n("TFLOP/s"), gbs: n("GB/s"), pct: n("% of roof") };
}

type Guess = number | "never" | null;

function verdict(guess: Guess, actual: number | null) {
  if (guess === null) return null;
  if (actual === null) return guess === "never" ? "Right: it never crosses." : `It never crosses, at any batch up to ${rowsAt(rowSteps)}.`;
  if (guess === "never") return `It does cross, at ${actual} rows.`;
  const ratio = guess / actual;
  const off = ratio >= 1 ? `${ratio.toFixed(1)}x too high` : `${(1 / ratio).toFixed(1)}x too low`;
  return `It crosses at ${actual} rows. You said ${guess}: ${Math.abs(ratio - 1) < 0.1 ? "within 10%" : off}.`;
}

const start = { op: "gate_up" as Op, step: 0, context: shape.roofline_context };

export function FoundationsSandbox({ pages }: { pages?: { one: HTMLElement | null; two: HTMLElement | null } } = {}) {
  const [op, setOp] = useState<Op>(start.op);
  const [step, setStep] = useState(start.step);
  const [context, setContext] = useState(start.context);
  const [guess, setGuess] = useState<Guess>(null);
  const reduced = usePrefersReducedMotion();
  const tweens = useRef<Tween[]>([]);
  useEffect(() => () => tweens.current.forEach((t) => t.stop()), []);
  const rows = rowsAt(Math.round(step));
  const cost = opCost(shape, op, rows, context);
  const a = analyse(cost, peak);
  const shown = analyse(opCost(shape, op, rowsBetween(step), context), peak);
  const m = measuredPoint(op, rows, context);
  const crosses = crossoverRows(shape, op, context, peak);
  const far = analyse(opCost(shape, op, rowsAt(rowSteps), context), peak);
  const barMax = Math.max(far.memUs, far.computeUs);
  const flop = scaled(cost.flops, flopUnits);
  const byte = scaled(cost.bytes, byteUnits);
  const dot = { x: lx(shown.intensity), y: ly(shown.tflops) };
  const mdot = m ? { x: lx(m.intensity), y: ly(m.tflops) } : dot;
  const text = verdict(guess, crosses);
  const gpu = String(environment.gpu ?? "").replace("NVIDIA GeForce ", "");
  const stop = () => {
    tweens.current.forEach((t) => t.stop());
    tweens.current = [];
  };
  const glide = (from: number, to: number, set: (v: number) => void, round: (v: number) => number) => {
    if (from === to) return;
    if (reduced) return set(to);
    tweens.current.push(tween(from, to, (x) => set(round(x))));
  };
  const reset = () => {
    stop();
    setOp(start.op);
    setGuess(null);
    glide(step, start.step, setStep, (v) => v);
    glide(context, start.context, setContext, (v) => 16 * Math.round(v / 16));
  };

  const readouts: { label: string; value: React.ReactNode }[] = [
    { label: "FLOPs", value: <NumberFlow value={flop.value} suffix={` ${flop.unit}`} format={sig} /> },
    { label: "bytes moved", value: <NumberFlow value={byte.value} suffix={` ${byte.unit}`} format={sig} /> },
    { label: "FLOP per byte", value: <NumberFlow value={a.intensity} format={sig} /> },
    { label: "ridge", value: `${ridge.toFixed(1)} FLOP/byte` },
    { label: "bound by", value: <span className="text-accent">{a.bound === "memory" ? "memory bandwidth" : "compute"}</span> },
    { label: "time at the ceiling", value: <NumberFlow value={a.timeUs} suffix=" µs" format={one} /> },
  ];

  const partOne = (
      <div className="space-y-4">
        <div className="space-y-2 rounded border border-border p-3 leading-relaxed">
          <div className="text-accent">predict first</div>
          <p className="text-text">
            {crosses === null
              ? `Will ${labels[op]} ever reach the ridge as the batch grows to ${rowsAt(rowSteps)} rows? Guess, then drag the batch and watch the dot.`
              : `At how many rows does ${labels[op]} cross the ridge? Set the batch where you think it turns compute bound, lock it in, then drag to check.`}
          </p>
          <div className="flex flex-wrap items-center gap-2">
            <button type="button" onClick={() => setGuess(rows)} className="pressable rounded border border-border px-2.5 py-1 text-text">
              {`lock in ${fmt(rows)} row${rows > 1 ? "s" : ""}`}
            </button>
            <button type="button" onClick={() => setGuess("never")} className="pressable rounded border border-border px-2.5 py-1 text-text">
              never crosses
            </button>
            {guess !== null && <span className="text-muted">{`your guess: ${guess === "never" ? "never" : `${fmt(guess)} rows`}`}</span>}
          </div>
          {text && (
            <p className="text-text" aria-live="polite">
              {text} <span className="text-muted">{lessons[op]}</span>
            </p>
          )}
        </div>

        <div role="group" aria-label="operation" className="flex flex-wrap items-center gap-2">
          <span className="text-muted">operation</span>
          {ops.map((o) => (
            <button
              key={o}
              type="button"
              aria-pressed={op === o}
              onClick={() => {
                setOp(o);
                setGuess(null);
              }}
              className={clsx("pressable rounded border px-2.5 py-1", op === o ? "border-accent bg-accent-dim text-accent" : "border-border text-text")}
            >
              {labels[o]}
            </button>
          ))}
        </div>

        <div className="grid gap-x-4 gap-y-2 sm:grid-cols-[auto_minmax(0,1fr)_5.5rem] sm:items-center">
          <label htmlFor="roofline-rows" className="text-muted">
            batch rows
          </label>
          <SmoothRange
            id="roofline-rows"
            min={0}
            max={rowSteps}
            value={step}
            continuous
            onValue={setStep}
            onPointerDown={stop}
            onKeyDown={stop}
            aria-valuetext={`${rows} rows`}
            className="w-full accent-[var(--accent)]"
          />
          <span className="text-right text-text tabular-nums">
            <NumberFlow value={rows} suffix={rows === 1 ? " row" : " rows"} />
          </span>
          <span className="text-muted">jump to</span>
          <div className="flex flex-wrap gap-2 sm:col-span-2">
            {presets.map((p) => (
              <button
                key={p}
                type="button"
                aria-pressed={rows === p}
                onClick={() => {
                  stop();
                  glide(step, stepOf(p), setStep, (v) => v);
                }}
                title={measuredRows.includes(String(p)) ? "has a measured point" : undefined}
                className={clsx("pressable rounded border px-2.5 py-1 tabular-nums", rows === p ? "border-accent bg-accent-dim text-accent" : "border-border text-text")}
              >
                {p}
              </button>
            ))}
          </div>
          <label htmlFor="roofline-context" className={clsx(op === "attention" ? "text-muted" : "text-muted/50")}>
            context
          </label>
          <SmoothRange
            id="roofline-context"
            min={16}
            max={shape.max_positions}
            step={16}
            value={context}
            disabled={op !== "attention"}
            onValue={setContext}
            onPointerDown={stop}
            onKeyDown={stop}
            aria-valuetext={`${context} tokens`}
            className="w-full accent-[var(--accent)] disabled:opacity-40"
          />
          <span className={clsx("text-right tabular-nums", op === "attention" ? "text-text" : "text-muted/50")}>
            <NumberFlow value={context} suffix=" tok" />
          </span>
        </div>

        <div className="aspect-[14/10]">
          <Frame label={`Log-log roofline of the ${labels[op]} op at ${rows} rows${op === "attention" ? ` and ${context} tokens of context` : ""}: ${a.intensity.toFixed(1)} FLOP per byte, ${a.bound} bound, ${a.tflops.toFixed(2)} TFLOP/s at the ceilings${m ? `, measured ${m.tflops.toFixed(2)} TFLOP/s` : ""}`}>
            {[0.1, 1, 10, 100].map((v) => (
              <g key={v}>
                <line x1={pad.l} x2={W - pad.r} y1={ly(v)} y2={ly(v)} className="stroke-border" strokeWidth={0.5} />
                <Label x={pad.l - 6} y={ly(v) + 3} anchor="end">{String(v)}</Label>
              </g>
            ))}
            {[1, 10, 100, 1000].map((v) => (
              <Label key={v} x={lx(v)} y={H - pad.b + 16} anchor="middle">{String(v)}</Label>
            ))}
            <Label x={W - pad.r} y={H - pad.b + 34} anchor="end">FLOP per byte of DRAM traffic</Label>
            <Label x={pad.l} y={pad.t - 10}>TFLOP/s per decode step</Label>
            <line x1={lx(ridge)} x2={lx(ridge)} y1={pad.t} y2={H - pad.b} className="stroke-border" strokeDasharray="3 3" />
            <path d={`M ${lx(xr[0])} ${ly(roof(xr[0]))} L ${lx(ridge)} ${ly(peak.tflops)} L ${lx(xr[1])} ${ly(peak.tflops)}`} className="fill-none stroke-text" strokeWidth={1.25} />
            <Label x={lx(1.1)} y={ly(roof(1.1)) + 22} className="fill-text">{`${peak.gbs} GB/s`}</Label>
            <Label x={lx(ridge) + 8} y={ly(peak.tflops) - 8} className="fill-text">{`${peak.tflops} TFLOP/s`}</Label>
            <Label x={lx(ridge)} y={H - pad.b + 50} anchor="middle" className="fill-text">{`ridge ${ridge.toFixed(0)}`}</Label>
            {guess !== null && guess !== "never" && (
              <g>
                <line x1={lx(analyse(opCost(shape, op, guess, context), peak).intensity)} x2={lx(analyse(opCost(shape, op, guess, context), peak).intensity)} y1={pad.t} y2={H - pad.b} className="stroke-warn" strokeDasharray="2 2" />
                <Label x={lx(analyse(opCost(shape, op, guess, context), peak).intensity) + 4} y={pad.t + 10} className="fill-warn">your guess</Label>
              </g>
            )}
            <circle r={4} className="dot-move fill-accent" style={{ transform: `translate(${mdot.x}px, ${mdot.y}px)`, opacity: m ? 0.9 : 0 }} />
            <circle r={7} className="dot-move fill-none stroke-text" strokeWidth={1.5} style={{ transform: `translate(${dot.x}px, ${dot.y}px)` }} />
          </Frame>
        </div>
        <div className="flex flex-wrap gap-x-4 gap-y-1 text-muted">
          <span className="flex items-center gap-1.5">
            <span aria-hidden className="size-2 rounded-full border border-text" />
            computed from the ceilings
          </span>
          <span className="flex items-center gap-1.5">
            <span aria-hidden className="size-2 rounded-full bg-accent" />
            {`measured on the ${gpu}`}
          </span>
        </div>
      </div>
  );
  const partTwo = (
      <div className="space-y-5">
        <div className="flex items-center justify-between gap-2">
          <span className="text-text">{`${labels[op]}, ${rows} row${rows > 1 ? "s" : ""}`}</span>
          <button type="button" onClick={reset} className="pressable rounded border border-border px-2.5 py-1 text-text">
            reset
          </button>
        </div>
        <p className="leading-relaxed text-muted">{`Counted per decode step, all ${shape.layers} layers: ${counted[op]}.`}</p>

        <table className="w-full tabular-nums">
          <thead className="text-left text-muted">
            <tr>
              <th colSpan={2} className="pb-1 font-normal">{`computed from the ceilings: ${peak.gbs} GB/s, ${peak.tflops} TFLOP/s`}</th>
            </tr>
          </thead>
          <tbody>
            {readouts.map((r) => (
              <tr key={r.label} className="border-t border-border">
                <td className="py-1 text-muted">{r.label}</td>
                <td className="py-1 text-right text-text">{r.value}</td>
              </tr>
            ))}
          </tbody>
        </table>

        <div className="space-y-1.5">
          <div className="text-muted">time if limited by only one thing</div>
          {[
            { label: "memory", us: a.memUs, bar: shown.memUs, bound: a.bound === "memory" },
            { label: "compute", us: a.computeUs, bar: shown.computeUs, bound: a.bound === "compute" },
          ].map((b) => (
            <div key={b.label} className="grid grid-cols-[4.5rem_1fr_5.5rem] items-center gap-2">
              <span className={b.bound ? "text-accent" : "text-muted"}>{b.label}</span>
              <div className="h-3 min-w-0 rounded-full bg-border">
                <EasedBar fraction={b.bar / barMax} className={b.bound ? "bg-accent" : "bg-baseline/70"} />
              </div>
              <span className="text-right text-muted">
                <NumberFlow value={b.us} suffix=" µs" format={one} />
              </span>
            </div>
          ))}
        </div>

        <div>
          <div className="pb-1 text-muted">{`measured: ${gpu}, commit ${environment.commit}, ${String(environment.date).slice(0, 10)}`}</div>
          <div className="grid">
            <table className={clsx("col-start-1 row-start-1 w-full tabular-nums", !m && "invisible")} aria-hidden={!m}>
              <tbody>
                {[
                  { label: "time", value: <NumberFlow value={m?.us ?? 0} suffix=" µs" format={one} /> },
                  { label: "of the roof", value: <NumberFlow value={m?.pct ?? 0} suffix="%" /> },
                  { label: "bandwidth", value: <NumberFlow value={m?.gbs ?? 0} suffix=" GB/s" /> },
                  { label: "compute", value: <NumberFlow value={m?.tflops ?? 0} suffix=" TFLOP/s" format={{ maximumFractionDigits: 2 }} /> },
                ].map((r) => (
                  <tr key={r.label} className="border-t border-border">
                    <td className="py-1 text-muted">{r.label}</td>
                    <td className="py-1 text-right text-text">{r.value}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {!m && (
              <p className="col-start-1 row-start-1 self-start border-t border-border py-1 leading-relaxed text-muted">
                {`no measured point here: ${op === "attention" ? `attention was measured at ${shape.roofline_context} tokens of context and ` : ""}the profiles cover ${measuredRows.join(" and ")} rows${["rmsnorm", "swiglu", "rope"].includes(op) ? ", and not this op" : ""}. Only the computed dot is drawn.`}
              </p>
            )}
          </div>
        </div>
      </div>
  );
  if (pages) {
    return (
      <>
        {pages.one && createPortal(<div className="tabular-nums text-xs">{partOne}</div>, pages.one)}
        {pages.two && createPortal(<div className="tabular-nums text-xs">{partTwo}</div>, pages.two)}
      </>
    );
  }
  return (
    <div className="grid gap-6 tabular-nums text-xs lg:grid-cols-[minmax(0,1fr)_19rem]">
      {partOne}
      {partTwo}
    </div>
  );
}
