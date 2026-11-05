"use client";

import NumberFlow from "@number-flow/react";
import clsx from "clsx";
import { useCallback, useEffect, useRef, useState } from "react";
import { SmoothRange } from "@/components/smooth-range";
import { glide, type Tween } from "@/lib/glide";
import { usePrefersReducedMotion } from "@/lib/reduced-motion";
import { prefillMs } from "@/content/kernels/chunked-prefill";
import { EasedBar } from "../eased-bar";
import { Toggle } from "../toggle";
import { type Cell, decoders, type Lane, maxPrompt, type Mode, Playground, type PoolSize, predict, type Run } from "./scheduler-model";

const budgets = [8, 16, 32, 64, 128, 256, 512, 1024, 2048];
const defaultBudget = 5;
const modes: Mode[] = ["schedule()", "schedule_chunked()"];
const pools: PoolSize[] = ["roomy", "tight"];
const quick = [128, 512, 2048];
const tickMs = 260;
const guessSteps = ["1", "4", "9", "16"];
const guessStall = ["0 steps", "1 step", "9 steps"];
const aloneSmall = prefillMs("engine", "128");
const aloneLarge = prefillMs("engine", "2,048");

const fmt = (v: number) => v.toLocaleString("en-US");
const promptTint = (id: number) => (id - decoders) % 2 === 0 ? "bg-series-2" : "bg-series-3";
const cellClass = (cell: Cell, lane: Lane) => {
  switch (cell) {
    case "decode":
      return "bg-accent";
    case "stall":
      return "bg-warn";
    case "chunk":
      return clsx(lane.kind === "prompt" ? promptTint(lane.id) : "bg-warn", "opacity-60");
    case "first":
      return clsx(promptTint(lane.id), "ring-1 ring-text");
    case "preempt":
      return "border border-warn text-warn";
    default:
      return "border border-dashed border-border";
  }
};

const cellName: Record<Cell, string> = {
  decode: "decode row",
  stall: "stalled, no row",
  chunk: "prefill chunk",
  first: "prefill done, first token",
  wait: "waiting",
  preempt: "preempted, blocks freed",
};

function status(run: Run, lane: Lane) {
  const r = run.sched.requests.get(lane.id)!;
  if (r.finished) return { text: "finished", warn: false };
  if (!r.running) return { text: r.output > 0 ? "preempted, waiting to replay" : "waiting for blocks or a slot", warn: r.output > 0 };
  const done = run.sched.progress.get(lane.id);
  if (done !== undefined) return { text: `prefill ${fmt(done)} / ${fmt(r.prompt + r.output)}`, warn: false };
  return { text: `decoding, ${fmt(r.prompt + r.output)} tokens`, warn: false };
}

function useColumns(ref: React.RefObject<HTMLDivElement | null>) {
  const [columns, setColumns] = useState(24);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const resize = new ResizeObserver(() => setColumns(el.clientWidth < 520 ? 14 : 28));
    resize.observe(el);
    return () => resize.disconnect();
  }, [ref]);
  return columns;
}

function Lanes({ run, budget, columns }: { run: Run; budget: number; columns: number }) {
  const n = run.steps.length;
  const start = n > 0 ? Math.floor((n - 1) / columns) * columns : 0;
  const idx = Array.from({ length: columns }, (_, i) => start + i);
  const shownMax = Math.max(1, ...run.steps.slice(start, start + columns).map((s) => s.tokens));
  const top = run.mode === "schedule_chunked()" ? Math.max(budget, shownMax) : shownMax;
  const grid = { gridTemplateColumns: `2.25rem repeat(${columns}, minmax(0, 1fr))` };
  return (
    <div className="space-y-1">
      <div className="flex items-baseline justify-between text-muted">
        <span>tokens in each step: one column per step, full height is</span>
        <span className="tabular-nums text-text">
          <NumberFlow value={top} />
        </span>
      </div>
      <div className="relative grid gap-x-[2px]" style={grid}>
        {n === 0 && <p className="pointer-events-none absolute inset-x-0 top-8 text-center text-muted">no steps yet: drop a prompt and press play</p>}
        <div />
        {idx.map((i) => {
          const s = run.steps[i];
          const layers: { frac: number; cls: string }[] = [];
          if (s) {
            let sum = s.decode;
            layers.push({ frac: sum / top, cls: "bg-accent" });
            for (const c of s.slices) {
              sum += c.count;
              layers.push({ frac: sum / top, cls: promptTint(c.id) });
            }
          }
          return (
            <div key={i} className="relative h-24 border-b border-border">
              {layers.map((l, k) => (
                <div
                  key={k}
                  className={clsx("absolute inset-0 origin-bottom transition-transform duration-700 ease-[cubic-bezier(0.65,0,0.35,1)] motion-reduce:transition-none", l.cls)}
                  style={{ transform: `scaleY(${Math.max(0.03, Math.min(1, l.frac))})`, zIndex: layers.length - k }}
                />
              ))}
            </div>
          );
        })}
        <div className="pt-0.5 text-[10px] text-muted">step</div>
        {idx.map((i) => (
          <div key={i} className="pt-0.5 text-center text-[10px] tabular-nums text-muted">
            {i < n && (i + 1) % 4 === 0 ? i + 1 : ""}
          </div>
        ))}
        {run.lanes.map((lane) => (
          <Row key={lane.id} lane={lane} idx={idx} />
        ))}
      </div>
    </div>
  );
}

function Row({ lane, idx }: { lane: Lane; idx: number[] }) {
  return (
    <>
      <div className={clsx("flex h-4 items-center text-[11px]", lane.kind === "prompt" ? "text-text" : "text-muted")}>{lane.label}</div>
      {idx.map((i) => {
        const cell = i >= lane.from ? lane.cells[i - lane.from] : undefined;
        return (
          <div key={i} className="flex h-4 items-center justify-center">
            {cell && (
              <div title={`${lane.label}, step ${i + 1}: ${cellName[cell]}`} className={clsx("flex h-3 w-full items-center justify-center rounded-[2px] text-[9px] leading-none transition-opacity duration-200", cellClass(cell, lane))}>
                {cell === "preempt" ? "×" : ""}
              </div>
            )}
          </div>
        );
      })}
    </>
  );
}

function Legend() {
  const items: [string, string][] = [
    ["bg-accent", "decode row"],
    ["bg-warn", "decoder stalled: no row this step"],
    ["bg-series-2 opacity-60", "prefill chunk"],
    ["bg-series-2 ring-1 ring-text", "chunk that ends the prompt: first token"],
    ["border border-warn", "preempted"],
    ["border border-dashed border-border", "waiting"],
  ];
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1 text-muted">
      {items.map(([cls, text]) => (
        <span key={text} className="flex items-center gap-1.5">
          <span aria-hidden className={clsx("size-2 rounded-sm", cls)} />
          {text}
        </span>
      ))}
    </div>
  );
}

type Readout = { label: string; value: (r: Run) => number | null; suffix?: (r: Run) => string; digits?: number; warn?: (v: number) => boolean };

const rows: Readout[] = [
  { label: "steps run", value: (r) => r.stats().steps },
  { label: "longest decode stall, steps", value: (r) => r.stats().longestStall, warn: (v) => v > 0 },
  { label: "biggest step, tokens", value: (r) => r.stats().biggest },
  { label: "tokens per step, mean", value: (r) => r.stats().perStep, digits: 1 },
  { label: "latest prompt: steps to first token", value: (r) => r.stats().ttft?.steps ?? null, suffix: (r) => (r.stats().ttft && !r.stats().ttft!.done ? "+" : "") },
  { label: "preemptions", value: (r) => r.stats().preemptions, warn: (v) => v > 0 },
];

export function ChunkedPrefillSandbox() {
  const [pool, setPool] = useState<PoolSize>("roomy");
  const [pg, setPg] = useState(() => new Playground("roomy"));
  const [, setVersion] = useState(0);
  const [mode, setMode] = useState<Mode>("schedule_chunked()");
  const [bi, setBi] = useState(defaultBudget);
  const [length, setLength] = useState(2048);
  const [playing, setPlaying] = useState(false);
  const [guess, setGuess] = useState<{ steps: string; stall: string } | null>(null);
  const [guessed, setGuessed] = useState({ steps: "", stall: "" });
  const wrapRef = useRef<HTMLDivElement>(null);
  const columns = useColumns(wrapRef);
  const budget = budgets[bi];
  const bump = useCallback(() => setVersion((v) => v + 1), []);

  useEffect(() => {
    if (!playing) return;
    const timer = setInterval(() => {
      pg.advance(budget);
      bump();
      if (pg.settled()) setPlaying(false);
    }, tickMs);
    return () => clearInterval(timer);
  }, [playing, pg, bump, budget]);

  const fresh = (next: PoolSize) => {
    setPg(new Playground(next));
    setPlaying(false);
  };

  const drop = (n: number) => {
    pg.drop(n);
    bump();
    setPlaying(true);
  };

  const toFirstToken = () => {
    for (let i = 0; i < 700 && !pg.settled(); i++) pg.advance(budget);
    setPlaying(false);
    bump();
  };

  const prediction = guess ? predict("roomy", 2048, 256) : null;
  const reduced = usePrefersReducedMotion();
  const tweens = useRef<Tween[]>([]);
  const cancelTween = () => {
    tweens.current.forEach((t) => t.stop());
    tweens.current = [];
  };
  const glideTo = (from: number, to: number, set: (v: number) => void, round: (v: number) => number, done?: () => void) => {
    if (reduced || from === to) {
      set(to);
      return done?.();
    }
    tweens.current.push(glide(from, to, (x) => set(round(x)), () => {
        set(to);
        done?.();
      }));
  };
  const tryIt = () => {
    setGuess(guessed);
    setPool("roomy");
    cancelTween();
    glideTo(bi, budgets.indexOf(256), setBi, Math.round, () => setPlaying(true));
    glideTo(length, 2048, setLength, (v) => Math.round(v / 16) * 16);
    setMode("schedule_chunked()");
    const next = new Playground("roomy");
    next.drop(2048);
    setPg(next);
    setPlaying(false);
  };

  const run = pg.runs[mode];
  const busy = !pg.settled();
  const last = (m: Mode) => pg.runs[m].steps[pg.runs[m].steps.length - 1]?.tokens ?? 0;
  const summary = `${mode}, step ${run.steps.length}: longest decode stall ${run.stats().longestStall} steps, biggest step ${run.stats().biggest} tokens`;

  return (
    <div className="grid gap-6 font-mono text-xs lg:grid-cols-[minmax(0,1fr)_19rem]">
      <div className="min-w-0 space-y-4">
        <div className="space-y-2 rounded border border-border p-3">
          <div className="text-text">predict first</div>
          <p className="text-muted">
            Four sequences are decoding. You drop a 2,048-token prompt and the engine runs schedule_chunked() with a budget of 256 tokens. How many steps until the prompt gets its first token? Under schedule(), how many steps do the decoders go without a row?
          </p>
          <div className="flex flex-wrap items-center gap-x-6 gap-y-2">
            <Toggle label="first token" choices={guessSteps} value={guessed.steps} onChange={(v) => setGuessed({ ...guessed, steps: v })} />
            <Toggle label="stall" choices={guessStall} value={guessed.stall} onChange={(v) => setGuessed({ ...guessed, stall: v })} />
            <button type="button" onClick={tryIt} className="pressable rounded border border-border px-2.5 py-1 text-accent">
              run it
            </button>
          </div>
          {guess && prediction && (
            <p aria-live="polite" className="text-muted">
              <span className="text-text">{`schedule_chunked() at budget 256: first token after ${prediction.chunked.ttft!.steps} steps, decoders never stall; schedule(): first token after ${prediction.legacy.ttft!.steps} step, decoders stall ${prediction.legacy.longestStall} step.`}</span>
              {` ${guess.steps ? (guess.steps === String(prediction.chunked.ttft!.steps) ? "Your first-token guess was right." : `You guessed ${guess.steps} for the first token.`) : "You gave no first-token guess."} ${guess.stall ? (guess.stall.startsWith(`${prediction.legacy.longestStall} `) ? "Your stall guess was right." : `You guessed ${guess.stall} for the stall.`) : "You gave no stall guess."} The catch is the size of that one schedule() step: ${fmt(prediction.legacy.biggest)} tokens against at most 256 per step with the budget. Compare the two columns on the right.`}
            </p>
          )}
        </div>

        <div className="space-y-3">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-muted">drop a prompt</span>
            {quick.map((q) => (
              <button key={q} type="button" onClick={() => drop(q)} className="pressable rounded border border-border px-2.5 py-1 text-text">
                {`+${fmt(q)}`}
              </button>
            ))}
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <label htmlFor="cp-length" className="w-14 shrink-0 text-muted">
              length
            </label>
            <SmoothRange id="cp-length" min={16} max={maxPrompt} step={16} value={length} onValue={(v) => { cancelTween(); setLength(v); }} onKeyDown={cancelTween} aria-valuetext={`${length} tokens`} className="min-w-32 flex-1 accent-[var(--accent)]" />
            <span className="w-14 text-right tabular-nums text-text">
              <NumberFlow value={length} />
            </span>
            <button type="button" onClick={() => drop(length)} className="pressable rounded border border-border px-2.5 py-1 text-text">
              drop it
            </button>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <label htmlFor="cp-budget" className="w-14 shrink-0 text-muted">
              budget
            </label>
            <SmoothRange id="cp-budget" min={0} max={budgets.length - 1} step={1} value={bi} onValue={(v) => { cancelTween(); setBi(v); }} onKeyDown={cancelTween} disabled={mode === "schedule()"} aria-valuetext={`${budget} tokens per step`} className="min-w-32 flex-1 accent-[var(--accent)] disabled:opacity-40" />
            <span className={clsx("w-14 text-right tabular-nums", mode === "schedule()" ? "text-muted" : "text-text")}>
              <NumberFlow value={budget} />
            </span>
            <span className="text-muted">tokens per step</span>
          </div>
          <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
            <Toggle label="rule" choices={modes} value={mode} onChange={(v) => setMode(v as Mode)} />
            <Toggle
              label="KV pool"
              choices={pools}
              value={pool}
              onChange={(v) => {
                setPool(v as PoolSize);
                fresh(v as PoolSize);
              }}
            />
          </div>
          <div className="flex flex-wrap gap-2">
            <button type="button" disabled={!busy} onClick={() => setPlaying(!playing)} aria-pressed={playing} className="pressable rounded border border-border px-2.5 py-1 text-accent disabled:opacity-40">
              {playing ? "pause" : "play"}
            </button>
            <button
              type="button"
              disabled={playing}
              onClick={() => {
                pg.advance(budget);
                bump();
              }}
              className="pressable rounded border border-border px-2.5 py-1 text-text disabled:opacity-40"
            >
              step
            </button>
            <button type="button" disabled={!busy} onClick={toFirstToken} className="pressable rounded border border-border px-2.5 py-1 text-text disabled:opacity-40">
              run to first token
            </button>
            <button type="button" onClick={() => fresh(pool)} className="pressable rounded border border-border px-2.5 py-1 text-text">
              reset
            </button>
          </div>
        </div>

        <div ref={wrapRef} className="space-y-3">
          <Lanes run={run} budget={budget} columns={columns} />
          <Legend />
        </div>
        <p role="status" className="sr-only">
          {summary}
        </p>
        <p className="max-w-2xl text-muted">
          {`For scale, measured on this engine (docs/results.md): prefilling 128 tokens alone takes ${fmt(aloneSmall)} ms and 2,048 tokens ${fmt(aloneLarge)} ms. A step's cost grows with its tokens, so one 2,048-token step is the long wait the decoders feel; the lanes count steps and tokens, not milliseconds.`}
        </p>
      </div>

      <div className="min-w-0 space-y-5">
        <table className="w-full tabular-nums">
          <thead className="text-left text-muted">
            <tr>
              <th className="pb-1 font-normal">{`step ${run.steps.length}`}</th>
              {modes.map((m) => (
                <th key={m} className={clsx("pb-1 text-right font-normal", m === mode ? "text-accent" : "text-muted")}>
                  {m.replace("schedule_chunked()", "chunked").replace("schedule()", "legacy")}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.label} className="border-t border-border">
                <td className="py-1 pr-2 text-muted">{r.label}</td>
                {modes.map((m) => {
                  const v = r.value(pg.runs[m]);
                  return (
                    <td key={m} className={clsx("py-1 text-right", v !== null && r.warn?.(v) ? "text-warn" : m === mode ? "text-text" : "text-muted")}>
                      {v === null ? "none yet" : <NumberFlow value={v} suffix={r.suffix?.(pg.runs[m])} format={{ minimumFractionDigits: r.digits ?? 0, maximumFractionDigits: r.digits ?? 0 }} />}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
        <p className="text-muted">a + after the first-token count means the prompt is still waiting</p>

        <div className="space-y-2">
          <div className="text-muted">latest step, tokens (full bar is {fmt(maxPrompt)})</div>
          {modes.map((m) => (
            <div key={m} className="space-y-1">
              <div className="flex justify-between">
                <span className={m === mode ? "text-text" : "text-muted"}>{m}</span>
                <span className="tabular-nums text-text">
                  <NumberFlow value={last(m)} />
                </span>
              </div>
              <div className="h-1.5 rounded-full bg-border">
                <EasedBar fraction={last(m) / maxPrompt} className={m === mode ? "bg-accent" : "bg-muted"} />
              </div>
            </div>
          ))}
        </div>

        <div className="space-y-2">
          <div className="text-muted">{`sequences under ${mode}`}</div>
          <ul className="max-h-64 space-y-1 overflow-y-auto pr-1">
            {run.lanes.map((lane) => {
              const s = status(run, lane);
              return (
                <li key={lane.id} className="flex items-baseline gap-1.5">
                  <span aria-hidden className={clsx("size-2 shrink-0 rounded-sm", lane.kind === "prompt" ? promptTint(lane.id) : "bg-accent")} />
                  <span className="text-text">{lane.kind === "prompt" ? `${lane.label} ${fmt(lane.prompt)}` : lane.label}</span>
                  <span className={s.warn ? "text-warn" : "text-muted"}>{s.text}</span>
                </li>
              );
            })}
          </ul>
          <p className="text-muted">{`${run.sched.blocks.held()} of ${run.sched.config.numBlocks - 1} KV blocks held, ${run.sched.waiting.length} waiting`}</p>
        </div>
      </div>
    </div>
  );
}
