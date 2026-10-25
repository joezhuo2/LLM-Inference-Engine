"use client";

import NumberFlow from "@number-flow/react";
import { Liveline, type LivelinePoint } from "liveline";
import { useTheme } from "next-themes";
import { type ReactNode, useEffect, useMemo, useState } from "react";
import { loopTime, type Snapshot, snapshotAt, stepsDoneBy, type Trace, traceDuration } from "@/lib/trace";

type History = { snap: Snapshot; points: Record<Key, LivelinePoint[]> };
type Key = "tps" | "running" | "kv" | "ttft" | "tpot";

const keys: Key[] = ["tps", "running", "kv", "ttft", "tpot"];

function useReplay(trace: Trace): History {
  const duration = traceDuration(trace);
  const [history, setHistory] = useState<History>(() => ({
    snap: snapshotAt(trace, 0),
    points: { tps: [], running: [], kv: [], ttft: [], tpot: [] },
  }));
  useEffect(() => {
    const start = performance.now();
    const id = window.setInterval(() => {
      const snap = snapshotAt(trace, loopTime((performance.now() - start) / 1000, duration));
      const now = Date.now() / 1000;
      const values: Record<Key, number> = {
        tps: snap.tokensPerS,
        running: snap.running,
        kv: snap.kvUsed,
        ttft: (snap.ttftP50 ?? 0) * 1e3,
        tpot: (snap.tpotP50 ?? 0) * 1e3,
      };
      setHistory((h) => {
        const points = { ...h.points };
        for (const k of keys) points[k] = [...h.points[k], { time: now, value: values[k] }].slice(-150);
        return { snap, points };
      });
    }, 100);
    return () => window.clearInterval(id);
  }, [trace, duration]);
  return history;
}

function useMaxima(trace: Trace) {
  return useMemo(() => {
    let tps = 1;
    let kv = 1;
    for (let t = 0; t <= traceDuration(trace); t += 0.25) {
      const s = snapshotAt(trace, t);
      tps = Math.max(tps, s.tokensPerS);
      kv = Math.max(kv, s.kvUsed);
    }
    return { tps, kv };
  }, [trace]);
}

function Timeline({ trace, t }: { trace: Trace; t: number }) {
  const done = stepsDoneBy(trace, t);
  const steps = trace.steps.slice(Math.max(0, done - 120), done);
  const max = Number(trace.meta.max_batch ?? 16);
  return (
    <section className="rounded border border-border bg-surface p-4">
      <div className="mb-3 font-mono text-xs text-muted">batch per step, last 120 steps · amber prefill, green decode</div>
      <div className="flex h-24 items-end gap-px" aria-hidden="true">
        {steps.map((s) => (
          <div
            key={s.index}
            className={s.kind === "prefill" ? "flex-1 bg-warn" : "flex-1 bg-accent/70"}
            style={{ height: `${(100 * s.seqs.length) / max}%` }}
          />
        ))}
      </div>
    </section>
  );
}

function Dashboard({ trace, history, children }: { trace: Trace; history: History; children: ReactNode }) {
  return (
    <div className="space-y-4">
      <div className="flex items-baseline justify-between gap-4">
        <h1 className="font-mono text-2xl">dashboard</h1>
        <span className="font-mono text-xs text-muted">replaying a recorded run · {String(trace.meta.gpu ?? "")}</span>
      </div>
      {children}
      <Timeline trace={trace} t={history.snap.t} />
    </div>
  );
}

function ms(v: number | undefined) {
  return v === undefined ? 0 : v * 1e3;
}

export function Readout({ trace }: { trace: Trace }) {
  const history = useReplay(trace);
  const s = history.snap;
  const tiles = [
    { label: "tokens / s", value: s.tokensPerS, suffix: "", digits: 0 },
    { label: `running of ${s.maxBatch}`, value: s.running, suffix: "", digits: 0, extra: `${s.waiting} waiting` },
    { label: "KV blocks used", value: s.kvUsed, suffix: "", digits: 0, extra: `of ${s.kvTotal}` },
    { label: "TTFT p50", value: ms(s.ttftP50), suffix: " ms", digits: 1 },
    { label: "TPOT p50", value: ms(s.tpotP50), suffix: " ms", digits: 2 },
  ];
  return (
    <Dashboard trace={trace} history={history}>
      <div className="grid grid-cols-2 gap-px overflow-hidden rounded border border-border bg-border sm:grid-cols-5">
        {tiles.map((tile, i) => (
          <div key={tile.label} className={i === 0 ? "col-span-2 bg-surface p-4 sm:col-span-1" : "bg-surface p-4"}>
            <div className="font-mono text-xs text-muted">{tile.label}</div>
            <div className={i === 0 ? "font-mono text-4xl text-accent" : "font-mono text-3xl"}>
              <NumberFlow
                value={tile.value}
                suffix={tile.suffix}
                format={{ maximumFractionDigits: tile.digits, minimumFractionDigits: tile.digits }}
                className="tabular-nums"
              />
            </div>
            {tile.extra && <div className="font-mono text-xs text-muted">{tile.extra}</div>}
          </div>
        ))}
      </div>
    </Dashboard>
  );
}

export function Spark({ trace }: { trace: Trace }) {
  const history = useReplay(trace);
  const { resolvedTheme } = useTheme();
  const theme = resolvedTheme === "light" ? "light" : "dark";
  const color = theme === "light" ? "#15803d" : "#4ade80";
  const s = history.snap;
  const tiles: { key: Key; label: string; value: number; suffix: string; digits: number }[] = [
    { key: "tps", label: "tokens / s", value: s.tokensPerS, suffix: "", digits: 0 },
    { key: "running", label: `running of ${s.maxBatch}`, value: s.running, suffix: "", digits: 0 },
    { key: "kv", label: "KV blocks used", value: s.kvUsed, suffix: "", digits: 0 },
    { key: "ttft", label: "TTFT p50", value: ms(s.ttftP50), suffix: " ms", digits: 1 },
    { key: "tpot", label: "TPOT p50", value: ms(s.tpotP50), suffix: " ms", digits: 2 },
  ];
  return (
    <Dashboard trace={trace} history={history}>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-5">
        {tiles.map((tile) => (
          <div key={tile.key} className="rounded border border-border bg-surface p-3">
            <div className="flex items-baseline justify-between gap-2">
              <span className="font-mono text-xs text-muted">{tile.label}</span>
              <NumberFlow
                value={tile.value}
                suffix={tile.suffix}
                format={{ maximumFractionDigits: tile.digits, minimumFractionDigits: tile.digits }}
                className="font-mono text-lg tabular-nums"
              />
            </div>
            <div className="mt-2 h-14">
              <Liveline
                data={history.points[tile.key]}
                value={history.points[tile.key].at(-1)?.value ?? 0}
                theme={theme}
                color={color}
                window={10}
                grid={false}
                badge={false}
                scrub={false}
                pulse={false}
                momentum={false}
                lineWidth={1.5}
                padding={{ top: 4, right: 4, bottom: 4, left: 4 }}
              />
            </div>
          </div>
        ))}
      </div>
    </Dashboard>
  );
}

function Bar({ fraction, tone = "accent" }: { fraction: number; tone?: "accent" | "warn" }) {
  return (
    <div className="h-2 overflow-hidden rounded-sm bg-border">
      <div
        className={tone === "accent" ? "h-full origin-left bg-accent" : "h-full origin-left bg-warn"}
        style={{ transform: `scaleX(${Math.max(0, Math.min(1, fraction))})`, transition: "transform 200ms var(--ease-out)" }}
      />
    </div>
  );
}

export function Gauge({ trace }: { trace: Trace }) {
  const history = useReplay(trace);
  const max = useMaxima(trace);
  const s = history.snap;
  return (
    <Dashboard trace={trace} history={history}>
      <div className="grid gap-3 sm:grid-cols-2">
        <div className="rounded border border-border bg-surface p-4 sm:col-span-2">
          <div className="flex items-baseline justify-between font-mono text-xs text-muted">
            <span>batch slots</span>
            <span className="tabular-nums">
              {s.running} running · {s.waiting} waiting
            </span>
          </div>
          <div className="mt-3 grid grid-cols-8 gap-1 sm:grid-cols-16" aria-hidden="true">
            {Array.from({ length: s.maxBatch }, (_, i) => (
              <div
                key={i}
                className={
                  i < s.running
                    ? "h-6 rounded-sm bg-accent"
                    : i < s.running + s.waiting
                      ? "h-6 rounded-sm border border-dashed border-warn"
                      : "h-6 rounded-sm border border-border"
                }
              />
            ))}
          </div>
        </div>
        {[
          { label: "tokens / s", value: s.tokensPerS.toFixed(0), fraction: s.tokensPerS / max.tps, note: `peak ${max.tps}` },
          { label: "KV blocks used", value: String(s.kvUsed), fraction: s.kvUsed / max.kv, note: `peak ${max.kv} of ${s.kvTotal}` },
          { label: "TTFT p50", value: `${ms(s.ttftP50).toFixed(1)} ms`, fraction: ms(s.ttftP50) / 100, note: "scale 100 ms" },
          { label: "TPOT p50", value: `${ms(s.tpotP50).toFixed(2)} ms`, fraction: ms(s.tpotP50) / 20, note: "scale 20 ms" },
        ].map((g) => (
          <div key={g.label} className="rounded border border-border bg-surface p-4">
            <div className="flex items-baseline justify-between font-mono">
              <span className="text-xs text-muted">{g.label}</span>
              <span className="text-lg tabular-nums">{g.value}</span>
            </div>
            <div className="mt-3">
              <Bar fraction={g.fraction} />
            </div>
            <div className="mt-2 font-mono text-xs text-muted">{g.note}</div>
          </div>
        ))}
      </div>
    </Dashboard>
  );
}
