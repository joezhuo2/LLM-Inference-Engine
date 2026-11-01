"use client";

import { useEffect, useState } from "react";
import { stepsDoneBy as stepsDone } from "@/lib/dashboard/stats";
import { preferredTrace } from "@/lib/data";

export type TraceToken = { id: number; text: string };

export type TraceStep = {
  index: number;
  start_s: number;
  end_s: number;
  kind: "prefill" | "decode";
  seqs: number[];
  tokens: TraceToken[];
  finished: number[];
  waiting: number;
  running: number;
  free_blocks: number;
};

export type TraceRequest = {
  id: number;
  arrival_s: number;
  prompt_tokens: number;
  output_tokens: number;
  preemptions: number;
  first_token_s: number;
  finished_s: number;
};

export type Trace = {
  format: string;
  version: number;
  meta: Record<string, string | number | boolean>;
  requests: TraceRequest[];
  steps: TraceStep[];
};

export function preferredTraceFile(): string | undefined {
  return preferredTrace()?.file;
}

export function useTrace(): Trace | null {
  const [trace, setTrace] = useState<Trace | null>(null);
  useEffect(() => {
    const file = preferredTraceFile();
    if (!file) return;
    let alive = true;
    fetch(file)
      .then((r) => r.json())
      .then((t: Trace) => {
        if (alive) setTrace(t);
      });
    return () => {
      alive = false;
    };
  }, []);
  return trace;
}

export function traceDuration(trace: Trace): number {
  return trace.steps.length ? trace.steps[trace.steps.length - 1].end_s : 0;
}

export function stepsDoneBy(trace: Trace, t: number): number {
  return stepsDone(trace.steps, t);
}

export function loopTime(elapsedS: number, duration: number, pauseS = 1.5): number {
  const t = elapsedS % (duration + pauseS);
  return Math.min(t, duration);
}

export type StepListener = (step: TraceStep) => void;

export function createReplayer(trace: Trace, onStep: StepListener, onReset: () => void) {
  let done = 0;
  let last = -1;
  return (t: number) => {
    if (t < last) {
      done = 0;
      onReset();
    }
    last = t;
    const target = stepsDoneBy(trace, t);
    while (done < target) onStep(trace.steps[done++]);
  };
}

export function snapshotAt(trace: Trace, t: number) {
  const done = stepsDoneBy(trace, t);
  const step = done > 0 ? trace.steps[done - 1] : undefined;
  let tokens = 0;
  for (let i = done - 1; i >= 0 && trace.steps[i].end_s > t - 1; i--) tokens += trace.steps[i].seqs.length;
  const finished = trace.requests.filter((r) => r.finished_s <= t);
  const ttft = finished.map((r) => r.first_token_s - r.arrival_s).sort((a, b) => a - b);
  const tpot = finished
    .filter((r) => r.output_tokens > 1)
    .map((r) => (r.finished_s - r.first_token_s) / (r.output_tokens - 1))
    .sort((a, b) => a - b);
  const kvTotal = Number(trace.meta.kv_blocks ?? 0);
  const idleFree = trace.steps.reduce((m, st) => Math.max(m, st.free_blocks), 0);
  return {
    t,
    tokensPerS: tokens,
    running: step?.running ?? 0,
    waiting: step?.waiting ?? 0,
    kvUsed: step ? idleFree - step.free_blocks : 0,
    kvTotal,
    maxBatch: Number(trace.meta.max_batch ?? 16),
    finished: finished.length,
    ttftP50: ttft.length ? ttft[Math.floor(ttft.length / 2)] : undefined,
    tpotP50: tpot.length ? tpot[Math.floor(tpot.length / 2)] : undefined,
  };
}

export type Snapshot = ReturnType<typeof snapshotAt>;
