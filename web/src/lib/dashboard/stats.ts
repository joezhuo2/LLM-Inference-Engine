import type { TraceRequest, TraceStep } from "../trace";

export type Spread = { p50: number; p99: number } | null;

export type TileValues = {
  tokensPerS: number | null;
  running: number | null;
  waiting: number | null;
  kvUsed: number | null;
  kvTotal: number | null;
  ttft: Spread;
  tpot: Spread;
};

export const emptyTiles: TileValues = { tokensPerS: null, running: null, waiting: null, kvUsed: null, kvTotal: null, ttft: null, tpot: null };

export function percentile(sorted: number[], p: number): number {
  const rank = Math.ceil((p / 100) * sorted.length);
  return sorted[Math.min(sorted.length, Math.max(1, rank)) - 1];
}

export function spread(values: number[]): Spread {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  return { p50: percentile(sorted, 50), p99: percentile(sorted, 99) };
}

export function tokensPerSecond(steps: TraceStep[], done: number, t: number, windowS: number): number {
  let tokens = 0;
  for (let i = done - 1; i >= 0 && steps[i].end_s > t - windowS; i--) tokens += steps[i].seqs.length;
  const span = Math.min(windowS, t);
  return span > 0 ? tokens / span : 0;
}

export function replayTiles(steps: TraceStep[], requests: TraceRequest[], done: number, t: number, kvTotal: number): TileValues {
  const last = done > 0 ? steps[done - 1] : undefined;
  const finished = requests.filter((r) => r.finished_s <= t);
  return {
    tokensPerS: tokensPerSecond(steps, done, t, 1),
    running: last?.running ?? 0,
    waiting: last?.waiting ?? requests.filter((r) => r.arrival_s <= t).length,
    kvUsed: last ? kvTotal - 1 - last.free_blocks : 0,
    kvTotal,
    ttft: spread(finished.map((r) => r.first_token_s - r.arrival_s)),
    tpot: spread(finished.filter((r) => r.output_tokens > 1).map((r) => (r.finished_s - r.first_token_s) / (r.output_tokens - 1))),
  };
}

export function stepsDoneBy(steps: TraceStep[], t: number): number {
  let lo = 0;
  let hi = steps.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (steps[mid].end_s <= t) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}
