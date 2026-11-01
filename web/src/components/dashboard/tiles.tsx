"use client";

import NumberFlow from "@number-flow/react";
import type { Spread, TileValues } from "@/lib/dashboard/stats";

const timing = { duration: 900, easing: "cubic-bezier(0.65, 0, 0.35, 1)" };

function Num({ value, digits = 0, suffix }: { value: number | null; digits?: number; suffix?: string }) {
  if (value === null || !Number.isFinite(value)) return <span className="text-muted">–</span>;
  return (
    <NumberFlow
      value={value}
      suffix={suffix}
      transformTiming={timing}
      spinTiming={timing}
      format={{ minimumFractionDigits: digits, maximumFractionDigits: digits }}
    />
  );
}

function Tile({ label, note, children }: { label: string; note: string; children: React.ReactNode }) {
  return (
    <div className="min-w-0 rounded border border-border bg-surface px-3 py-2.5">
      <div className="truncate text-muted">{label}</div>
      <div className="mt-1 truncate text-lg tabular-nums text-text sm:text-xl">{children}</div>
      <div className="truncate text-[11px] text-muted">{note}</div>
    </div>
  );
}

function Pair({ spread, digits }: { spread: Spread; digits: number }) {
  const ms = (v: number | undefined) => (v === undefined ? null : v * 1e3);
  return (
    <>
      <Num value={ms(spread?.p50)} digits={digits} />
      <span className="text-muted"> / </span>
      <Num value={ms(spread?.p99)} digits={digits} />
      <span className="text-sm text-muted"> ms</span>
    </>
  );
}

export function Tiles({ tiles, windows }: { tiles: TileValues; windows: { rate: string; latency: string } }) {
  const share = tiles.kvUsed !== null && tiles.kvTotal ? (100 * tiles.kvUsed) / tiles.kvTotal : null;
  return (
    <section aria-label="Metrics" className="grid grid-cols-2 gap-2 font-mono text-xs sm:grid-cols-3 lg:grid-cols-5">
      <Tile label="tokens/s" note={windows.rate}>
        <Num value={tiles.tokensPerS} />
      </Tile>
      <Tile label="running / waiting" note="after the last step">
        <Num value={tiles.running} />
        <span className="text-muted"> / </span>
        <Num value={tiles.waiting} />
      </Tile>
      <Tile label="KV blocks used" note={tiles.kvTotal ? `of ${tiles.kvTotal.toLocaleString("en-US")} · ${share === null ? "–" : share.toFixed(2)}%` : "of the pool"}>
        <Num value={tiles.kvUsed} />
      </Tile>
      <Tile label="TTFT p50 / p99" note={windows.latency}>
        <Pair spread={tiles.ttft} digits={0} />
      </Tile>
      <Tile label="TPOT p50 / p99" note={windows.latency}>
        <Pair spread={tiles.tpot} digits={1} />
      </Tile>
    </section>
  );
}
