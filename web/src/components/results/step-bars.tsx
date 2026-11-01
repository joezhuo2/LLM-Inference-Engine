"use client";

import clsx from "clsx";
import { AnimatePresence, motion } from "motion/react";
import { type KeyboardEvent, useState } from "react";
import type { Segment, StepRow } from "@/content/results";
import { usePrefersReducedMotion } from "@/lib/reduced-motion";
import { stepKey, useChartWidth } from "./line-chart";
import { charWidth, easeInOut } from "./tones";

const H = 246;
const left = 16;
const right = 16;
const maxUs = 10000;
const barH = 26;
const rowY = [36, 112];
const axisY = rowY[1] + barH + 22;

const groupFill: Record<Segment["group"], string> = {
  gemm: "fill-accent",
  attention: "fill-series-2",
  other: "fill-series-3",
  idle: "fill-baseline",
  graph: "fill-muted",
};

const labelFill: Record<Segment["group"], string> = {
  gemm: "fill-bg",
  attention: "fill-white",
  other: "fill-white",
  idle: "fill-black dark:fill-white",
  graph: "fill-white dark:fill-black",
};

export const groupLegend: { group: Segment["group"]; label: string }[] = [
  { group: "gemm", label: "cuBLAS GEMMs" },
  { group: "attention", label: "attention" },
  { group: "other", label: "other kernels" },
  { group: "graph", label: "one graph replay" },
  { group: "idle", label: "GPU idle" },
];


type Placed = Segment & { slot: number; x: number; w: number };

function place(rows: StepRow[], sx: (us: number) => number): Placed[] {
  return rows.flatMap((row, slot) => {
    let at = 0;
    return row.segments.map((s) => {
      const x = sx(at);
      at += s.us;
      return { ...s, slot, x, w: Math.max(0.5, sx(at) - x - 2) };
    });
  });
}

function wrap(text: string, width: number): string[] {
  const lines: string[] = [];
  for (const part of text.split(", ")) {
    const last = lines.length - 1;
    if (last >= 0 && (lines[last] + ", " + part).length <= width) lines[last] += `, ${part}`;
    else lines.push(part);
  }
  return lines;
}

export function StepBars({ rows }: { rows: StepRow[] }) {
  const reduced = usePrefersReducedMotion();
  const [hover, setHover] = useState<number | null>(null);
  const { ref, width: W } = useChartWidth();
  const sx = (us: number) => Math.round((left + (us / maxUs) * (W - left - right)) * 100) / 100;
  const placed = place(rows, sx);
  const active = hover === null ? null : placed[hover];
  const grow = (s: Placed, i: number) => ({
    transform: { duration: reduced ? 0 : 0.7, ease: easeInOut, delay: reduced ? 0 : 0.1 + (i % 9) * 0.05 },
    opacity: { duration: reduced ? 0 : 0.3 },
  });

  const onKey = (e: KeyboardEvent) => {
    const next = stepKey(e, hover, placed.length);
    if (next === undefined) return;
    e.preventDefault();
    setHover(next);
  };

  return (
    <div ref={ref} className="relative min-h-0 flex-1">
      <svg
        key={W}
        viewBox={`0 0 ${W} ${H}`}
        className="h-full w-full font-mono [font-variant-ligatures:none]"
        role="img"
        aria-label={`GPU time of one decode step by op: ${rows.map((r) => r.label).join(" and ")}. Focus and use the arrow keys to read each segment.`}
        tabIndex={0}
        onKeyDown={onKey}
        onFocus={() => setHover((h) => h ?? 0)}
        onBlur={() => setHover(null)}
      >
        {[0, 2000, 4000, 6000, 8000, 10000].map((t) => (
          <g key={t}>
            <line x1={sx(t)} x2={sx(t)} y1={rowY[0] - 6} y2={axisY - 12} className="stroke-border" strokeWidth={1} />
            <text x={sx(t)} y={axisY} textAnchor={t === 0 ? "start" : t === maxUs ? "end" : "middle"} className="fill-muted text-[11px] tabular-nums">
              {`${t / 1000} ms`}
            </text>
          </g>
        ))}
        {rows.map((row, slot) => (
          <AnimatePresence key={slot}>
            <motion.text
              key={row.id}
              x={left}
              y={rowY[slot] - 8}
              className="fill-text text-[11px]"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0, transition: { duration: reduced ? 0 : 0.2 } }}
              transition={{ duration: reduced ? 0 : 0.4, delay: reduced ? 0 : 0.2 }}
            >
              {`${row.label}: ${(row.span / 1000).toFixed(2)} ms span, ${row.idlePct}% idle`}
            </motion.text>
          </AnimatePresence>
        ))}
        <AnimatePresence>
          {placed.map((s, i) => {
            const at = `translate(${s.x}px, ${rowY[s.slot]}px)`;
            return (
              <motion.rect
                key={`${s.slot}-${s.id}`}
                data-seg={`${s.slot}-${s.id}`}
                width={1}
                height={barH}
                className={groupFill[s.group]}
                style={{ transformBox: "fill-box", originX: 0, originY: 0 }}
                initial={{ transform: `${at} scaleX(0)`, opacity: 1 }}
                animate={{ transform: `${at} scaleX(${s.w})`, opacity: 1 }}
                exit={{ opacity: 0, transition: { duration: reduced ? 0 : 0.3 } }}
                transition={grow(s, i)}
                onPointerEnter={() => setHover(i)}
                onPointerDown={() => setHover(i)}
                onPointerLeave={() => setHover(null)}
              />
            );
          })}
        </AnimatePresence>
        <AnimatePresence>
          {placed
            .filter((s) => s.w > s.label.length * charWidth + 12)
            .map((s) => (
              <motion.text
                key={`${s.slot}-${s.id}-label-${Math.round(s.x)}`}
                x={s.x + 6}
                y={rowY[s.slot] + barH / 2 + 4}
                className={clsx("pointer-events-none text-[11px]", labelFill[s.group])}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0, transition: { duration: reduced ? 0 : 0.15 } }}
                transition={{ duration: reduced ? 0 : 0.3, delay: reduced ? 0 : 0.8 }}
              >
                {s.label}
              </motion.text>
            ))}
        </AnimatePresence>
        {active && <rect x={active.x - 1} y={rowY[active.slot] - 1} width={active.w + 2} height={barH + 2} className="pointer-events-none fill-none stroke-text" strokeWidth={1.5} />}
        <g transform={`translate(${left} ${axisY + 30})`} className="text-[11px]">
          {active ? (
            <>
              <text className="fill-text">{`${rows[active.slot].label} · ${active.label}: ${active.us.toLocaleString("en-US", { maximumFractionDigits: 1 })} µs`}</text>
              {wrap(active.detail, Math.floor((W - 2 * left) / charWidth)).map((line, i) => (
                <text key={i} y={16 + i * 15} className="fill-muted">
                  {line}
                </text>
              ))}
            </>
          ) : (
            <text className="fill-muted">hover or focus a segment for its time per step</text>
          )}
        </g>
      </svg>
      <p className="sr-only" aria-live="polite">
        {active ? `${rows[active.slot].label}, ${active.label}: ${active.detail}` : ""}
      </p>
    </div>
  );
}

export function GroupLegend({ groups }: { groups: Segment["group"][] }) {
  return (
    <ul className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-muted">
      {groupLegend
        .filter((g) => groups.includes(g.group))
        .map((g) => (
          <li key={g.group} className="flex items-center gap-1.5">
            <svg width="10" height="10" aria-hidden="true">
              <rect width="10" height="10" rx="2" className={groupFill[g.group]} />
            </svg>
            {g.label}
          </li>
        ))}
    </ul>
  );
}
