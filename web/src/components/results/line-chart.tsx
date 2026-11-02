"use client";

import NumberFlow from "@number-flow/react";
import clsx from "clsx";
import { AnimatePresence, motion } from "motion/react";
import { type KeyboardEvent, type PointerEvent, type ReactNode, useEffect, useRef, useState } from "react";
import type { Series } from "@/content/results";
import { usePrefersReducedMotion } from "@/lib/reduced-motion";
import { charWidth, easeInOut, fillOf, strokeOf } from "./tones";

export const pad = { l: 52, r: 20, t: 22, b: 36 };

export function useChartWidth() {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(560);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const observer = new ResizeObserver(([entry]) => setWidth(Math.round(Math.max(340, Math.min(560, entry.contentRect.width)) / 20) * 20));
    observer.observe(el);
    return () => observer.disconnect();
  }, []);
  return { ref, width };
}

export type Axis = { kind: "linear" | "log"; min: number; max: number; ticks: number[]; format: (v: number) => string; title: string };

export function scale(axis: Axis, from: number, to: number) {
  const f = (v: number) => (axis.kind === "log" ? Math.log10(v) : v);
  return (v: number) => Math.round((from + ((f(v) - f(axis.min)) / (f(axis.max) - f(axis.min))) * (to - from)) * 100) / 100;
}

export function svgX(svg: SVGSVGElement | null, clientX: number): number | null {
  const m = svg?.getScreenCTM();
  return m ? (clientX - m.e) / m.a : null;
}

export function stepKey(e: KeyboardEvent, current: number | null, count: number): number | null | undefined {
  const at = current ?? -1;
  switch (e.key) {
    case "ArrowRight":
    case "ArrowDown":
      return Math.min(at + 1, count - 1);
    case "ArrowLeft":
    case "ArrowUp":
      return Math.max(at - 1, 0);
    case "Home":
      return 0;
    case "End":
      return count - 1;
    case "Escape":
      return null;
    default:
      return undefined;
  }
}

type Props = {
  label: string;
  view: string;
  xs: number[];
  xAxis: Axis;
  yAxis: Axis;
  series: Series[];
  heading: (i: number) => string;
  value: (v: number) => string;
  underlay?: (sx: (v: number) => number, sy: (v: number) => number) => ReactNode;
};

function pathOf(px: number[], py: (number | undefined)[]): string {
  return px.flatMap((x, i) => (py[i] === undefined ? [] : [`${x} ${py[i]}`])).map((p, i) => `${i ? "L" : "M"} ${p}`).join(" ");
}

type Row = { s: Series; v: number; text: string; note?: string };

function Readout({ x, W, heading, rows }: { x: number; W: number; heading: string; rows: Row[] }) {
  const width = Math.max(heading.length, ...rows.map((r) => Math.max(r.text.length, r.note?.length ?? 0) + 4)) * charWidth + 20;
  const offsets = rows.map((_, i) => rows.slice(0, i).reduce((y, r) => y + (r.note ? 30 : 16), 0));
  const height = 22 + rows.reduce((y, r) => y + (r.note ? 30 : 16), 0);
  const left = x + 12 + width > W - 4 ? x - 12 - width : x + 12;
  return (
    <g transform={`translate(${Math.max(4, left)} ${pad.t})`} pointerEvents="none">
      <rect width={width} height={height} rx={4} className="fill-surface stroke-border" strokeWidth={1} />
      <text x={10} y={15} className="fill-muted text-[11px]">
        {heading}
      </text>
      {rows.map((r, i) => (
        <g key={r.s.id} transform={`translate(10 ${30 + offsets[i]})`}>
          <line x1={0} x2={14} y1={-4} y2={-4} className={strokeOf[r.s.tone]} strokeWidth={2} strokeDasharray={r.s.dash === "2 4" ? "2 2" : r.s.dash === "6 4" ? "4 2" : undefined} />
          <text x={22} y={0} className="fill-text text-[11px]">
            {r.text}
          </text>
          {r.note && (
            <text x={22} y={14} className="fill-muted text-[11px]">
              {r.note}
            </text>
          )}
        </g>
      ))}
    </g>
  );
}

export function LineChart({ label, view, xs, xAxis, yAxis, series, heading, value, underlay }: Props) {
  const reduced = usePrefersReducedMotion();
  const svg = useRef<SVGSVGElement>(null);
  const { ref, width: W } = useChartWidth();
  const H = Math.round(W * 0.54);
  const [hover, setHover] = useState<number | null>(null);
  const sx = scale(xAxis, pad.l, W - pad.r);
  const sy = scale(yAxis, H - pad.b, pad.t);
  const px = xs.map(sx);
  const plotW = W - pad.l - pad.r;
  const draw = (k: number) => (reduced ? 0 : 0.25 + k * 0.12);

  const pick = (e: PointerEvent) => {
    const x = svgX(svg.current, e.clientX);
    if (x === null) return;
    setHover(px.reduce((best, p, i) => (Math.abs(p - x) < Math.abs(px[best] - x) ? i : best), 0));
  };

  const onKey = (e: KeyboardEvent) => {
    const next = stepKey(e, hover, xs.length);
    if (next === undefined) return;
    e.preventDefault();
    setHover(next);
  };

  const rows: Row[] =
    hover === null
      ? []
      : series
          .flatMap((s) => {
            const v = s.values[hover];
            return v === undefined ? [] : [{ s, v, text: `${value(v)}  ${s.label}`, note: s.notes?.[hover] }];
          })
          .sort((a, b) => b.v - a.v);

  return (
    <div ref={ref} className="relative min-h-0 flex-1">
      <svg
        key={W}
        ref={svg}
        viewBox={`0 0 ${W} ${H}`}
        className="h-full w-full touch-pan-y font-mono [font-variant-ligatures:none]"
        role="img"
        aria-label={`${label}. Focus and use the arrow keys to read the values.`}
        tabIndex={0}
        onKeyDown={onKey}
        onFocus={() => setHover((h) => h ?? 0)}
        onBlur={() => setHover(null)}
      >
        <AnimatePresence>
          <motion.g key={`y-${view}`} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: reduced ? 0 : 0.4 }}>
            {yAxis.ticks.map((t) => (
              <g key={t}>
                <line x1={pad.l} x2={W - pad.r} y1={sy(t)} y2={sy(t)} className="stroke-border" strokeWidth={1} />
                <text x={pad.l - 8} y={sy(t) + 4} textAnchor="end" className="fill-muted text-[11px] tabular-nums">
                  {yAxis.format(t)}
                </text>
              </g>
            ))}
            <text x={pad.l} y={pad.t - 10} className="fill-muted text-[11px]">
              {yAxis.title}
            </text>
          </motion.g>
        </AnimatePresence>
        {xAxis.ticks.map((t) => (
          <text key={t} x={sx(t)} y={H - pad.b + 16} textAnchor="middle" className="fill-muted text-[11px] tabular-nums">
            {xAxis.format(t)}
          </text>
        ))}
        <text x={W - pad.r} y={H - 4} textAnchor="end" className="fill-muted text-[11px]">
          {xAxis.title}
        </text>
        {underlay?.(sx, sy)}
        {hover !== null && <line x1={px[hover]} x2={px[hover]} y1={pad.t} y2={H - pad.b} className="stroke-muted" strokeWidth={1} />}
        <AnimatePresence>
          {series.map((s, k) => (
            <motion.g
              key={`${view}-${s.id}`}
              data-line={s.id}
              initial={{ clipPath: "inset(0% 100% 0% 0%)", opacity: 1 }}
              animate={{ clipPath: "inset(0% 0% 0% 0%)", opacity: 1 }}
              exit={{ opacity: 0, transition: { duration: reduced ? 0 : 0.25 } }}
              transition={{ duration: reduced ? 0 : 1.1, delay: draw(k), ease: easeInOut }}
            >
              <rect x={pad.l - 8} y={0} width={plotW + 16} height={H} fill="none" />
              <path
                d={pathOf(px, s.values.map((v) => (v === undefined ? undefined : sy(v))))}
                className={clsx("fill-none", strokeOf[s.tone])}
                strokeWidth={2}
                strokeDasharray={s.dash}
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </motion.g>
          ))}
        </AnimatePresence>
        {series.map((s, k) =>
          s.values.map((v, i) => {
            if (v === undefined) return null;
            const to = `translate(${px[i]}px, ${sy(v)}px)`;
            return (
              <motion.circle
                key={`${s.id}-${i}`}
                data-dot={`${s.id}-${i}`}
                r={hover === i ? 5 : 4}
                className={clsx(fillOf[s.tone], "stroke-surface")}
                strokeWidth={2}
                initial={{ opacity: 0, transform: to }}
                animate={{ opacity: 1, transform: to }}
                transition={{
                  transform: { duration: reduced ? 0 : 0.7, ease: easeInOut },
                  opacity: { duration: reduced ? 0 : 0.3, delay: reduced ? 0 : draw(k) + (1.1 * (px[i] - pad.l)) / plotW },
                }}
              />
            );
          }),
        )}
        <rect x={pad.l - 8} y={pad.t} width={plotW + 16} height={H - pad.t - pad.b} fill="transparent" onPointerMove={pick} onPointerDown={pick} onPointerLeave={() => setHover(null)} />
        {hover !== null && <Readout x={px[hover]} W={W} heading={heading(hover)} rows={rows} />}
      </svg>
      <p className="sr-only" aria-live="polite">
        {hover === null ? "" : `${heading(hover)}: ${rows.map((r) => (r.note ? `${r.text}, ${r.note}` : r.text)).join("; ")}`}
      </p>
    </div>
  );
}

export function Legend({ series }: { series: Series[] }) {
  return (
    <ul className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-muted">
      {series.map((s) => (
        <li key={s.id} className="flex items-center gap-1.5">
          <svg width="18" height="6" aria-hidden="true">
            <line x1="1" x2="17" y1="3" y2="3" className={strokeOf[s.tone]} strokeWidth={2} strokeDasharray={s.dash === "2 4" ? "2 2" : s.dash === "6 4" ? "4 2" : undefined} strokeLinecap="butt" />
          </svg>
          {s.label}
        </li>
      ))}
    </ul>
  );
}

export type Stat = { label: string; value: number; digits: number; suffix?: string };

export function Stats({ stats }: { stats: Stat[] }) {
  return (
    <div className="grid grid-cols-3 gap-x-4 border-t border-border pt-2 text-[11px] tabular-nums">
      {stats.map((s, i) => (
        <div key={i}>
          <div className="text-muted">{s.label}</div>
          <div className="text-sm text-text">
            <NumberFlow value={s.value} suffix={s.suffix} format={{ minimumFractionDigits: s.digits, maximumFractionDigits: s.digits }} />
          </div>
        </div>
      ))}
    </div>
  );
}
