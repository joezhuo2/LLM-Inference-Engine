"use client";

import clsx from "clsx";
import { motion } from "motion/react";
import { useState } from "react";
import type { Criterion, Gauge } from "@/content/results";
import { usePrefersReducedMotion } from "@/lib/reduced-motion";
import { easeInOut } from "./tones";

function BarGauge({ gauge, delay, reduced }: { gauge: Extract<Gauge, { kind: "bar" }>; delay: number; reduced: boolean }) {
  const pct = (v: number) => `${((Math.min(v, gauge.max) / gauge.max) * 100).toFixed(2)}%`;
  const met = gauge.higher ? gauge.value >= gauge.target : gauge.value <= gauge.target;
  return (
    <div className="flex items-center gap-3">
      <div className="relative h-1.5 min-w-0 flex-1 rounded-full bg-border">
        <motion.div
          data-gauge
          className={clsx("absolute inset-y-0 left-0 rounded-full", met ? "bg-accent" : "bg-warn")}
          style={{ width: pct(gauge.value), transformOrigin: "0% 50%" }}
          initial={{ transform: reduced ? "scaleX(1)" : "scaleX(0)" }}
          animate={{ transform: "scaleX(1)" }}
          transition={{ duration: reduced ? 0 : 0.9, delay: reduced ? 0 : delay, ease: easeInOut }}
        />
        <div className="absolute -top-1 h-3.5 w-px bg-text" style={{ left: pct(gauge.target) }} />
      </div>
      <span className="w-36 shrink-0 whitespace-nowrap tabular-nums">
        <span className="text-text">{gauge.format(gauge.value)}</span>
        <span className="text-muted">{`  ${gauge.higher ? "target" : "limit"} ${gauge.format(gauge.target)}`}</span>
      </span>
    </div>
  );
}

function CellGauge({ gauge, delay, reduced }: { gauge: Extract<Gauge, { kind: "cells" }>; delay: number; reduced: boolean }) {
  return (
    <div className="flex items-center gap-3">
      <div className="flex min-w-0 flex-1 gap-[3px]">
        {Array.from({ length: gauge.total }, (_, i) => (
          <motion.span
            key={i}
            className={clsx("aspect-square max-w-2.5 flex-1 rounded-[2px]", i < gauge.exact ? "bg-accent" : "border border-baseline")}
            initial={{ opacity: reduced ? 1 : 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: reduced ? 0 : 0.25, delay: reduced ? 0 : delay + i * 0.035 }}
          />
        ))}
      </div>
      <span className="w-36 shrink-0 whitespace-nowrap tabular-nums">
        <span className="text-text">{`${gauge.exact} exact`}</span>
        <span className="text-muted">{`  ${gauge.ties} near ties`}</span>
      </span>
    </div>
  );
}

export function Scoreboard({ criteria }: { criteria: Criterion[] }) {
  const reduced = usePrefersReducedMotion();
  const [hover, setHover] = useState<number | null>(null);
  const active = hover === null ? null : criteria[hover];
  return (
    <div className="flex min-h-0 flex-1 flex-col justify-center gap-3 text-[11px]">
      <ul className="space-y-1" aria-label="MVP criteria">
        {criteria.map((c, k) => {
          const delay = 0.2 + k * 0.12;
          return (
            <li
              key={c.id}
              data-criterion={c.id}
              tabIndex={0}
              aria-label={`${c.id} ${c.name}: target ${c.target}; measured ${c.measured}; ${c.status}`}
              onPointerEnter={() => setHover(k)}
              onPointerLeave={() => setHover(null)}
              onFocus={() => setHover(k)}
              onBlur={() => setHover(null)}
              className={clsx("grid grid-cols-[1.5rem_minmax(0,1fr)_3.5rem] gap-x-2 gap-y-1 rounded px-1 py-1", hover === k && "bg-accent-dim")}
            >
              <span className="text-muted">{c.id}</span>
              <span className="truncate text-text">{c.name}</span>
              <span className={clsx("rounded px-1 text-center text-[10px] leading-4", c.status === "met" ? "bg-accent-dim text-accent" : "bg-border text-muted")}>{c.status}</span>
              <div className="col-start-2 col-end-4">
                {c.gauge?.kind === "bar" && <BarGauge gauge={c.gauge} delay={delay} reduced={reduced} />}
                {c.gauge?.kind === "cells" && <CellGauge gauge={c.gauge} delay={delay} reduced={reduced} />}
                {!c.gauge && <span className="block truncate text-muted">{c.measured}</span>}
              </div>
            </li>
          );
        })}
      </ul>
      <div className="min-h-[3.75rem] border-t border-border pt-2 leading-relaxed" aria-hidden="true">
        {active ? (
          <>
            <div className="text-text">{`target: ${active.target}`}</div>
            <div className="text-muted">{`measured: ${active.measured}`}</div>
          </>
        ) : (
          <div className="text-muted">hover or focus a row for its target and measurement</div>
        )}
      </div>
    </div>
  );
}
