"use client";

import clsx from "clsx";
import { AnimatePresence, motion } from "motion/react";
import Link from "next/link";
import { useState } from "react";
import { MetricView } from "@/components/build-timeline";
import { repoUrl } from "@/content/build-log";
import { type DeadEndKind, deadEnds, kindLabel } from "@/content/dead-ends";
import { authorLabel } from "@/content/kernels/types";
import { radioKeyDown } from "@/lib/radio-keys";
import { usePrefersReducedMotion } from "@/lib/reduced-motion";

const ease = [0.65, 0, 0.35, 1] as const;
type Filter = "all" | DeadEndKind;
const filters: Filter[] = ["all", "slower", "broke", "cliff", "misread"];
const filterLabel = (f: Filter) => (f === "all" ? "all" : kindLabel[f]);

export function DeadEndList() {
  const reduced = usePrefersReducedMotion();
  const [filter, setFilter] = useState<Filter>("all");
  const shown = deadEnds.filter((d) => filter === "all" || d.kind === filter);
  return (
    <div className="space-y-8">
      <div role="radiogroup" aria-label="kind of dead end" onKeyDown={radioKeyDown(filters, filter, setFilter)} className="flex flex-wrap gap-2 font-mono text-xs">
        {filters.map((f) => {
          const count = f === "all" ? deadEnds.length : deadEnds.filter((d) => d.kind === f).length;
          return (
            <button
              key={f}
              type="button"
              role="radio"
              aria-checked={filter === f}
              tabIndex={filter === f ? 0 : -1}
              onClick={() => setFilter(f)}
              className={clsx("pressable rounded border px-3 py-1", filter === f ? "border-accent/40 bg-accent-dim text-accent" : "border-border text-muted")}
            >
              {`${filterLabel(f)} · ${count}`}
            </button>
          );
        })}
      </div>
      <ol className="space-y-6">
        <AnimatePresence initial={false} mode="popLayout">
          {shown.map((d) => (
            <motion.li
              key={`${d.branch}-${d.title}`}
              layout={!reduced}
              initial={reduced ? false : { opacity: 0, transform: "translateY(8px)" }}
              animate={{ opacity: 1, transform: "translateY(0px)" }}
              exit={reduced ? { opacity: 0, transition: { duration: 0 } } : { opacity: 0, transform: "translateY(-4px)" }}
              transition={{ duration: 0.5, ease }}
              className="rounded border border-border bg-surface p-4 sm:p-5"
            >
              <div className="flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-xs text-muted">
                <span className="tabular-nums">{d.date}</span>
                <span className="rounded-sm border border-warn/40 px-1.5 py-px text-warn">{kindLabel[d.kind]}</span>
                <Link href={`/kernels/${d.slug}`} transitionTypes={["nav-forward"]} className="link-underline text-accent">
                  {d.kernel}
                </Link>
                <span className="text-text">{d.branch}</span>
                <span>
                  {d.prs.map((n, i) => (
                    <span key={n}>
                      {i > 0 && " "}
                      <a href={`${repoUrl}/pull/${n}`} className="link-underline tabular-nums">
                        #{n}
                      </a>
                    </span>
                  ))}
                </span>
                <span className={d.author === "claude" ? "text-muted" : "text-text"}>· {authorLabel[d.author]}</span>
              </div>
              <h2 className="mt-2 text-lg text-text">{d.title}</h2>
              <dl className="mt-3 grid gap-x-6 gap-y-2 text-sm sm:grid-cols-[6rem_minmax(0,1fr)]">
                <dt className="font-mono text-xs text-muted sm:pt-0.5">tried</dt>
                <dd className="text-muted">{d.tried}</dd>
                <dt className="font-mono text-xs text-muted sm:pt-0.5">happened</dt>
                <dd className="text-text">{d.happened}</dd>
                <dt className="font-mono text-xs text-muted sm:pt-0.5">kept</dt>
                <dd className="text-muted">{d.kept}</dd>
              </dl>
              {d.metric && (
                <div className="mt-4 max-w-xl">
                  <MetricView metric={d.metric} reduced={reduced} />
                </div>
              )}
              <a href={`${repoUrl}/blob/main/${d.source.path}`} className="link-underline mt-3 inline-block font-mono text-[11px] text-muted">
                source: {d.source.label}
              </a>
            </motion.li>
          ))}
        </AnimatePresence>
      </ol>
    </div>
  );
}
