"use client";

import { motion, useScroll, useTransform } from "motion/react";
import { useRef } from "react";
import { type Entry, type Metric, type Milestone, milestones, repoUrl } from "@/content/build-log";
import { usePrefersReducedMotion } from "@/lib/reduced-motion";

const ease = [0.23, 1, 0.32, 1] as const;
const inView = { once: true, margin: "0px 0px -15% 0px" };

function MetricView({ metric, reduced }: { metric: Metric; reduced: boolean }) {
  if (metric.kind === "value") {
    return (
      <div className="font-mono">
        <div className="text-2xl text-accent tabular-nums">{metric.value}</div>
        <div className="mt-1 text-xs text-muted">{metric.label}</div>
      </div>
    );
  }
  const max = Math.max(...metric.bars.map((b) => b.value));
  return (
    <figure className="font-mono text-xs">
      <figcaption className="mb-2 text-muted">{metric.label}</figcaption>
      <div className="space-y-1.5">
        {metric.bars.map((bar, i) => (
          <div key={bar.label} className="grid grid-cols-[7.5rem_1fr] items-center gap-3 sm:grid-cols-[9rem_1fr]">
            <span className={bar.highlight ? "truncate text-text" : "truncate text-muted"}>{bar.label}</span>
            <div className="flex items-center gap-2">
              <div className="h-2 min-w-0 flex-1">
                <motion.div
                  className={bar.highlight ? "h-full rounded-sm bg-accent" : "h-full rounded-sm bg-baseline/60"}
                  style={{ width: `${max ? (100 * bar.value) / max : 0}%`, transformOrigin: "left" }}
                  initial={reduced ? false : { transform: "scaleX(0)" }}
                  whileInView={{ transform: "scaleX(1)" }}
                  viewport={inView}
                  transition={{ duration: 0.7, delay: 0.15 + i * 0.08, ease }}
                />
              </div>
              <span className={bar.highlight ? "w-20 text-right text-accent tabular-nums" : "w-20 text-right text-muted tabular-nums"}>
                {bar.value.toFixed(metric.digits)}
                {metric.unit && ` ${metric.unit}`}
              </span>
            </div>
          </div>
        ))}
      </div>
    </figure>
  );
}

function EntryView({ entry, reduced }: { entry: Entry; reduced: boolean }) {
  return (
    <motion.li
      className="relative pb-12 pl-8 last:pb-0"
      initial={reduced ? { opacity: 0 } : { opacity: 0, transform: "translateY(8px)" }}
      whileInView={reduced ? { opacity: 1 } : { opacity: 1, transform: "translateY(0px)" }}
      viewport={inView}
      transition={{ duration: 0.35, ease }}
    >
      <span aria-hidden="true" className="absolute left-[-4px] top-1.5 h-[9px] w-[9px] rounded-full border border-accent bg-bg" />
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1 font-mono text-xs text-muted">
        <span className="tabular-nums">{entry.date}</span>
        <span className="text-accent">{entry.branch}</span>
        <span>
          {entry.prs.map((n, i) => (
            <span key={n}>
              {i > 0 && " "}
              <a href={`${repoUrl}/pull/${n}`} className="link-hover tabular-nums">
                #{n}
              </a>
            </span>
          ))}
        </span>
      </div>
      <h3 className="mt-2 text-lg text-text">{entry.title}</h3>
      <p className="mt-1 max-w-xl text-muted">{entry.text}</p>
      {entry.metric && (
        <div className="mt-4 max-w-xl">
          <MetricView metric={entry.metric} reduced={reduced} />
        </div>
      )}
    </motion.li>
  );
}

function MilestoneView({ milestone, reduced }: { milestone: Milestone; reduced: boolean }) {
  const ref = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({ target: ref, offset: ["start 70%", "end 70%"] });
  const fill = useTransform(scrollYProgress, (p) => `scaleY(${p})`);
  return (
    <section aria-labelledby={`milestone-${milestone.id}`} className="grid gap-6 md:grid-cols-[13rem_1fr] md:gap-10">
      <header className="md:sticky md:top-20 md:self-start">
        <div className="font-mono text-xs text-muted tabular-nums">
          {milestone.id} · {milestone.dates}
        </div>
        <h2 id={`milestone-${milestone.id}`} className="mt-1 font-mono text-xl text-text">
          {milestone.name}
        </h2>
        <p className="mt-2 text-sm text-muted">{milestone.goal}</p>
      </header>
      <div ref={ref} className="relative">
        <div aria-hidden="true" className="absolute bottom-0 left-0 top-2 w-px bg-border" />
        {reduced ? (
          <div aria-hidden="true" className="absolute bottom-0 left-0 top-2 w-px bg-accent" />
        ) : (
          <motion.div
            aria-hidden="true"
            className="absolute bottom-0 left-0 top-2 w-px origin-top bg-accent"
            style={{ transform: fill }}
          />
        )}
        <ol>
          {milestone.entries.map((entry) => (
            <EntryView key={entry.branch} entry={entry} reduced={reduced} />
          ))}
        </ol>
      </div>
    </section>
  );
}

export function BuildTimeline() {
  const reduced = usePrefersReducedMotion();
  const prs = milestones.flatMap((m) => m.entries.flatMap((e) => e.prs)).length;
  return (
    <div className="space-y-20">
      {milestones.map((m) => (
        <MilestoneView key={m.id} milestone={m} reduced={reduced} />
      ))}
      <p className="font-mono text-xs text-muted">
        {prs} of the pull requests shown;{" "}
        <a href={`${repoUrl}/pulls?q=is%3Apr+is%3Amerged`} className="link-underline text-text">
          all of them on GitHub
        </a>
      </p>
    </div>
  );
}
