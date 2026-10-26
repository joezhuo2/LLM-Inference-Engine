"use client";

import clsx from "clsx";
import { motion, useScroll, useTransform } from "motion/react";
import { useRef } from "react";
import { MetricView } from "@/components/build-timeline";
import { repoUrl } from "@/content/build-log";
import { authorLabel, type KernelEntry } from "@/content/kernels/types";
import { usePrefersReducedMotion } from "@/lib/reduced-motion";

const ease = [0.23, 1, 0.32, 1] as const;
const inView = { once: true, margin: "0px 0px -15% 0px" };

export function KernelTimeline({ entries }: { entries: KernelEntry[] }) {
  const reduced = usePrefersReducedMotion();
  const ref = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({ target: ref, offset: ["start 70%", "end 70%"] });
  const fill = useTransform(scrollYProgress, (p) => `scaleY(${p})`);
  return (
    <div ref={ref} className="relative">
      <div aria-hidden="true" className="absolute bottom-0 left-0 top-2 w-px bg-border" />
      {reduced ? (
        <div aria-hidden="true" className="absolute bottom-0 left-0 top-2 w-px bg-accent" />
      ) : (
        <motion.div aria-hidden="true" className="absolute bottom-0 left-0 top-2 w-px origin-top bg-accent" style={{ transform: fill }} />
      )}
      <ol>
        {entries.map((e) => (
          <motion.li
            key={`${e.branch}-${e.prs.join("-")}-${e.title}`}
            className="relative pb-12 pl-8 last:pb-0"
            initial={reduced ? { opacity: 0 } : { opacity: 0, transform: "translateY(8px)" }}
            whileInView={reduced ? { opacity: 1 } : { opacity: 1, transform: "translateY(0px)" }}
            viewport={inView}
            transition={{ duration: 0.35, ease }}
          >
            <span aria-hidden="true" className="absolute left-[-4px] top-1.5 h-[9px] w-[9px] rounded-full border border-accent bg-bg" />
            <div className="flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-xs text-muted">
              <span className="tabular-nums">{e.date}</span>
              <span
                className={clsx(
                  "rounded-sm border px-1.5 py-px",
                  e.version === "spec" ? "border-border text-muted" : "border-accent/40 bg-accent-dim text-accent",
                )}
              >
                {e.version}
              </span>
              <span className="text-text">{e.branch}</span>
              <span>
                {e.prs.map((n, i) => (
                  <span key={n}>
                    {i > 0 && " "}
                    <a href={`${repoUrl}/pull/${n}`} className="link-underline tabular-nums">
                      #{n}
                    </a>
                  </span>
                ))}
              </span>
              <span className={clsx(e.author === "claude" ? "text-muted" : "text-text")}>· {authorLabel[e.author]}</span>
            </div>
            <h3 className="mt-2 text-lg text-text">{e.title}</h3>
            <p className="mt-1 max-w-2xl text-muted">{e.text}</p>
            {e.metric && (
              <div className="mt-4 max-w-xl">
                <MetricView metric={e.metric} reduced={reduced} />
              </div>
            )}
          </motion.li>
        ))}
      </ol>
    </div>
  );
}
