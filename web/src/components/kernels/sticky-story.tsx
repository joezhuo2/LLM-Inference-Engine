"use client";

import clsx from "clsx";
import { AnimatePresence, motion } from "motion/react";
import { type ReactNode, useEffect, useRef, useState } from "react";
import { usePrefersReducedMotion } from "@/lib/reduced-motion";

export type StoryStep = { id: string; title: string; body: ReactNode };

const ease = [0.23, 1, 0.32, 1] as const;

export function StickyStory({
  steps,
  figure,
  caption,
}: {
  steps: StoryStep[];
  figure: (step: number) => ReactNode;
  caption: (step: number) => ReactNode;
}) {
  const [active, setActive] = useState(0);
  const refs = useRef<(HTMLElement | null)[]>([]);
  const reduced = usePrefersReducedMotion();

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting) setActive(Number((e.target as HTMLElement).dataset.step));
        }
      },
      { rootMargin: "-40% 0px -40% 0px" },
    );
    refs.current.forEach((el) => el && observer.observe(el));
    return () => observer.disconnect();
  }, []);

  return (
    <div className="md:grid md:grid-cols-[minmax(0,0.85fr)_minmax(0,1.15fr)] md:gap-10">
      <div className="sticky top-12 z-10 -mx-4 mb-2 border-b border-border bg-bg px-4 pb-3 pt-2 sm:mx-0 sm:px-0 md:top-20 md:col-start-2 md:row-start-1 md:mb-0 md:self-start md:border-0 md:bg-transparent md:p-0">
        <div className="rounded border border-border bg-surface p-3 sm:p-4">
          <div className="relative aspect-[14/10] w-full">
            <AnimatePresence initial={false}>
              <motion.div
                key={active}
                className="absolute inset-0"
                initial={reduced ? false : { opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: reduced ? 0 : 0.3, ease }}
              >
                {figure(active)}
              </motion.div>
            </AnimatePresence>
          </div>
          <div className="mt-3 min-h-10 border-t border-border pt-3 font-mono text-[11px] leading-relaxed text-muted">
            {caption(active)}
          </div>
        </div>
      </div>
      <div className="md:col-start-1 md:row-start-1">
        {steps.map((s, i) => (
          <section
            key={s.id}
            aria-current={i === active ? "step" : undefined}
            className="flex min-h-[70svh] flex-col justify-center py-8 first:pt-0 md:min-h-[80svh]"
          >
            <div
              data-step={i}
              ref={(el) => {
                refs.current[i] = el;
              }}
            >
              <div className="font-mono text-xs text-muted tabular-nums">
                {String(i + 1).padStart(2, "0")} / {String(steps.length).padStart(2, "0")}
              </div>
              <h2
                className={clsx(
                  "mt-1 font-mono text-lg transition-colors duration-200",
                  i === active ? "text-accent" : "text-muted",
                )}
              >
                {s.title}
              </h2>
              <div className="mt-3 space-y-3 text-muted">{s.body}</div>
            </div>
          </section>
        ))}
      </div>
    </div>
  );
}
