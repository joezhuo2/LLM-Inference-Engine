"use client";

import clsx from "clsx";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { type ReactNode, useEffect, useRef, useState } from "react";
import { AttentionFigure, steps } from "./figure";

const ease = [0.23, 1, 0.32, 1] as const;

function StoryPage({ children }: { children: ReactNode }) {
  return (
    <article className="space-y-8">
      <header className="space-y-3">
        <div className="font-mono text-xs text-muted">kernels / 01</div>
        <h1 className="font-mono text-2xl sm:text-3xl">paged attention: v1 to v2 to split</h1>
        <p className="max-w-2xl text-muted">
          Decode attention reads one query per sequence against its whole cached context. Three versions of the kernel,
          each fixing what limited the one before.
        </p>
      </header>
      {children}
      <p className="font-mono text-xs text-muted">
        Times per launch, RTX 5060 Laptop GPU, 32 heads, 4 KV heads, head_dim 64; details in docs/profiling.md.
      </p>
    </article>
  );
}

export function Sticky() {
  const [active, setActive] = useState(0);
  const refs = useRef<(HTMLElement | null)[]>([]);
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting) setActive(Number((e.target as HTMLElement).dataset.step));
        }
      },
      { rootMargin: "-45% 0px -45% 0px" },
    );
    refs.current.forEach((el) => el && observer.observe(el));
    return () => observer.disconnect();
  }, []);
  return (
    <StoryPage>
      <div className="grid gap-8 md:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)]">
        <div className="order-2 md:order-1">
          {steps.map((s, i) => (
            <section
              key={s.id}
              data-step={i}
              ref={(el) => {
                refs.current[i] = el;
              }}
              className="flex min-h-[70vh] flex-col justify-center"
            >
              <h2 className={clsx("font-mono text-lg", i === active ? "text-accent" : "text-muted")}>{s.title}</h2>
              <p className="mt-3 text-muted">{s.text}</p>
            </section>
          ))}
        </div>
        <div className="order-1 md:order-2">
          <div className="sticky top-16 rounded border border-border bg-surface p-4">
            <AttentionFigure step={active} />
          </div>
        </div>
      </div>
    </StoryPage>
  );
}

export function Inline() {
  const reduced = useReducedMotion();
  return (
    <StoryPage>
      <div className="space-y-16">
        {steps.map((s, i) => (
          <section key={s.id} className="space-y-4">
            <h2 className="font-mono text-lg text-accent">{s.title}</h2>
            <p className="max-w-2xl text-muted">{s.text}</p>
            <motion.div
              className="rounded border border-border bg-surface p-4"
              initial={reduced ? false : { opacity: 0, transform: "translateY(8px)" }}
              whileInView={{ opacity: 1, transform: "translateY(0px)" }}
              viewport={{ once: true, margin: "-80px" }}
              transition={{ duration: 0.3, ease }}
            >
              <AttentionFigure step={i} />
            </motion.div>
          </section>
        ))}
      </div>
    </StoryPage>
  );
}

export function Stepper() {
  const [step, setStep] = useState(0);
  const reduced = useReducedMotion();
  return (
    <StoryPage>
      <div className="overflow-hidden rounded border border-border bg-surface">
        <div className="p-4">
          <AttentionFigure step={step} />
        </div>
        <div className="flex border-t border-border font-mono text-sm" role="tablist">
          {steps.map((s, i) => (
            <button
              key={s.id}
              type="button"
              role="tab"
              aria-selected={i === step}
              onClick={() => setStep(i)}
              className={clsx(
                "pressable flex-1 border-r border-border px-3 py-2 text-left last:border-r-0",
                i === step ? "bg-accent-dim text-accent" : "text-muted",
              )}
            >
              <span className="text-muted">{String(i + 1).padStart(2, "0")} </span>
              {s.id}
            </button>
          ))}
        </div>
        <div className="min-h-40 border-t border-border p-4">
          <AnimatePresence mode="wait" initial={false}>
            <motion.div
              key={step}
              initial={reduced ? false : { opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.15, ease }}
            >
              <h2 className="font-mono text-lg text-accent">{steps[step].title}</h2>
              <p className="mt-2 text-muted">{steps[step].text}</p>
            </motion.div>
          </AnimatePresence>
        </div>
        <div className="flex justify-between border-t border-border p-3 font-mono text-xs">
          <button
            type="button"
            disabled={step === 0}
            onClick={() => setStep((s) => s - 1)}
            className="pressable rounded border border-border px-3 py-1 text-muted disabled:opacity-40"
          >
            previous
          </button>
          <button
            type="button"
            disabled={step === steps.length - 1}
            onClick={() => setStep((s) => s + 1)}
            className="pressable rounded border border-border px-3 py-1 text-muted disabled:opacity-40"
          >
            next
          </button>
        </div>
      </div>
    </StoryPage>
  );
}
