"use client";

import { AnimatePresence, motion, type MotionValue, useMotionValueEvent, useScroll, useTransform } from "motion/react";
import { createElement, useRef, useState } from "react";
import { GemmTilesFigure, GpuFloorplan, ReductionRaceFigure } from "@/components/gpu-floorplan";
import { GenerateFigure, RooflineFigure, SampleFigure, TokenizeFigure } from "@/components/scene-figures";
import { useActiveStep } from "@/components/kernels/use-active-step";
import { Scramble } from "@/components/scramble";
import { type Entry, type Metric, type Milestone, milestones, repoUrl } from "@/content/build-log";
import { usePrefersReducedMotion } from "@/lib/reduced-motion";

const ease = [0.23, 1, 0.32, 1] as const;
const inView = { once: true, margin: "0px 0px -15% 0px" };

type Scene = NonNullable<Entry["scene"]>;

const entries = milestones.flatMap((m) => m.entries);
const starts = milestones.map((m) => entries.indexOf(m.entries[0]));
const sceneAt = entries.map((_, i) => entries.slice(0, i + 1).reverse().find((e) => e.scene)?.scene);

const captions: Record<Scene | "idle", string> = {
  idle: "The engine has not reached the GPU memory hierarchy yet.",
  bandwidth: "A bandwidth benchmark streams global VRAM into the SMs: the read speed every later kernel is judged against.",
  reduce: "One sum three ways: global atomics, a tree in shared memory, then warp shuffles that stay in registers.",
  gemm: "BF16 tiles of A and B pass through the tensor cores into C; cuBLAS gets the transposed product, so nothing is copied.",
  weights: "The safetensors file goes from disk into VRAM through pinned buffers, then every tensor is checksummed against PyTorch.",
  smem: "The row is read into one SM once; its sum of squares stays in registers and shared memory instead of a round trip through VRAM.",
  vram: "write_kv_kernel looks up the block table and writes each new K and V into a KV page block; a full block gets a fresh one from the pool.",
  tokenize: "The prompt is wrapped in TinyLlama's chat template, cut into SentencePiece pieces, and each piece becomes an integer id.",
  generate: "The prompt goes in, then each decode step picks the most likely next token from the model's probabilities until the reply is complete.",
  sample: "Temperature reshapes the logits, Gumbel noise is added to each, and the largest sum wins; the same seed makes the same pick.",
  roofline: "On a log-log roofline the batch-1 matrix-vector products sit right under the bandwidth roof, which is why the whole step reaches most of it.",
  warps: "Warps read the KV blocks a chunk at a time with a running max and sum in registers, so shared memory never holds the whole context.",
};

const figures = { reduce: ReductionRaceFigure, gemm: GemmTilesFigure, tokenize: TokenizeFigure, generate: GenerateFigure, sample: SampleFigure, roofline: RooflineFigure };

const isStandalone = (s: Scene | "idle"): s is keyof typeof figures => s in figures;

type Go = (t: number, behavior: ScrollBehavior) => void;

function Scrubber({ progress, active, reduced, go }: { progress: MotionValue<number>; active: number; reduced: boolean; go: Go }) {
  const [value, setValue] = useState(0);
  const down = useRef(0);
  const target = useRef(0);
  const last = entries.length - 1;
  const behavior: ScrollBehavior = reduced ? "instant" : "smooth";
  const held = () => performance.now() - down.current > 250;
  useMotionValueEvent(progress, "change", setValue);
  return (
    <>
      <input
        type="range"
        min={0}
        max={last}
        step="any"
        value={value}
        onChange={(e) => {
          target.current = Number(e.target.value);
          if (!down.current) go(target.current, behavior);
          else if (held()) go(target.current, "instant");
        }}
        onPointerDown={() => {
          down.current = performance.now();
        }}
        onPointerUp={() => {
          if (!held()) go(target.current, behavior);
          down.current = 0;
        }}
        onPointerCancel={() => {
          down.current = 0;
        }}
        aria-label="timeline position"
        aria-valuetext={entries[active].title}
        className="mt-3 block w-full accent-accent"
      />
      <div className="relative mt-1 h-4 font-mono text-[10px] text-muted">
        {milestones.map((m, i) => (
          <button
            key={m.id}
            type="button"
            title={m.name}
            aria-label={`${m.id} ${m.name}`}
            onClick={() => go(starts[i], behavior)}
            className="link-hover absolute -translate-x-1/2 tabular-nums"
            style={{ left: `calc(${(100 * starts[i]) / last}% + ${0.5 - starts[i] / last}rem)` }}
          >
            {m.id}
          </button>
        ))}
      </div>
    </>
  );
}

function GpuPanel({ active, progress, reduced, go }: { active: number; progress: MotionValue<number>; reduced: boolean; go: Go }) {
  const scene = sceneAt[active] ?? "idle";
  return (
    <div className="mx-auto max-w-md rounded border border-border bg-surface p-3 sm:p-4 md:max-w-none">
      <div className="grid aspect-[14/10] w-full">
        <AnimatePresence>
          <motion.div
            key={isStandalone(scene) ? scene : "floorplan"}
            className="col-start-1 row-start-1 min-w-0"
            initial={reduced ? false : { opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: reduced ? 0 : 0.3, ease }}
          >
            {isStandalone(scene) ? createElement(figures[scene]) : <GpuFloorplan flow={scene} />}
          </motion.div>
        </AnimatePresence>
      </div>
      <p className="mt-3 min-h-10 border-t border-border pt-3 font-mono text-[11px] leading-relaxed text-muted">{captions[scene]}</p>
      <Scrubber progress={progress} active={active} reduced={reduced} go={go} />
    </div>
  );
}

export function MetricView({ metric, reduced }: { metric: Metric; reduced: boolean }) {
  if (metric.kind === "value") {
    return (
      <div className="font-mono">
        <div className="text-2xl text-accent tabular-nums">
          <Scramble text={metric.value} trigger="view" durationMs={1200} />
        </div>
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
                <Scramble
                  text={`${bar.value.toFixed(metric.digits)}${metric.unit ? ` ${metric.unit}` : ""}`}
                  trigger="view"
                  delayMs={150 + i * 80}
                  durationMs={900}
                />
              </span>
            </div>
          </div>
        ))}
      </div>
    </figure>
  );
}

function EntryView({ entry, reduced, stepRef }: { entry: Entry; reduced: boolean; stepRef: (el: HTMLLIElement | null) => void }) {
  return (
    <motion.li
      ref={stepRef}
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
      <h3 className="mt-2 text-lg text-text">
        <Scramble text={entry.title} trigger="view" durationMs={900} />
      </h3>
      <p className="mt-1 max-w-xl text-muted">{entry.text}</p>
      {entry.metric && (
        <div className="mt-4 max-w-xl">
          <MetricView metric={entry.metric} reduced={reduced} />
        </div>
      )}
    </motion.li>
  );
}

function MilestoneView({ milestone, reduced, start, refs }: { milestone: Milestone; reduced: boolean; start: number; refs: { current: (HTMLElement | null)[] } }) {
  const ref = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({ target: ref, offset: ["start 70%", "end 70%"] });
  const fill = useTransform(scrollYProgress, (p) => `scaleY(${p})`);
  return (
    <section aria-labelledby={`milestone-${milestone.id}`} className="space-y-6">
      <header>
        <div className="font-mono text-xs text-muted tabular-nums">
          {milestone.id} · {milestone.dates}
        </div>
        <h2 id={`milestone-${milestone.id}`} className="mt-1 font-mono text-xl text-text">
          <Scramble text={milestone.name} trigger="view" durationMs={900} />
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
          {milestone.entries.map((entry, i) => (
            <EntryView
              key={entry.branch}
              entry={entry}
              reduced={reduced}
              stepRef={(el) => {
                refs.current[start + i] = el;
              }}
            />
          ))}
        </ol>
      </div>
    </section>
  );
}

export function BuildTimeline() {
  const reduced = usePrefersReducedMotion();
  const panel = useRef<HTMLDivElement>(null);
  const { active, progress, refs, scrollTo } = useActiveStep(panel);
  const prs = entries.flatMap((e) => e.prs).length;
  return (
    <div className="md:grid md:grid-cols-[minmax(0,3fr)_minmax(0,2fr)] md:gap-10">
      <div ref={panel} className="sticky top-12 z-10 -mx-4 mb-8 border-b border-border bg-bg px-4 pb-3 pt-2 sm:mx-0 sm:px-0 md:top-20 md:mb-0 md:self-start md:border-0 md:bg-transparent md:p-0">
        <GpuPanel active={active} progress={progress} reduced={reduced} go={scrollTo} />
      </div>
      <div className="space-y-20">
        {milestones.map((m, i) => (
          <MilestoneView key={m.id} milestone={m} reduced={reduced} start={starts[i]} refs={refs} />
        ))}
        <p className="font-mono text-xs text-muted">
          {prs} of the pull requests shown;{" "}
          <a href={`${repoUrl}/pulls?q=is%3Apr+is%3Amerged`} className="link-underline text-text">
            all of them on GitHub
          </a>
        </p>
      </div>
    </div>
  );
}
