"use client";

import NumberFlow from "@/components/number-flow";
import clsx from "clsx";
import { useEffect, useMemo, useRef, useState } from "react";
import { SmoothRange } from "@/components/smooth-range";
import { glide, type Tween } from "@/lib/glide";
import { usePrefersReducedMotion } from "@/lib/reduced-motion";
import { EasedBar } from "../eased-bar";
import { Toggle } from "../toggle";
import { START_MAX, directAttention, makeProblem, maxDiff, pagedAttention } from "./online-softmax";

const defaults = { context: 100, block: "16", seed: 15 };
const engineHeadDim = 64;
const playMs = 900;
const sizes = ["8", "16", "32"];
const fade = "transition-opacity duration-500 [transition-timing-function:var(--ease-in-out)] motion-reduce:transition-none";
const three = { minimumFractionDigits: 3, maximumFractionDigits: 3 } as const;
const button = "pressable rounded border border-border px-2.5 py-1 text-text disabled:opacity-40";

const options = [
  { id: "keep", text: "nothing: l and the accumulator stay, the new block's weights are added to them" },
  { id: "scale", text: "both are multiplied by exp(old max - new max), a number below 1, before the new weights are added" },
  { id: "redo", text: "the kernel goes back and recomputes the blocks it already read" },
];

function Num({ value }: { value: number }) {
  return <NumberFlow value={value} format={three} />;
}

export function PagedAttentionSandbox() {
  const [context, setContext] = useState(defaults.context);
  const [block, setBlock] = useState(defaults.block);
  const [seed, setSeed] = useState(defaults.seed);
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [guess, setGuess] = useState<string | null>(null);
  const reduced = usePrefersReducedMotion();
  const tween = useRef<Tween | null>(null);

  useEffect(() => () => tween.current?.stop(), []);

  const problem = useMemo(() => makeProblem(context, Number(block), seed), [context, block, seed]);
  const run = useMemo(() => pagedAttention(problem), [problem]);
  const direct = useMemo(() => directAttention(problem), [problem]);
  const blocks = problem.table.length;
  const last = step === blocks;
  const now = step > 0 ? run.steps[step - 1] : null;

  const running = playing && !last;

  useEffect(() => {
    if (!running) return;
    const timer = setInterval(() => setStep((s) => Math.min(s + 1, blocks)), playMs);
    return () => clearInterval(timer);
  }, [running, blocks]);

  const change = (apply: () => void) => {
    tween.current?.stop();
    apply();
    setStep(0);
    setPlaying(false);
  };
  const reset = () => {
    tween.current?.stop();
    if (reduced || context === defaults.context) setContext(defaults.context);
    else tween.current = glide(context, defaults.context, (v) => setContext(Math.round(v)));
    setBlock(defaults.block);
    setSeed(defaults.seed);
    setStep(0);
    setPlaying(false);
    setGuess(null);
  };
  const nextRise = run.steps.findIndex((s, i) => i + 1 > step && s.logical > 0 && s.grew);
  const tokensRead = run.steps.slice(0, step).reduce((n, s) => n + s.tokens, 0);
  const soFar = now ? now.accAfter.map((a) => (now.sumAfter > 0 ? a / now.sumAfter : 0)) : problem.q.map(() => 0);
  const diff = last ? maxDiff(run.out, direct) : null;

  return (
    <div className="grid gap-6 font-mono text-xs lg:grid-cols-[minmax(0,1fr)_19rem]">
      <div className="space-y-4">
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
          <label className="flex items-center gap-2 text-muted">
            <span className="w-14 shrink-0">context</span>
            <SmoothRange
              min={1}
              max={256}
              value={context}
              onValue={(v) => change(() => setContext(v))}
              onPointerDown={() => tween.current?.stop()}
              aria-label="context length in tokens"
              className="w-40 accent-[var(--accent)]"
            />
            <span className="w-20 text-text tabular-nums">
              <NumberFlow value={context} suffix=" tokens" />
            </span>
          </label>
          <Toggle label="block" choices={sizes} value={block} onChange={(v) => change(() => setBlock(v))} />
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <button type="button" disabled={step === 0} onClick={() => {
              setPlaying(false);
              setStep(step - 1);
            }} className={button}>
            back
          </button>
          <button type="button" disabled={last} onClick={() => {
              setPlaying(false);
              setStep(step + 1);
            }} className={clsx(button, "text-accent")}>
            next block
          </button>
          <button
            type="button"
            aria-pressed={running}
            onClick={() => {
              if (last) setStep(0);
              setPlaying(!running);
            }}
            className={clsx(button, "text-accent")}
          >
            {running ? "pause" : "play"}
          </button>
          <button type="button" onClick={() => change(() => setSeed(seed + 1))} className={button}>
            new random data
          </button>
          <button type="button" onClick={reset} className={button}>
            reset
          </button>
        </div>

        <div className="grid gap-4 sm:grid-cols-[13rem_minmax(0,1fr)]">
          <div className="space-y-1.5">
            <div className="text-muted">{`pool of ${problem.poolBlocks} blocks`}</div>
            <div role="img" aria-label={`A pool of ${problem.poolBlocks} physical blocks; the context occupies ${blocks} of them, scattered, ${step} already read`} className="grid grid-cols-8 gap-1">
              {Array.from({ length: problem.poolBlocks }, (_, id) => {
                const logical = problem.table.indexOf(id);
                const used = logical >= 0;
                return (
                  <div key={id} className={clsx("relative flex aspect-square items-center justify-center rounded-sm border text-[9px] tabular-nums", used ? "border-muted text-text" : "border-border text-muted opacity-50")}>
                    {used ? logical : id === 0 ? "∅" : ""}
                    {used && <span aria-hidden className={clsx("absolute inset-0 rounded-sm bg-accent-dim", fade, logical < step ? "opacity-100" : "opacity-0")} />}
                    {used && <span aria-hidden className={clsx("absolute inset-0 rounded-sm border-2 border-accent", fade, logical === step - 1 ? "opacity-100" : "opacity-0")} />}
                  </div>
                );
              })}
            </div>
            <div className="text-muted">number = logical block; ∅ = block 0, the null block</div>
          </div>
          <div className="space-y-1.5">
            <div className="text-muted">block table, logical → physical (click to jump)</div>
            <ul className="grid h-56 grid-cols-2 content-start gap-1 overflow-y-auto pr-1 sm:grid-cols-3">
              {problem.table.map((physical, logical) => (
                <li key={logical}>
                  <button
                    type="button"
                    onClick={() => {
                      setPlaying(false);
                      setStep(logical + 1);
                    }}
                    aria-label={`logical block ${logical}, physical block ${physical}; show this step`}
                    aria-current={logical === step - 1 ? "step" : undefined}
                    className={clsx("pressable w-full rounded border px-1.5 py-0.5 text-left tabular-nums", logical === step - 1 ? "border-accent text-accent" : logical < step ? "border-border text-text" : "border-border text-muted")}
                  >
                    {`${logical} → ${physical}`}
                  </button>
                </li>
              ))}
            </ul>
          </div>
        </div>

        <div className="space-y-3 rounded border border-border p-3" aria-live="polite">
          {now ? (
            <>
              <div className="text-text">{`block ${now.logical} (physical ${now.physical}): ${now.tokens} key${now.tokens === 1 ? "" : "s"} read, scores q · k / √${problem.headDim}`}</div>
              <div className="flex flex-wrap gap-1.5">
                {now.scores.map((s, i) => (
                  <div key={i} className={clsx("w-11 space-y-1", s === now.blockMax ? "text-accent" : "text-muted")}>
                    <div className="tabular-nums">{s.toFixed(2)}</div>
                    <div className="h-1 rounded-full bg-border">
                      <EasedBar fraction={now.weights[i]} className="bg-accent" />
                    </div>
                  </div>
                ))}
              </div>
              <div className="text-muted">bar = weight exp(score - new m); the best score of the block is green</div>
              <table className="w-full tabular-nums">
                <tbody>
                  <tr className="border-t border-border">
                    <td className="py-1 pr-2 text-muted">m</td>
                    <td className="py-1 text-text">
                      {now.maxBefore === START_MAX ? "start" : now.maxBefore.toFixed(3)} → {now.maxAfter.toFixed(3)}
                      <span className={clsx("ml-2", now.grew ? "text-warn" : "text-muted")}>{now.logical === 0 ? "first block" : now.grew ? "raised" : "unchanged"}</span>
                    </td>
                  </tr>
                  <tr className="border-t border-border">
                    <td className="py-1 pr-2 text-muted">rescale</td>
                    <td className={clsx("py-1", now.rescale < 1 && now.logical > 0 ? "text-warn" : "text-text")}>{`exp(m old - m new) = ${now.rescale.toFixed(3)}`}</td>
                  </tr>
                  <tr className="border-t border-border">
                    <td className="py-1 pr-2 text-muted">l</td>
                    <td className="py-1 text-text">{`${now.sumBefore.toFixed(3)} × ${now.rescale.toFixed(3)} + Σ weights ${now.weights.reduce((a, w) => a + w, 0).toFixed(3)} = ${now.sumAfter.toFixed(3)}`}</td>
                  </tr>
                  <tr className="border-t border-border">
                    <td className="py-1 pr-2 text-muted">acc</td>
                    <td className="py-1 text-text">{`[${now.accBefore.map((a) => a.toFixed(2)).join(", ")}] × ${now.rescale.toFixed(3)} + Σ weight · v = [${now.accAfter.map((a) => a.toFixed(2)).join(", ")}]`}</td>
                  </tr>
                </tbody>
              </table>
            </>
          ) : (
            <div className="text-muted">{`Nothing read yet: m is the kernel's start value (${START_MAX.toExponential(0)}), l and acc are 0. Press next block.`}</div>
          )}
        </div>
        <p className="text-muted">
          {`Illustrative: head dim ${problem.headDim}, random seeded values (new random data reshuffles the blocks and the numbers), float32. The engine uses head dim ${engineHeadDim} and reads 256 tokens per step (16 blocks); here one block per step, with the same order of operations.`}
        </p>
      </div>

      <div className="space-y-5">
        <table className="w-full tabular-nums">
          <tbody>
            <tr className="border-t border-border">
              <td className="py-1 text-muted">blocks read</td>
              <td className="py-1 text-right text-text">
                <NumberFlow value={step} suffix={` / ${blocks}`} />
              </td>
            </tr>
            <tr className="border-t border-border">
              <td className="py-1 text-muted">{`bytes read (head dim ${engineHeadDim})`}</td>
              <td className="py-1 text-right text-text">
                <NumberFlow value={tokensRead * engineHeadDim * 2 * 2} suffix=" B" />
              </td>
            </tr>
            <tr className="border-t border-border">
              <td className="py-1 text-muted">m, running max</td>
              <td className="py-1 text-right text-text">{now ? <Num value={now.maxAfter} /> : "start"}</td>
            </tr>
            <tr className="border-t border-border">
              <td className="py-1 text-muted">l, running sum</td>
              <td className="py-1 text-right text-text">
                <Num value={now?.sumAfter ?? 0} />
              </td>
            </tr>
          </tbody>
        </table>
        <div className="text-[11px] text-muted">bytes: tokens read × 64 dims × K and V × 2 B (BF16), one head</div>

        <div className="space-y-2">
          <div className="text-muted">{last ? "final: acc / l against one-shot softmax" : "acc / l so far against one-shot softmax"}</div>
          <table className="w-full tabular-nums">
            <thead className="text-left text-muted">
              <tr>
                <th className="pb-1 font-normal">dim</th>
                <th className="pb-1 text-right font-normal">online</th>
                <th className="pb-1 text-right font-normal">one go</th>
              </tr>
            </thead>
            <tbody>
              {direct.map((d, i) => (
                <tr key={i} className="border-t border-border">
                  <td className="py-1 text-muted">{i}</td>
                  <td className={clsx("py-1 text-right", last ? "text-accent" : "text-text")}>
                    <Num value={last ? run.out[i] : soFar[i]} />
                  </td>
                  <td className="py-1 text-right text-text">{d.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className={clsx(diff !== null && diff < 1e-5 ? "text-accent" : "text-muted")}>
            {diff === null ? "read every block to compare" : `max |difference| ${diff.toExponential(1)}: ${diff < 1e-5 ? "equal to float precision" : "MISMATCH"}`}
          </div>
        </div>

        <div className="space-y-2 rounded border border-border p-3">
          <div className="text-text">predict first</div>
          <div id="pa-predict" className="text-muted">
            A block arrives whose best score is bigger than any before. What happens to l and the accumulator?
          </div>
          <div role="radiogroup" aria-labelledby="pa-predict" className="space-y-1.5">
            {options.map((o) => (
              <button
                key={o.id}
                type="button"
                role="radio"
                aria-checked={guess === o.id}
                onClick={() => setGuess(o.id)}
                className={clsx("pressable w-full rounded border px-2 py-1 text-left", guess === o.id ? (o.id === "scale" ? "border-accent text-accent" : "border-warn text-warn") : "border-border text-text")}
              >
                {o.text}
              </button>
            ))}
          </div>
          {guess && (
            <p className={guess === "scale" ? "text-accent" : "text-warn"}>
              {guess === "scale"
                ? "Right. Every earlier weight was measured against the old max, so one factor exp(old - new) re-expresses them against the new one: no block is read twice."
                : "Not quite. Earlier weights were exp(score - old max); with a new max they must become exp(score - new max), which is the same numbers times exp(old - new), so l and acc are scaled by it and nothing is recomputed."}
            </p>
          )}
          <button
            type="button"
            disabled={nextRise < 0}
            onClick={() => {
              setPlaying(false);
              setStep(nextRise + 1);
            }}
            className={button}
          >
            {nextRise < 0 ? "no later block raises m" : "jump to the next raised m"}
          </button>
        </div>
      </div>
    </div>
  );
}
