"use client";

import NumberFlow from "@/components/number-flow";
import clsx from "clsx";
import { useEffect, useRef, useState } from "react";
import { SmoothRange } from "@/components/smooth-range";
import { glide as tween, type Tween } from "@/lib/glide";
import { samplingLogits } from "@/lib/data";
import { usePrefersReducedMotion } from "@/lib/reduced-motion";
import { EasedBar } from "../eased-bar";
import { Toggle } from "../toggle";
import { buildGroups, draw, type Group, type Knobs, stages, type Stages } from "./sampler";

const dump = samplingLogits();
const contexts = dump.contexts.map((c) => ({ ...c, groups: buildGroups(c) }));
const names = contexts.map((c) => c.name);
const stageNames = ["logits", "÷ T", "top-k", "top-p", "final"];
const SHOW = 12;
const KOFF = 201;
const defaults = { temperature: 1, k: KOFF, p: 1 };
const historyLength = 24;
const snap = (v: number, step: number) => Math.round(v / step) * step;

const fmt = (v: number) => v.toLocaleString("en-US");
const piece = (t: string) => (t === "<0x0A>" ? "⏎" : t.replaceAll("▁", "␣"));
const nextSeed = (s: number) => (s * 48271 + 11) % 2147483647;
const niceAxis = (max: number) => [0.05, 0.1, 0.2, 0.25, 0.5, 1].find((a) => a >= max) ?? 1;
const knobsOf = (temperature: number, k: number, p: number): Knobs => ({ temperature, topK: k >= KOFF ? 0 : k, topP: p });

type Row = { label: string; count: number; logit: number; s: Stages; at: number[]; keptK: boolean; kept: boolean; cum: number };

function sumRange(v: number[], from: number) {
  return v.slice(from).reduce((a, b) => a + b, 0);
}

function rowsOf(groups: Group[], s: Stages): Row[] {
  const norm = s.afterK.reduce((a, b) => a + b, 0);
  const pick = (i: number) => [s.scaled[i], s.base[i], s.afterK[i], s.afterP[i], s.final[i]];
  const rows: Row[] = groups.slice(0, SHOW).map((g, i) => ({ label: piece(g.label), count: g.count, logit: g.logit, s, at: pick(i), keptK: s.keptK[i], kept: s.kept[i], cum: 0 }));
  rows.push({
    label: `${fmt(groups.slice(SHOW).reduce((a, g) => a + g.count, 0))} others`,
    count: groups.slice(SHOW).reduce((a, g) => a + g.count, 0),
    logit: groups[SHOW].logit,
    s,
    at: [groups[SHOW].logit, sumRange(s.base, SHOW), sumRange(s.afterK, SHOW), sumRange(s.afterP, SHOW), sumRange(s.final, SHOW)],
    keptK: s.keptK.slice(SHOW).some(Boolean),
    kept: s.kept.slice(SHOW).some(Boolean),
    cum: 0,
  });
  let run = 0;
  for (const r of rows) {
    run += r.at[2] / norm;
    r.cum = Math.min(1, run);
  }
  return rows;
}

function Slider({ label, value, min, max, step, onChange, children }: { label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void; children: React.ReactNode }) {
  return (
    <label className="flex items-center gap-3">
      <span className="w-14 shrink-0 text-muted">{label}</span>
      <SmoothRange min={min} max={max} step={step} value={value} onValue={onChange} className="w-full accent-[var(--accent)]" />
      <span className="w-14 shrink-0 text-right tabular-nums text-text">{children}</span>
    </label>
  );
}

export function SamplingSandbox() {
  const [ctx, setCtx] = useState(names[0]);
  const [temperature, setTemperature] = useState(defaults.temperature);
  const [k, setK] = useState(defaults.k);
  const [p, setP] = useState(defaults.p);
  const [stage, setStage] = useState("final");
  const [seed, setSeed] = useState(1);
  const [step, setStep] = useState(0);
  const [history, setHistory] = useState<{ label: string; index: number }[]>([]);
  const [guess, setGuess] = useState<string | null>(null);

  const reduced = usePrefersReducedMotion();
  const run = useRef<Tween | null>(null);
  useEffect(() => () => run.current?.stop(), []);

  const glide = (to: { temperature: number; k: number; p: number }) => {
    run.current?.stop();
    if (reduced) {
      setTemperature(to.temperature);
      setK(to.k);
      setP(to.p);
      return;
    }
    const from = { temperature, k, p };
    run.current = tween(0, 1, (x) => {
      setTemperature(snap(from.temperature + (to.temperature - from.temperature) * x, 0.05));
      setK(Math.round(from.k + (to.k - from.k) * x));
      setP(snap(from.p + (to.p - from.p) * x, 0.05));
    });
  };

  const user = (set: (v: number) => void) => (v: number) => {
    run.current?.stop();
    set(v);
  };

  const context = contexts[names.indexOf(ctx)];
  const knobs = knobsOf(temperature, k, p);
  const s = stages(context.groups, knobs);
  const rows = rowsOf(context.groups, s);
  const stageIndex = stageNames.indexOf(stage);
  const probabilityStage = stageIndex > 0;
  const axis = probabilityStage ? niceAxis(Math.max(...rows.map((r) => r.at[stageIndex]))) : 1;
  const maxLogit = context.groups[0].logit;
  const greedy = temperature <= 0;
  const lastDraw = history.length ? history[history.length - 1].index : -1;
  const lastRow = lastDraw < 0 ? -1 : Math.min(lastDraw, SHOW);
  const distinct = new Set(history.map((h) => h.label)).size;

  const survivorsAt = (t: number) => stages(context.groups, knobsOf(t, KOFF, 0.9)).survivors;
  const before = survivorsAt(0.7);
  const after = survivorsAt(1.5);
  const truth = after > before ? "grow" : after < before ? "shrink" : "same";

  const sample = (n: number) => {
    const drawn = Array.from({ length: n }, (_, i) => {
      const index = draw(context.groups, knobs, seed, step + i);
      const g = context.groups[index];
      return { label: g.label === "other" ? "(other)" : piece(g.label), index };
    });
    setStep(step + n);
    setHistory((h) => [...h, ...drawn].slice(-historyLength));
  };

  const pickContext = (v: string) => {
    setCtx(v);
    setHistory([]);
    setStep(0);
  };

  const reset = () => {
    setCtx(names[0]);
    glide({ temperature: defaults.temperature, k: defaults.k, p: defaults.p });
    setStage("final");
    setSeed(1);
    setStep(0);
    setHistory([]);
    setGuess(null);
  };

  const readouts = [
    { label: "candidates left", value: <NumberFlow value={s.survivors} format={{ maximumFractionDigits: 0 }} /> },
    { label: "entropy of what is left", value: <NumberFlow value={s.entropy} suffix=" bits" format={{ minimumFractionDigits: 2, maximumFractionDigits: 2 }} /> },
    { label: "most likely token", value: <NumberFlow value={Math.max(...s.final)} format={{ style: "percent", minimumFractionDigits: 1, maximumFractionDigits: 1 }} /> },
    { label: "mass cut by top-k and top-p", value: <NumberFlow value={1 - s.afterP.reduce((a, b) => a + b, 0)} format={{ style: "percent", minimumFractionDigits: 1, maximumFractionDigits: 1 }} /> },
    { label: "different tokens drawn", value: <NumberFlow value={distinct} suffix={` of ${history.length}`} /> },
  ];

  return (
    <div className="grid gap-6 font-mono text-xs lg:grid-cols-[minmax(0,1fr)_19rem]">
      <div className="space-y-4">
        <div className="space-y-2">
          <Toggle label="prompt" choices={names} value={ctx} onChange={pickContext} />
          <p className="leading-relaxed text-muted">
            {`${context.note}: "${context.user}"${context.answer_so_far ? ` with the answer so far "${context.answer_so_far}"` : ""}. Next token, ${fmt(context.groups.reduce((a, g) => a + g.count, 0))} to choose from.`}
          </p>
        </div>

        <div className="space-y-2">
          <Slider label="temp T" value={temperature} min={0} max={2} step={0.05} onChange={user(setTemperature)}>
            {greedy ? "greedy" : <NumberFlow value={temperature} format={{ minimumFractionDigits: 2, maximumFractionDigits: 2 }} />}
          </Slider>
          <Slider label="top-k" value={k} min={1} max={KOFF} step={1} onChange={user(setK)}>
            {k >= KOFF ? "off" : <NumberFlow value={k} />}
          </Slider>
          <Slider label="top-p" value={p} min={0.05} max={1} step={0.05} onChange={user(setP)}>
            {p >= 1 ? "off" : <NumberFlow value={p} format={{ minimumFractionDigits: 2, maximumFractionDigits: 2 }} />}
          </Slider>
          {greedy && <p className="text-warn">T = 0 is greedy: the engine takes the largest logit and ignores top-k and top-p.</p>}
        </div>

        <Toggle label="stage" choices={stageNames} value={stage} onChange={setStage} />

        <div role="img" aria-label={`The ${SHOW} likeliest next tokens and the rest, at the ${stage} stage: ${s.survivors} candidates left, entropy ${s.entropy.toFixed(2)} bits`} className="space-y-1.5">
          {rows.map((r, i) => {
            const v = r.at[stageIndex];
            const dim = (stageIndex >= 2 && !r.keptK) || (stageIndex >= 3 && !r.kept);
            return (
              <div key={i} className="grid grid-cols-[6.5rem_minmax(0,1fr)_3.75rem] items-center gap-2">
                <span className={clsx("truncate", i === lastRow ? "text-accent" : dim ? "text-muted" : "text-text")}>{i === lastRow ? `> ${r.label}` : r.label}</span>
                <div className="space-y-0.5">
                  <div className="h-2.5">
                    <EasedBar fraction={Math.max(0, v) / (probabilityStage ? axis : Math.max(maxLogit, 1))} className={dim ? "bg-border" : "bg-accent"} />
                  </div>
                  <div className="relative h-1 bg-surface">
                    <EasedBar fraction={r.cum} className={r.kept ? "bg-accent/50" : "bg-muted/40"} />
                    <div
                      aria-hidden
                      className="glide-snap absolute inset-0 bg-text transition-[clip-path,opacity] duration-700 ease-[cubic-bezier(0.65,0,0.35,1)] motion-reduce:transition-[opacity]"
                      style={{ clipPath: `inset(0 calc(${(1 - p) * 100}% - 1px) 0 calc(${p * 100}% - 1px))`, opacity: p < 1 && !greedy ? 1 : 0 }}
                    />
                  </div>
                </div>
                <span className="text-right tabular-nums text-text">
                  {stageIndex === 0 ? <NumberFlow value={r.logit} format={{ minimumFractionDigits: 1, maximumFractionDigits: 1 }} /> : <NumberFlow value={v} format={{ style: "percent", minimumFractionDigits: 1, maximumFractionDigits: 1 }} />}
                </span>
              </div>
            );
          })}
        </div>
        <p className="leading-relaxed text-muted">
          {probabilityStage ? `Full bar width is ${axis * 100}% (logit bars: the largest logit).` : "Bar length is the logit against the largest logit."} The thin bar under each token is the running total of what top-k left; the white line is top-p, and tokens past it are cut. Grey bars are cut.
        </p>

        <div className="space-y-2">
          <div className="flex flex-wrap items-center gap-2">
            <button type="button" onClick={() => sample(1)} className="pressable rounded border border-border px-2.5 py-1 text-accent">
              sample
            </button>
            <button type="button" onClick={() => sample(20)} className="pressable rounded border border-border px-2.5 py-1 text-text">
              sample 20
            </button>
            <button
              type="button"
              onClick={() => {
                setSeed(nextSeed(seed));
                setStep(0);
              }}
              className="pressable rounded border border-border px-2.5 py-1 text-text"
            >
              new seed
            </button>
            <button type="button" onClick={() => setHistory([])} className="pressable rounded border border-border px-2.5 py-1 text-text">
              clear draws
            </button>
            <button type="button" onClick={reset} className="pressable rounded border border-border px-2.5 py-1 text-text">
              reset
            </button>
            <span className="tabular-nums text-muted">{`seed ${seed}, draw ${step}`}</span>
          </div>
          <ul aria-label="recent draws" className="flex min-h-7 flex-wrap gap-1">
            {history.length === 0 && <li className="text-muted">nothing drawn yet</li>}
            {history.map((h, i) => (
              <li key={i} className={clsx("rounded border px-1.5 py-0.5", i === history.length - 1 ? "border-accent text-accent" : "border-border text-text")}>
                {h.label}
              </li>
            ))}
          </ul>
        </div>
      </div>

      <div className="space-y-5">
        <table className="w-full tabular-nums">
          <tbody>
            {readouts.map((r) => (
              <tr key={r.label} className="border-t border-border first:border-t-0">
                <td className="py-1 text-muted">{r.label}</td>
                <td className="py-1 text-right text-text">{r.value}</td>
              </tr>
            ))}
          </tbody>
        </table>

        <div className="space-y-2 rounded border border-border p-3">
          <div className="text-text">predict first</div>
          <p className="leading-relaxed text-muted">Set top-p to 0.90 with top-k off, then raise the temperature from 0.7 to 1.5. Do the candidates left grow, shrink or stay the same?</p>
          <div className="flex flex-wrap gap-2">
            {["grow", "shrink", "same"].map((g) => (
              <button key={g} type="button" aria-pressed={guess === g} onClick={() => setGuess(g)} className={clsx("pressable rounded border px-2.5 py-1", guess === g ? "border-accent text-accent" : "border-border text-text")}>
                {g}
              </button>
            ))}
          </div>
          {guess && (
            <div className="space-y-2 leading-relaxed">
              <p className={guess === truth ? "text-accent" : "text-warn"}>{`${guess === truth ? "Right" : "Not quite"}: on this prompt they ${truth === "same" ? "stay at" : truth === "grow" ? "grow from" : "shrink from"} ${fmt(before)}${truth === "same" ? "" : ` to ${fmt(after)}`}. Flatter odds spread the mass over more tokens, so the 90% line sits further down the list.`}</p>
              <button
                type="button"
                onClick={() => {
                  glide({ temperature: 0.7, k: KOFF, p: 0.9 });
                  setStage("final");
                }}
                className="pressable rounded border border-border px-2.5 py-1 text-text"
              >
                set T 0.70, top-p 0.90
              </button>
            </div>
          )}
        </div>

        <div className="space-y-1.5 leading-relaxed text-muted">
          <p>Try: drag T to 0 and sample, then to 2 and sample 20. Pull top-k down to 3 and watch the bars after it. Switch the stage to see what each knob does to the same row.</p>
          <p>{`Logits are the BF16 output of TinyLlama on the CPU (${dump.commit}, ${dump.date.slice(0, 10)}): the ${dump.top} largest and the rest in bins of ${dump.bin}. Draws use the engine's Gumbel-max rule.`}</p>
        </div>
      </div>
    </div>
  );
}
