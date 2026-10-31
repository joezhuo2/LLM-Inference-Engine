"use client";

import { motion } from "motion/react";
import { useEffect, useState } from "react";
import { usePrefersReducedMotion } from "@/lib/reduced-motion";
import { Frame, Label } from "../frame";
import { useLoop } from "../loop";

const logits = [1.2, 2.6, 0.4, 3.1, 1.8, 0.9, 2.2, 0.1, 1.5, 2.9, 0.7, 1.1];
const top = Math.max(...logits);
const bx = 40;
const bw = 36;
const base = 190;
const scale = 40;

function Bars({ values, highlight, className = "fill-baseline" }: { values: number[]; highlight?: number; className?: string }) {
  return (
    <g>
      {values.map((v, i) => (
        <rect key={i} x={bx + i * bw} y={base - v * scale} width={bw - 8} height={v * scale} rx={2} className={i === highlight ? "fill-accent" : className} />
      ))}
      <line x1={bx - 4} y1={base} x2={bx + values.length * bw} y2={base} className="stroke-border" />
    </g>
  );
}

export function ArgmaxFigure({ us }: { us: number }) {
  const { on } = useLoop(6);
  const best = logits.indexOf(top);
  const levels = [6, 3, 2, 1];
  return (
    <Frame label={`Greedy decoding: 1,024 threads each keep the best logit they saw, then a tree halves them to one id in ${us} µs`}>
      <Label x={20} y={24}>logits of one row (12 of 32,000 shown)</Label>
      <Bars values={logits} />
      <motion.g {...on(0.3)}>
        <Bars values={logits} highlight={best} />
      </motion.g>
      <Label x={20} y={226}>1,024 threads stride the row, each keeping its best (value, index)</Label>
      {levels.map((n, li) => (
        <motion.g key={n} {...on(0.35 + li * 0.1)}>
          {Array.from({ length: n }, (_, j) => (
            <circle key={j} cx={60 + ((j + 0.5) * 440) / n} cy={252 + li * 26} r={li === levels.length - 1 ? 7 : 5} className={li === levels.length - 1 ? "fill-accent" : "fill-baseline"} />
          ))}
        </motion.g>
      ))}
      <motion.g {...on(0.75)}>
        <Label x={20} y={346} className="fill-text">{`token ${best}: the largest logit, lowest index on ties`}</Label>
      </motion.g>
      <Label x={20} y={372}>{`one block per row, ${us} µs per call at a 32,000-token vocabulary`}</Label>
    </Frame>
  );
}

function mulberry(seed: number) {
  let a = seed;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const gumbel = (u: number) => -Math.log(-Math.log(u));
const temperature = 1;
const softmax = (() => {
  const e = logits.map((l) => Math.exp((l - top) / temperature));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
})();

function stillDraws() {
  const rand = mulberry(7);
  const counts = logits.map(() => 0);
  for (let n = 0; n < 2000; n++) {
    const s = logits.map((l) => l / temperature + gumbel(Math.min(Math.max(rand(), 1e-6), 1 - 1e-6)));
    counts[s.indexOf(Math.max(...s))] += 1;
  }
  return { noise: logits.map(() => 0), pick: -1, counts };
}

const stillPicks = stillDraws();

export function GumbelFigure() {
  const reduced = usePrefersReducedMotion();
  const [live, setLive] = useState<{ noise: number[]; pick: number; counts: number[] }>(() => ({ noise: logits.map(() => 0), pick: logits.indexOf(top), counts: logits.map(() => 0) }));
  useEffect(() => {
    if (reduced) return;
    const rand = mulberry(7);
    const id = setInterval(() => {
      setLive((d) => {
        const noise = logits.map(() => gumbel(Math.min(Math.max(rand(), 1e-6), 1 - 1e-6)));
        const s = logits.map((l, i) => l / temperature + noise[i]);
        const pick = s.indexOf(Math.max(...s));
        const counts = d.counts.map((c, i) => c + (i === pick ? 1 : 0));
        return { noise, pick, counts };
      });
    }, 700);
    return () => clearInterval(id);
  }, [reduced]);
  const draws = reduced ? stillPicks : live;
  const total = draws.counts.reduce((a, b) => a + b, 0);
  const hy = 330;
  const hh = 70;
  return (
    <Frame label="Gumbel-max: add independent Gumbel noise to every scaled logit and take the argmax; the picks follow the softmax">
      <Label x={20} y={24}>logit / T + Gumbel noise, then argmax</Label>
      {logits.map((l, i) => {
        const n = draws.noise[i];
        const y = base - l * scale;
        return (
          <g key={i}>
            <rect x={bx + i * bw} y={y} width={bw - 8} height={l * scale} rx={2} className={i === draws.pick ? "fill-accent" : "fill-baseline"} />
            <motion.rect
              x={bx + i * bw}
              width={bw - 8}
              rx={2}
              className={n >= 0 ? "fill-accent/30" : "fill-warn/30"}
              initial={false}
              animate={{ y: n >= 0 ? y - n * scale * 0.5 : y, height: Math.abs(n) * scale * 0.5 }}
              transition={{ duration: 0.45, ease: [0.65, 0, 0.35, 1] }}
            />
          </g>
        );
      })}
      <line x1={bx - 4} y1={base} x2={bx + logits.length * bw} y2={base} className="stroke-border" />
      <Label x={20} y={226}>{draws.pick >= 0 ? `this draw: token ${draws.pick}` : "2,000 draws, simulated in your browser"}</Label>
      <Label x={20} y={252}>{`picks so far (${total}) against softmax (outline)`}</Label>
      {logits.map((_, i) => {
        const share = total ? draws.counts[i] / total : 0;
        return (
          <g key={i}>
            <rect x={bx + i * bw} y={hy - share * hh * 2} width={bw - 8} height={share * hh * 2} rx={1} className="fill-accent/70" />
            <rect x={bx + i * bw} y={hy - softmax[i] * hh * 2} width={bw - 8} height={softmax[i] * hh * 2} rx={1} className="fill-none stroke-text" strokeWidth={0.8} />
          </g>
        );
      })}
      <line x1={bx - 4} y1={hy} x2={bx + logits.length * bw} y2={hy} className="stroke-border" />
      <Label x={20} y={372}>no exp, no sum, no prefix scan: one more argmax</Label>
    </Frame>
  );
}

export function PhiloxFigure() {
  const { on } = useLoop(6);
  const steps = 4;
  const tokens = 10;
  const cw = 46;
  const ch = 30;
  return (
    <Frame label="Each token's noise comes from a counter-based generator keyed by the request's seed and the offset step × vocab + token, so a draw never depends on the batch">
      <Label x={20} y={24}>noise for token i at decode step s: Philox(seed, offset = s · vocab + i)</Label>
      {Array.from({ length: steps }, (_, s) => (
        <g key={s}>
          <Label x={20} y={70 + s * (ch + 8)}>{`step ${s}`}</Label>
          {Array.from({ length: tokens }, (_, i) => {
            const t = 0.04 + (s * tokens + i) * 0.012;
            return (
              <g key={i}>
                <rect x={80 + i * cw} y={52 + s * (ch + 8)} width={cw - 6} height={ch} rx={3} className="fill-surface stroke-border" />
                <motion.rect x={80 + i * cw} y={52 + s * (ch + 8)} width={cw - 6} height={ch} rx={3} className="fill-accent/40" {...on(t)} />
              </g>
            );
          })}
        </g>
      ))}
      <motion.g {...on(0.55)}>
        <Label x={20} y={230} className="fill-text">
          no state carried between calls
        </Label>
        <Label x={20} y={250}>the same seed and step give the same noise, so a request repeats exactly,</Label>
        <Label x={20} y={268}>and a row draws the same token alone or in a batch of 16</Label>
      </motion.g>
      <motion.g {...on(0.72)}>
        <Label x={20} y={306} className="fill-text">
          each request carries its own seed, step and temperature
        </Label>
        <Label x={20} y={326}>so one kernel launch serves greedy and sampled rows side by side</Label>
      </motion.g>
    </Frame>
  );
}

const keys = Array.from({ length: 40 }, (_, i) => {
  const r = mulberry(11 + i)();
  return Number((Math.sin(i * 1.7) * 1.4 + r * 2.2).toFixed(2));
});

function searchSteps(target: (cut: number) => boolean, lo0: number, hi0: number, n: number) {
  const out: number[] = [];
  let lo = lo0;
  let hi = hi0;
  for (let i = 0; i < n; i++) {
    const mid = (lo + hi) / 2;
    out.push(mid);
    if (target(mid)) lo = mid;
    else hi = mid;
  }
  return out;
}

export function TopKFigure({ k }: { k: number }) {
  const { on } = useLoop(8);
  const min = Math.min(...keys) - 0.2;
  const max = Math.max(...keys) + 0.2;
  const y = (v: number) => 190 - ((v - min) / (max - min)) * 150;
  const cuts = searchSteps((c) => keys.filter((v) => v >= c).length >= k, min, max, 7);
  const span = 0.8 / cuts.length;
  const final = cuts[cuts.length - 1];
  return (
    <Frame label={`Top-k without sorting: binary search a threshold, counting the logits at or above it, until exactly ${k} remain`}>
      <Label x={20} y={24}>{`top-k ${k}: find the cut, never sort the row`}</Label>
      {keys.map((v, i) => (
        <circle key={i} cx={40 + i * 12.5} cy={y(v)} r={3.5} className={v >= final ? "fill-accent" : "fill-baseline"} />
      ))}
      {cuts.map((c, j) => {
        const count = keys.filter((v) => v >= c).length;
        const t0 = 0.04 + j * span;
        const last = j === cuts.length - 1;
        return (
          <motion.g key={j} {...on(t0, last ? 0.94 : t0 + span, last)}>
            <line x1={34} x2={540} y1={y(c)} y2={y(c)} className="stroke-accent" strokeDasharray="4 3" />
            <Label x={20} y={232} className="fill-text">{`pass ${j + 1}: ${count} logits at or above the cut → ${count >= k ? "move it up" : "move it down"}`}</Label>
          </motion.g>
        );
      })}
      <Label x={20} y={270}>each pass: every thread counts its share of the row, one block sum</Label>
      <Label x={20} y={290}>the cut is a 32-bit key, so 32 passes find it exactly</Label>
      <Label x={20} y={310}>every logit tied with the k-th largest stays in</Label>
      <Label x={20} y={350}>then Gumbel noise and the argmax, only over the kept tokens</Label>
    </Frame>
  );
}

export function TopPFigure({ p }: { p: number }) {
  const { on } = useLoop(8);
  const sorted = [...keys].sort((a, b) => b - a);
  const kmax = sorted[0];
  const mass = (cut: number) => keys.filter((v) => v >= cut).reduce((a, v) => a + Math.exp(v - kmax), 0);
  const all = mass(-Infinity);
  const min = Math.min(...keys) - 0.2;
  const max = Math.max(...keys) + 0.2;
  const y = (v: number) => 190 - ((v - min) / (max - min)) * 150;
  const cuts = searchSteps((c) => mass(c) >= p * all, min, max, 7);
  const span = 0.8 / cuts.length;
  const final = cuts[cuts.length - 1];
  return (
    <Frame label={`Top-p: binary search the cut whose kept tokens hold at least ${p} of the probability mass`}>
      <Label x={20} y={24}>{`top-p ${p}: the smallest set of top tokens holding ${p * 100}% of the mass`}</Label>
      {keys.map((v, i) => (
        <circle key={i} cx={40 + i * 12.5} cy={y(v)} r={3.5} className={v >= final ? "fill-accent" : "fill-baseline"} />
      ))}
      {cuts.map((c, j) => {
        const share = mass(c) / all;
        const t0 = 0.04 + j * span;
        const last = j === cuts.length - 1;
        return (
          <motion.g key={j} {...on(t0, last ? 0.94 : t0 + span, last)}>
            <line x1={34} x2={540} y1={y(c)} y2={y(c)} className="stroke-accent" strokeDasharray="4 3" />
            <rect x={20} y={222} width={(share * 500) | 0} height={12} rx={2} className="fill-accent/60" />
            <Label x={20} y={254} className="fill-text">{`pass ${j + 1}: kept mass ${(share * 100).toFixed(0)}%`}</Label>
          </motion.g>
        );
      })}
      <rect x={20} y={222} width={500} height={12} rx={2} className="fill-none stroke-border" />
      <line x1={20 + p * 500} x2={20 + p * 500} y1={216} y2={240} className="stroke-warn" strokeWidth={2} />
      <Label x={20} y={290}>mass(cut) = Σ exp(s − max) over the logits at or above the cut</Label>
      <Label x={20} y={310}>each pass is an exp and a block sum over the row; the cut starts at top-k&apos;s</Label>
      <Label x={20} y={350}>a sort would give the same set, after ordering all 32,000</Label>
    </Frame>
  );
}

export type Mode = { label: string; us: number; accent?: boolean; warn?: boolean };

export function CostFigure({ modes }: { modes: Mode[] }) {
  const { grow } = useLoop(6);
  const max = Math.max(...modes.map((m) => m.us));
  return (
    <Frame label="Measured time per call for each sampling mode against sorting the row">
      <Label x={20} y={24}>µs per call, one row, vocab 32,000, temperature 0.7</Label>
      {modes.map((m, i) => (
        <g key={m.label}>
          <Label x={20} y={70 + i * 44} className={m.accent ? "fill-accent" : m.warn ? "fill-warn" : "fill-muted"}>
            {m.label}
          </Label>
          <rect x={20} y={78 + i * 44} width={440} height={12} rx={2} className="fill-border" />
          <motion.rect x={20} y={78 + i * 44} width={(440 * m.us) / max} height={12} rx={2} className={m.accent ? "fill-accent" : m.warn ? "fill-warn" : "fill-baseline"} {...grow(0.04 + i * 0.06, 0.3 + i * 0.06)} />
          <Label x={470} y={89 + i * 44}>{`${m.us.toFixed(1)}`}</Label>
        </g>
      ))}
    </Frame>
  );
}

export function Fp64Figure({ busy, share, now, float }: { busy: number; share: number; now: number; float: [number, number] }) {
  const { grow, on } = useLoop(7);
  return (
    <Frame label={`The Gumbel noise is computed in double precision; the FP64 pipe is busy ${busy}% of the cycles though FP64 is ${share}% of the instructions`}>
      <Label x={20} y={24} className="fill-text">
        gumbel(u) = −log(−log(u)), in double precision
      </Label>
      <Label x={20} y={64}>FP64 share of instructions</Label>
      <rect x={220} y={53} width={300} height={14} rx={3} className="fill-border" />
      <motion.rect x={220} y={53} width={(300 * share) / 100} height={14} rx={3} className="fill-baseline" {...grow(0.05, 0.25)} />
      <Label x={226 + (300 * share) / 100} y={64} className="fill-text">{`${share}%`}</Label>
      <Label x={20} y={98}>FP64 pipe busy</Label>
      <rect x={220} y={87} width={300} height={14} rx={3} className="fill-border" />
      <motion.rect x={220} y={87} width={(300 * busy) / 100} height={14} rx={3} className="fill-warn" {...grow(0.1, 0.4)} />
      <Label x={226} y={98} className="fill-bg">{`${busy}%`}</Label>
      <Label x={20} y={136}>a consumer GPU runs FP64 at a small fraction of FP32,</Label>
      <Label x={20} y={154}>so 13% of the instructions take most of the time</Label>
      <motion.g {...on(0.45)}>
        <Label x={20} y={200} className="fill-text">
          temperature only: noise for all 32,000 tokens
        </Label>
        <Label x={20} y={220}>{`${now.toFixed(0)} µs as written`}</Label>
        <Label x={20} y={240} className="fill-accent">{`${float[0].toFixed(0)} to ${float[1].toFixed(0)} µs with float and __logf (a throwaway copy)`}</Label>
      </motion.g>
      <motion.g {...on(0.65)}>
        <Label x={20} y={284} className="fill-text">
          with top-k 50: noise for 50 tokens only
        </Label>
        <Label x={20} y={304}>which is why adding top-k made the call faster</Label>
      </motion.g>
      <Label x={20} y={350}>switching to float changes the tokens a fixed seed draws,</Label>
      <Label x={20} y={368}>so it is a decision for the sampler&apos;s author, not done here</Label>
    </Frame>
  );
}
