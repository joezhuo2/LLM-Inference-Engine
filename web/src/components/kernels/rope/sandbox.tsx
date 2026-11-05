"use client";

import NumberFlow from "@/components/number-flow";
import clsx from "clsx";
import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { SmoothRange } from "@/components/smooth-range";
import { glide as tween, type Tween } from "@/lib/glide";
import { usePrefersReducedMotion } from "@/lib/reduced-motion";
import { Toggle } from "../toggle";
import { MAX_POS, PAIRS, angle, invFreq, rotatePair, score, seededVector, wavelength } from "./rotation";

const q = seededVector(7);
const k = seededVector(11);
const TAU = 2 * Math.PI;
const deg = (rad: number) => ((rad % TAU) * 180) / Math.PI;
const SHIFT = 100;
const CURVE_W = 560;
const CURVE_H = 130;
const curveX = (gap: number) => ((gap + MAX_POS) / (2 * MAX_POS)) * CURVE_W;

function scoreCurve() {
  const points = Array.from({ length: MAX_POS / 2 + 1 }, (_, i) => 4 * i - MAX_POS).map((gap) => [gap, score(q, gap, k, 0)] as const);
  const peak = Math.max(...points.map(([, s]) => Math.abs(s)));
  const y = (s: number) => CURVE_H / 2 - (s / peak) * (CURVE_H / 2 - 8);
  return { y, path: points.map(([gap, s], i) => `${i ? "L" : "M"}${curveX(gap).toFixed(1)} ${y(s).toFixed(2)}`).join("") };
}

const button = "pressable rounded border border-border px-2.5 py-1 text-text";
const slider = "w-full accent-[var(--accent)]";

const defaults = { pos: 300, pair: 0, qPos: 300, kPos: 280 };

type Guess = "changes" | "stays the same" | null;

export function RopeSandbox({ pages }: { pages?: { one: HTMLElement | null; two: HTMLElement | null } } = {}) {
  const [pos, setPos] = useState(300);
  const [pair, setPair] = useState(0);
  const [pairF, setPairF] = useState(0);
  const [qPos, setQPos] = useState(300);
  const [kPos, setKPos] = useState(280);
  const [lock, setLock] = useState("locked");
  const [curve] = useState(scoreCurve);
  const [guess, setGuess] = useState<Guess>(null);

  const reduced = usePrefersReducedMotion();
  const run = useRef<Tween | null>(null);
  const stop = () => {
    run.current?.stop();
    run.current = null;
  };
  useEffect(() => stop, []);

  const glide = (to: Partial<typeof defaults>) => {
    stop();
    const from = { pos, pair, qPos, kPos };
    const target = { ...from, ...to };
    if (reduced) {
      setPos(target.pos);
      setPair(target.pair);
      setPairF(target.pair);
      setQPos(target.qPos);
      setKPos(target.kPos);
      return;
    }
    const mix = (key: keyof typeof from, t: number) => from[key] + (target[key] - from[key]) * t;
    const at = (key: keyof typeof from, t: number) => Math.round(mix(key, t));
    run.current = tween(
      0,
      1,
      (t) => {
        setPos(at("pos", t));
        setPair(at("pair", t));
        setPairF(mix("pair", t));
        setQPos(at("qPos", t));
        setKPos(at("kPos", t));
      },
      () => {
        run.current = null;
      },
    );
  };

  const reset = () => {
    if (run.current) return stop();
    setLock("locked");
    setGuess(null);
    glide(defaults);
  };

  const theta = invFreq(pair);
  const turned = angle(pair, pos);
  const lo = q[pair];
  const hi = q[pair + PAIRS];
  const [rlo, rhi] = rotatePair(lo, hi, turned);
  const floor = Math.floor(pairF);
  const ceil = Math.min(floor + 1, PAIRS - 1);
  const frac = pairF - floor;
  const polar = (i: number) => [Math.hypot(q[i], q[i + PAIRS]), Math.atan2(q[i + PAIRS], q[i])] as const;
  const [r0, b0] = polar(floor);
  const [r1, b1raw] = polar(ceil);
  const b1 = b1raw + TAU * Math.round((b0 - b1raw) / TAU);
  const radius = r0 + (r1 - r0) * frac;
  const base = b0 + (b1 - b0) * frac;
  const swing0 = angle(floor, pos);
  const swing1 = angle(ceil, pos);
  const swing = swing0 + (swing1 + TAU * Math.round((swing0 - swing1) / TAU) - swing0) * frac;
  const vx = radius * Math.cos(base);
  const vy = radius * Math.sin(base);
  const scale = 70 / Math.max(...Array.from({ length: PAIRS }, (_, i) => Math.hypot(q[i], q[i + PAIRS])));

  const gap = qPos - kPos;
  const now = score(q, qPos, k, kPos);
  const direction = Math.max(qPos, kPos) + SHIFT <= MAX_POS ? SHIFT : -SHIFT;
  const moved = score(q, qPos + direction, k, kPos + direction);

  const moveQ = (v: number) => {
    stop();
    const d = lock === "locked" ? Math.max(-kPos, Math.min(MAX_POS - kPos, v - qPos)) : v - qPos;
    setQPos(qPos + d);
    if (lock === "locked") setKPos(kPos + d);
  };
  const moveK = (v: number) => {
    stop();
    const d = lock === "locked" ? Math.max(-qPos, Math.min(MAX_POS - qPos, v - kPos)) : v - kPos;
    setKPos(kPos + d);
    if (lock === "locked") setQPos(qPos + d);
  };
  const shift = (d: number) => {
    const s = Math.max(-Math.min(qPos, kPos), Math.min(MAX_POS - Math.max(qPos, kPos), d));
    glide({ qPos: qPos + s, kPos: kPos + s });
  };

  const intro = (
      <div className="flex flex-wrap items-center justify-between gap-3">
        <p className="max-w-2xl text-muted">
          Part 1: drag the position and pick a dimension pair. Watch the pair at the top left turn fast while the pairs at the bottom right barely move. Part 2: move the query and the key and watch the score. The vectors are made up for illustration (seeded random numbers, not from the model); the frequencies, the pairing of dimension i with i + 32 and the rotation are the engine&apos;s.
        </p>
        <button type="button" onClick={reset} className={button}>
          reset
        </button>
      </div>
  );
  const partOne = (
        <div className="space-y-4">
          <h3 className="text-sm text-text">1 · one position, 32 pairs</h3>
          <label className="block space-y-1">
            <span className="flex justify-between text-muted">
              <span>position m</span>
              <NumberFlow value={pos} className="text-text" />
            </span>
            <SmoothRange min={0} max={MAX_POS} value={pos} onValue={(v) => {
                stop();
                setPos(v);
              }} aria-label="position m" className={slider} />
          </label>
          <label className="block space-y-1">
            <span className="flex justify-between text-muted">
              <span>dimension pair i (dims i and i + 32)</span>
              <NumberFlow value={pair} className="text-text" />
            </span>
            <SmoothRange min={0} max={PAIRS - 1} value={pair} onValue={(v) => {
                stop();
                setPair(v);
                setPairF(v);
              }} aria-label="dimension pair i" className={slider} />
          </label>
          <div className="grid items-center gap-4 sm:grid-cols-[11rem_minmax(0,1fr)]">
            <svg viewBox="0 0 200 200" className="mx-auto w-full max-w-[11rem]" role="img" aria-label={`Pair ${pair} of the query, drawn as a 2D vector, rotated by ${deg(turned).toFixed(0)} degrees`}>
              <circle cx={100} cy={100} r={80} className="fill-none stroke-border" />
              <line x1={20} y1={100} x2={180} y2={100} className="stroke-border" />
              <line x1={100} y1={20} x2={100} y2={180} className="stroke-border" />
              <line x1={100} y1={100} x2={100 + vx * scale} y2={100 - vy * scale} className="stroke-muted" strokeDasharray="3 3" />
              <g style={{ transform: `rotate(${-deg(swing)}deg)`, transformOrigin: "100px 100px" }}>
                <line x1={100} y1={100} x2={100 + vx * scale} y2={100 - vy * scale} className="stroke-accent" strokeWidth={2} />
                <circle cx={100 + vx * scale} cy={100 - vy * scale} r={3.5} className="fill-accent" />
              </g>
              <text x={176} y={94} textAnchor="end" className="fill-muted text-[13px]">{`x${pair}`}</text>
              <text x={104} y={30} className="fill-muted text-[13px]">{`x${pair + PAIRS}`}</text>
            </svg>
            <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-1 tabular-nums">
              <dt className="text-muted">θ{"ᵢ"}</dt>
              <dd className="text-right text-text">
                <NumberFlow value={theta} format={{ maximumSignificantDigits: 3 }} />
              </dd>
              <dt className="text-muted">angle m·θ</dt>
              <dd className="text-right text-text">
                <NumberFlow value={turned} format={{ maximumFractionDigits: 2 }} suffix=" rad" />
              </dd>
              <dt className="text-muted">on the dial</dt>
              <dd className="text-right text-text">
                <NumberFlow value={Math.round(deg(turned))} suffix="°" />
              </dd>
              <dt className="text-muted">full turns</dt>
              <dd className="text-right text-text">
                <NumberFlow value={Math.floor(turned / TAU)} />
              </dd>
              <dt className="text-muted">wavelength</dt>
              <dd className="text-right text-text">
                <NumberFlow value={wavelength(pair)} format={{ maximumSignificantDigits: 4 }} suffix=" tokens" />
              </dd>
              <dt className="text-muted">rotated pair</dt>
              <dd className="text-right text-muted">
                {`(${rlo.toFixed(2)}, ${rhi.toFixed(2)})`}
              </dd>
            </dl>
          </div>
          <p className="text-muted">
            {wavelength(pair) > MAX_POS
              ? `Wavelength ${Math.round(wavelength(pair)).toLocaleString("en-US")} tokens is longer than the 2,048-token context: this pair never completes a turn.`
              : `One full turn every ${wavelength(pair).toFixed(1)} tokens: positions ${Math.round(wavelength(pair))} apart look the same to this pair.`}
          </p>
          <div role="group" aria-label="all 32 dimension pairs" className="grid grid-cols-8 gap-1.5">
            {Array.from({ length: PAIRS }, (_, i) => (
              <button
                key={i}
                type="button"
                onClick={() => glide({ pair: i })}
                aria-pressed={pair === i}
                aria-label={`pair ${i}, ${Math.round(deg(angle(i, pos)))} degrees`}
                className={clsx("pressable rounded border p-0.5", pair === i ? "border-accent bg-accent-dim" : "border-border")}
              >
                <svg viewBox="0 0 40 40" className="w-full" aria-hidden>
                  <circle cx={20} cy={20} r={17} className="fill-none stroke-border" />
                  <g style={{ transform: `rotate(${-deg(angle(i, pos))}deg)`, transformOrigin: "20px 20px" }}>
                    <line x1={20} y1={20} x2={35} y2={20} className={pair === i ? "stroke-accent" : "stroke-text"} strokeWidth={2} />
                  </g>
                </svg>
                <span className={clsx("block text-center text-[10px] tabular-nums", pair === i ? "text-accent" : "text-muted")}>{i}</span>
              </button>
            ))}
          </div>
        </div>
  );
  const partTwo = (
        <div className="space-y-4">
          <h3 className="text-sm text-text">2 · the score depends on the gap</h3>
          <div className="space-y-2 rounded border border-border p-3">
            <div className="text-text">Predict first: move the query and the key together by {SHIFT} positions, keeping the gap. Does the score q·k change?</div>
            <div className="flex flex-wrap items-center gap-2">
              {(["changes", "stays the same"] as const).map((g) => (
                <button key={g} type="button" onClick={() => {
                    stop();
                    setGuess(g);
                  }} aria-pressed={guess === g} className={clsx("pressable rounded border px-2.5 py-1", guess === g ? "border-accent text-accent" : "border-border text-text")}>
                  {g}
                </button>
              ))}
            </div>
            {guess && (
              <p className="text-muted">
                {`${Math.abs(moved - now) < 1e-6 ? "It stays the same" : "It changes"}: at positions ${qPos} and ${kPos} the score is ${now.toFixed(3)}, at ${qPos + direction} and ${kPos + direction} it is ${moved.toFixed(3)}. ${(Math.abs(moved - now) < 1e-6) === (guess === "stays the same") ? "You predicted it." : "Not what you predicted."} Now drag the sliders yourself, then unlock the gap to see it change.`}
              </p>
            )}
          </div>
          <Toggle label="gap" choices={["locked", "free"]} value={lock} onChange={(v) => {
            stop();
            setLock(v);
          }} />
          <label className="block space-y-1">
            <span className="flex justify-between text-muted">
              <span>query position</span>
              <NumberFlow value={qPos} className="text-text" />
            </span>
            <SmoothRange min={0} max={MAX_POS} value={qPos} onValue={moveQ} aria-label="query position" className={slider} />
          </label>
          <label className="block space-y-1">
            <span className="flex justify-between text-muted">
              <span>key position</span>
              <NumberFlow value={kPos} className="text-text" />
            </span>
            <SmoothRange min={0} max={MAX_POS} value={kPos} onValue={moveK} aria-label="key position" className={slider} />
          </label>
          <div className="flex flex-wrap gap-2">
            <button type="button" onClick={() => shift(-SHIFT)} className={button}>{`both −${SHIFT}`}</button>
            <button type="button" onClick={() => shift(SHIFT)} className={button}>{`both +${SHIFT}`}</button>
          </div>
          <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-1 tabular-nums">
            <dt className="text-muted">gap m − n</dt>
            <dd className="text-right text-text">
              <NumberFlow value={gap} />
            </dd>
            <dt className="text-muted">score q·k</dt>
            <dd className="text-right text-accent">
              <NumberFlow value={now} format={{ minimumFractionDigits: 3, maximumFractionDigits: 3 }} />
            </dd>
          </dl>
          <svg viewBox={`0 0 ${CURVE_W} ${CURVE_H}`} className="w-full" role="img" aria-label={`The score as a function of the gap from -2048 to 2048; the current gap is ${gap} and the score ${now.toFixed(3)}`}>
            <line x1={0} y1={CURVE_H / 2} x2={CURVE_W} y2={CURVE_H / 2} className="stroke-border" />
            <line x1={curveX(0)} y1={0} x2={curveX(0)} y2={CURVE_H} className="stroke-border" strokeDasharray="3 3" />
            <path d={curve.path} className="fill-none stroke-muted" strokeWidth={1} />
            <line x1={curveX(gap)} y1={0} x2={curveX(gap)} y2={CURVE_H} className="stroke-accent" />
            <circle cx={curveX(gap)} cy={curve.y(now)} r={4} className="fill-accent" />
            <text x={4} y={14} className="fill-muted text-[12px]">{`score for every gap, ${-MAX_POS} to ${MAX_POS} (same vectors)`}</text>
          </svg>
          <p className="text-muted">The dot depends on the gap alone. Drag both positions with the gap locked and it does not move; unlock the gap and move one position and it slides along the curve.</p>
        </div>
  );
  if (pages) {
    const resetButton = (
      <div className="flex justify-end">
        <button type="button" onClick={reset} className={button}>
          reset
        </button>
      </div>
    );
    return (
      <>
        {pages.one &&
          createPortal(
            <div className="space-y-6 tabular-nums text-xs">
              {resetButton}
              {partOne}
            </div>,
            pages.one,
          )}
        {pages.two &&
          createPortal(
            <div className="space-y-6 tabular-nums text-xs">
              {resetButton}
              {partTwo}
            </div>,
            pages.two,
          )}
      </>
    );
  }
  return (
    <div className="space-y-10 tabular-nums text-xs">
      {intro}
      <div className="grid gap-x-10 gap-y-10 lg:grid-cols-2">
        {partOne}
        {partTwo}
      </div>
    </div>
  );
}
