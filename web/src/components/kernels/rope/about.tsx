"use client";

import { motion } from "motion/react";
import { Frame, Label } from "../frame";
import { useLoop } from "../loop";
import { StickyStory, type StoryStep } from "../sticky-story";
import { invFreq, PAIRS } from "./rotation";

export const ABOUT_PAIR = 3;
export const aboutTheta = invFreq(ABOUT_PAIR);

export const ropeAbout: StoryStep = {
  id: "about",
  title: "what RoPE does",
  body: (
    <>
      <p>
        RoPE is short for rotary position embedding. It is how TinyLlama, like the other Llama models, tells attention where
        each token sits in the prompt. It was introduced in 2021 in the RoFormer paper. The original transformer added a fixed
        position vector to each token&apos;s embedding once, before the first layer; RoPE works inside every attention layer
        instead.
      </p>
      <p>
        In attention every token produces a query, what it is looking for, and a key, what it offers. The score between two
        tokens is the dot product of one&apos;s query and the other&apos;s key: multiply them value by value and add up. A
        dot product only sees the numbers, so without help a query would give a key the same score whether that key&apos;s
        token came just before it or a thousand tokens back.
      </p>
      <p>
        Take two values of a vector as a point on a plane. Turning the point around the centre keeps its length and changes
        only its direction. RoPE turns each query and each key by an angle proportional to its token&apos;s position, so a
        query at position 9 and a key at position 6 end up three positions&apos; worth of angle apart, exactly as at
        positions 5 and 2. The score then changes with the distance between the two tokens and stays the same when both move
        along together.
      </p>
      <p>The figure turns one pair of values. A head has {PAIRS} such pairs, each turning at its own rate.</p>
    </>
  ),
};

export const ropeAboutCaption = `illustrative: one pair of values turning at pair ${ABOUT_PAIR}'s real rate of ${aboutTheta.toFixed(2)} radians per position; the query and key start out pointing the same way`;

const slots = 12;
const boxW = 30;
const pitch = 36;
const stripX = 64;
const stripY = 42;
const r = 44;
const big = 62;
const dialY = 178;
const qX = 100;
const kX = 236;
const mX = 420;

const states = [
  { m: 5, n: 2 },
  { m: 9, n: 6 },
  { m: 9, n: 3 },
];
const times = [0, 0.25, 0.33, 0.58, 0.66, 0.92, 1];
const final = states[2];
const deg = (pos: number) => (-pos * aboutTheta * 180) / Math.PI;
const track = (pick: (s: { m: number; n: number }) => number) => {
  const [a, b, c] = states.map(pick);
  return [a, a, b, b, c, c, a];
};

function wedge(radius: number, gap: number) {
  const a = gap * aboutTheta;
  const x = radius * Math.cos(a);
  const y = -radius * Math.sin(a);
  return `M0 0 L${radius} 0 A${radius} ${radius} 0 ${a > Math.PI ? 1 : 0} 0 ${x.toFixed(2)} ${y.toFixed(2)} Z`;
}

function Arm({ length, className }: { length: number; className: string }) {
  return (
    <>
      <circle r={length} className="fill-transparent" />
      <line x1={0} y1={0} x2={length - 5} y2={0} className={className} strokeWidth={2.5} />
      <circle cx={length - 5} cy={0} r={4} className={className.replace("stroke-", "fill-")} />
    </>
  );
}

function Dial({ cx, radius, children }: { cx: number; radius: number; children: React.ReactNode }) {
  return (
    <g transform={`translate(${cx} ${dialY})`}>
      <circle r={radius} className="fill-surface stroke-border" />
      <line x1={-radius} x2={radius} y1={0} y2={0} className="stroke-border" strokeDasharray="2 3" />
      <line x1={0} x2={0} y1={-radius} y2={radius} className="stroke-border" strokeDasharray="2 3" />
      {children}
    </g>
  );
}

export function RopeAboutFigure() {
  const { on, reduced } = useLoop(20);
  const turn = (pick: (s: { m: number; n: number }) => number) =>
    reduced
      ? { initial: false, style: { rotate: deg(pick(final)) } }
      : { initial: false, animate: { rotate: track(pick).map(deg) }, transition: { duration: 20, repeat: Infinity, times, ease: "easeInOut" as const } };
  const slide = (pick: (s: { m: number; n: number }) => number) =>
    reduced
      ? { initial: false, style: { x: pick(final) * pitch } }
      : { initial: false, animate: { x: track(pick).map((p) => p * pitch) }, transition: { duration: 20, repeat: Infinity, times, ease: "easeInOut" as const } };
  const readout = (s: { m: number; n: number }, note: string) => {
    const gap = s.m - s.n;
    return (
      <>
        <Label x={20} y={300} className="fill-text">{`query at ${s.m}, key at ${s.n}: ${gap} positions apart`}</Label>
        <Label x={20} y={318}>{`angle between them ${gap} × ${aboutTheta.toFixed(2)} = ${(gap * aboutTheta).toFixed(2)} radians, score q · k = ${Math.cos(gap * aboutTheta).toFixed(2)}`}</Label>
        <Label x={20} y={336} className="fill-accent">{note}</Label>
      </>
    );
  };
  return (
    <Frame label="A query and a key from two positions of a prompt, each turned by an angle proportional to its position. When both move along together the angle between them, and so their score, stays the same; when the gap grows the score changes">
      <Label x={20} y={24}>{"RoPE, rotary position embedding: turn each query and key by position × θ"}</Label>
      {Array.from({ length: slots }, (_, i) => (
        <g key={i}>
          <rect x={stripX + i * pitch} y={stripY} width={boxW} height={22} rx={3} className="fill-surface stroke-border" />
          <Label x={stripX + i * pitch + boxW / 2} y={stripY + 38} anchor="middle">
            {i}
          </Label>
        </g>
      ))}
      <Label x={20} y={stripY + 15}>tokens</Label>
      <Label x={20} y={stripY + 38}>position</Label>
      <motion.g {...slide((s) => s.n)}>
        <rect x={stripX} y={stripY} width={boxW} height={22} rx={3} className="fill-accent/45" />
      </motion.g>
      <motion.g {...slide((s) => s.m)}>
        <rect x={stripX} y={stripY} width={boxW} height={22} rx={3} className="fill-accent" />
      </motion.g>
      <Dial cx={qX} radius={r}>
        <motion.g {...turn((s) => s.m)}>
          <Arm length={r} className="stroke-accent" />
        </motion.g>
      </Dial>
      <Dial cx={kX} radius={r}>
        <motion.g {...turn((s) => s.n)}>
          <Arm length={r} className="stroke-accent/45" />
        </motion.g>
      </Dial>
      <Dial cx={mX} radius={big}>
        <motion.g {...turn((s) => s.n)}>
          <circle r={big} className="fill-transparent" />
          <motion.path d={wedge(big - 8, 3)} className="fill-accent/15" {...on(0.02, 0.57, false)} />
          <motion.path d={wedge(big - 8, 6)} className="fill-accent/15" {...on(0.67, 0.91)} />
        </motion.g>
        <motion.g {...turn((s) => s.n)}>
          <Arm length={big} className="stroke-accent/45" />
        </motion.g>
        <motion.g {...turn((s) => s.m)}>
          <Arm length={big} className="stroke-accent" />
        </motion.g>
      </Dial>
      <Label x={qX} y={dialY + r + 20} anchor="middle" className="fill-text">query at position m</Label>
      <Label x={qX} y={dialY + r + 34} anchor="middle">turned by m × θ</Label>
      <Label x={kX} y={dialY + r + 20} anchor="middle" className="fill-text">key at position n</Label>
      <Label x={kX} y={dialY + r + 34} anchor="middle">turned by n × θ</Label>
      <Label x={mX} y={dialY + big + 20} anchor="middle" className="fill-text">both together</Label>
      <Label x={mX} y={dialY + big + 34} anchor="middle">angle between: (m − n) × θ</Label>
      <motion.g {...on(0.02, 0.3, false)}>{readout(states[0], "the starting pair")}</motion.g>
      <motion.g {...on(0.34, 0.62, false)}>{readout(states[1], "both moved 4 places along: same gap, same angle, same score")}</motion.g>
      <motion.g {...on(0.66, 0.97)}>{readout(states[2], "the key moved 3 places back: a wider gap, a new score")}</motion.g>
      <Label x={20} y={372}>{`one pair of values at pair ${ABOUT_PAIR}'s rate, θ = ${aboutTheta.toFixed(2)} radians per position; both vectors have length 1`}</Label>
      <Label x={20} y={388}>so the score is the cosine of the angle between them</Label>
    </Frame>
  );
}

export function RopeAboutStory() {
  return <StickyStory kicker="the idea" steps={[ropeAbout]} caption={() => ropeAboutCaption} figure={() => <RopeAboutFigure />} />;
}
