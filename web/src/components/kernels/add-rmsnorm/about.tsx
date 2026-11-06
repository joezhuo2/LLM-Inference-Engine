"use client";

import { motion } from "motion/react";
import { modelShape } from "@/lib/data";
import { Frame, Label } from "../frame";
import { useLoop } from "../loop";
import { StickyStory, type StoryStep } from "../sticky-story";

const shape = modelShape();
const hidden = shape.hidden.toLocaleString("en-US");
const blocks = 2 * shape.layers;

export const addRmsnormAbout: StoryStep = {
  id: "about",
  title: "what the residual add and RMSNorm are",
  body: (
    <>
      <p>
        Each token goes through the model as one vector of {hidden} numbers, its hidden state. Attention reads it and
        works out a change, and that change is added back on; then the MLP does the same. The running sum is called the
        residual stream, and the add that writes into it is the residual add.
      </p>
      <p>
        RMSNorm stands for root mean square normalization. The root mean square of a vector is the square root of the
        average of its squares: one number for how big its values are. Divide the vector by it, with a small ε added
        under the root so a row of zeros never divides by zero, and what is left has a root mean square of 1; a learned
        weight per channel then sets the size of each channel. It is a simpler LayerNorm that skips subtracting the
        mean.
      </p>
      <p>
        The stream itself is never normalized, and every add can make it bigger. So each block reads a normalized copy instead,
        values of a steady size whatever the stream has grown to. That is why the two come as a pair: the sum one block
        writes is the very row the next block&apos;s norm reads. TinyLlama has a norm before each of its {blocks} attention
        and MLP blocks, and an add after each.
      </p>
    </>
  ),
};

export const addRmsnormAboutCaption = `illustrative: 6 channels with made-up values; TinyLlama's stream has ${hidden} per token`;

const S0 = [0.8, 1.2, 0.5, 1.0, 0.7, 1.1];
const A = [0.3, 0.2, 0.5, 0.25, 0.4, 0.2];
const M = [0.25, 0.4, 0.2, 0.35, 0.3, 0.3];
const W1 = [1.0, 0.9, 1.1, 1.0, 0.95, 1.05];
const W2 = [1.05, 1.0, 0.9, 1.1, 1.0, 0.95];
const rms = (v: number[]) => Math.sqrt(v.reduce((a, x) => a + x * x, 0) / v.length + 1e-5);
const normed = (v: number[], w: number[]) => v.map((x, i) => (w[i] * x) / rms(v));

const bar = 9;
const pitch = 12;
const kS = 22;
const kN = 34;
const streamY = 352;
const normY = 214;
const stations = [42, 262, 452];
const norms = [140, 330];
const blockX = [128, 318];
const plus = [216, 406];

function Bars({ x, base, values, k, className, from }: { x: number; base: number; values: number[]; k: number; className: string; from?: number[] }) {
  return (
    <>
      {values.map((v, i) => {
        const under = from ? from[i] * k : 0;
        return <rect key={i} x={x + i * pitch} y={base - under - v * k} width={bar} height={v * k} rx={1.5} className={className} />;
      })}
    </>
  );
}

function Phase({ scale, start, end, still }: { scale: number; start: number; end: number; still: boolean }) {
  const { on } = useLoop(16);
  const span = end - start;
  const at = (f: number) => start + f * span;
  const s0 = S0.map((v) => v * scale);
  const s1 = s0.map((v, i) => v + A[i]);
  const s2 = s1.map((v, i) => v + M[i]);
  const n1 = normed(s0, W1);
  const n2 = normed(s1, W2);
  return (
    <motion.g {...on(start, end, still)}>
      <Bars x={stations[0]} base={streamY} values={s0} k={kS} className="fill-baseline" />
      <motion.g {...on(at(0.12), end, still)}>
        <Bars x={norms[0]} base={normY} values={n1} k={kN} className="fill-accent" />
      </motion.g>
      <motion.g {...on(at(0.3), end, still)}>
        <Bars x={stations[1]} base={streamY} values={s0} k={kS} className="fill-baseline" />
        <Bars x={stations[1]} base={streamY} values={A} k={kS} from={s0} className="fill-surface stroke-text" />
      </motion.g>
      <motion.g {...on(at(0.46), end, still)}>
        <Bars x={norms[1]} base={normY} values={n2} k={kN} className="fill-accent" />
      </motion.g>
      <motion.g {...on(at(0.64), end, still)}>
        <Bars x={stations[2]} base={streamY} values={s1} k={kS} className="fill-baseline" />
        <Bars x={stations[2]} base={streamY} values={M} k={kS} from={s1} className="fill-surface stroke-text" />
      </motion.g>
      <Label x={20} y={48} className="fill-text">
        {scale === 1 ? "the stream arrives at its usual size" : `the same stream, ${scale} times larger`}
      </Label>
      <Label x={stations[0]} y={streamY + 20}>{`RMS ${rms(s0).toFixed(2)}`}</Label>
      <motion.g {...on(at(0.3), end, still)}>
        <Label x={stations[1]} y={streamY + 20}>{`RMS ${rms(s1).toFixed(2)}`}</Label>
      </motion.g>
      <motion.g {...on(at(0.64), end, still)}>
        <Label x={stations[2]} y={streamY + 20}>{`RMS ${rms(s2).toFixed(2)}`}</Label>
      </motion.g>
    </motion.g>
  );
}

export function AddRmsnormAboutFigure() {
  return (
    <Frame label="The residual stream runs left to right. Before attention and before the MLP, RMSNorm makes a copy of the stream with a root mean square of 1, times a learned weight, and the block reads that copy. The block's output is added back onto the stream, which grows. When the stream arrives larger, the normalized copies stay the same size">
      <Label x={20} y={24}>{"stream = stream + block(RMSNorm(stream)), before attention and before the MLP"}</Label>
      {[0, 1].map((b) => (
        <g key={b}>
          <rect x={blockX[b]} y={74} width={84} height={38} rx={4} className="fill-surface stroke-border" />
          <Label x={blockX[b] + 42} y={97} anchor="middle" className="fill-text">
            {b === 0 ? "attention" : "MLP"}
          </Label>
          <line x1={norms[b] + 30} y1={normY - 64} x2={norms[b] + 30} y2={116} className="stroke-border" markerEnd="url(#ar-about-head)" />
          <line x1={blockX[b] + 84} y1={93} x2={plus[b]} y2={93} className="stroke-border" />
          <line x1={plus[b]} y1={93} x2={plus[b]} y2={streamY - 8} className="stroke-border" markerEnd="url(#ar-about-head)" />
          <circle cx={plus[b]} cy={streamY} r={7} className="fill-surface stroke-text" />
          <Label x={plus[b]} y={streamY + 4} anchor="middle" className="fill-text">
            +
          </Label>
          <line x1={stations[b] + 2 * pitch} y1={streamY - 120} x2={norms[b] + 30} y2={normY + 12} className="stroke-border" strokeDasharray="3 3" />
          <Label x={norms[b] + 30} y={normY + 26} anchor="middle" className="fill-accent">
            RMSNorm
          </Label>
          <Label x={norms[b] + 30} y={normY + 40} anchor="middle">
            RMS 1, then × w
          </Label>
        </g>
      ))}
      <defs>
        <marker id="ar-about-head" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">
          <path d="M0 0 L8 4 L0 8 z" className="fill-muted" />
        </marker>
      </defs>
      <line x1={20} y1={streamY} x2={540} y2={streamY} className="stroke-text" strokeWidth={1.5} markerEnd="url(#ar-about-head)" />
      <Label x={20} y={streamY + 40}>residual stream: filled is what was there, outlined is what the block added</Label>
      <Phase scale={1} start={0.02} end={0.48} still={false} />
      <Phase scale={2.5} start={0.5} end={0.97} still />
    </Frame>
  );
}

export function AddRmsnormAboutStory() {
  return <StickyStory kicker="the idea" steps={[addRmsnormAbout]} caption={() => addRmsnormAboutCaption} figure={() => <AddRmsnormAboutFigure />} />;
}
