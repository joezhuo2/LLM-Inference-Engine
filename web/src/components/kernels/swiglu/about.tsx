"use client";

import { motion } from "motion/react";
import { Frame, Label } from "../frame";
import { useLoop } from "../loop";
import { StickyStory, type StoryStep } from "../sticky-story";
import { silu } from "./figures";

export const swigluAbout: StoryStep = {
  id: "about",
  title: "what SwiGLU is",
  body: (
    <>
      <p>
        SwiGLU stands for Swish-gated linear unit. A gated linear unit, or GLU, makes two linear projections of the same
        input and multiplies them value by value: one of them, the gate, decides how much of the other gets through. The
        first GLU squashed its gate with a sigmoid, which runs from zero to one. SwiGLU, one of the variants Noam Shazeer
        compared in 2020, puts the gate through Swish instead, which with its parameter at one is the function PyTorch calls
        SiLU: silu(g) = g × sigmoid(g).
      </p>
      <p>
        In TinyLlama it is the middle of every layer&apos;s MLP, the block after attention. The normalized hidden state
        is projected twice, by a gate matrix and an up matrix, into 5,632 values each. SwiGLU turns each gate
        and up pair into one value, and the down matrix takes the results back to the hidden size, and they are added onto the
        hidden state.
      </p>
      <p>
        sigmoid(g) is the share of the gate that silu keeps: about a half at zero, almost none for a strongly negative gate, nearly all of it
        for a strongly positive one. So a negative gate shuts its up value off and a large one lets it through, scaled by about g. The curve
        bends smoothly, with a small dip below zero, where a ReLU would cut off at zero with a corner.
      </p>
      <p>
        The gate costs a third matrix. The width is two thirds of the usual four times the hidden size, rounded up to a multiple of a GPU-friendly block size, so the three
        matrices hold about as many weights as a plain MLP&apos;s two that widen to four times the hidden size.
      </p>
    </>
  ),
};

export const FF = 5632;
export const HIDDEN = 2048;
export const PER_CELL = 512;
export const sigmoid = (g: number) => 1 / (1 + Math.exp(-g));
export const aboutGates = [-2.6, 1.8, 0.4, 2.2, -1.2, 3.1, -3.4, 0.9, 2.7, -0.5, 1.4];
export const aboutUps = [1.2, 0.7, 1.4, 1, 0.9, 1.1, 1.3, 0.6, 0.8, 1.5, 1];
export const aboutPicked = 3;

const ffCells = FF / PER_CELL;
const hiddenCells = HIDDEN / PER_CELL;
const fmt = (n: number) => n.toLocaleString("en-US");

export const swigluAboutCaption = `illustrative: each cell stands for ${PER_CELL} values, ${ffCells} for ${fmt(FF)} and ${hiddenCells} for ${fmt(HIDDEN)}; the gate and up values are made up, the curves are exact; calculated: ${fmt(FF)} is two thirds of 4 × ${fmt(HIDDEN)} rounded up to a multiple of 256, and a plain MLP would widen to ${fmt(4 * HIDDEN)}`;

const top = 52;
const cellH = 13;
const pitch = 15;
const xCol = 30;
const gateCol = 150;
const upCol = 186;
const outCol = 316;
const resCol = 470;
const colW = 26;
const outMax = 66;
const mid = top + (ffCells * pitch) / 2 - 1;
const maxSilu = Math.max(...aboutGates.map(silu));
const maxOut = Math.max(...aboutGates.map((g, j) => silu(g) * aboutUps[j]));

const px = (g: number) => 40 + ((g + 4) / 8) * 230;
const py = (s: number) => 376 - s * 26;
const bx = 330;
const unit = 40;
const samples = Array.from({ length: 49 }, (_, k) => 3.6 * Math.sin((2 * Math.PI * k) / 48));

export function SwigluAboutFigure() {
  const { on, grow, reduced } = useLoop(12);
  const curve = (f: (g: number) => number) =>
    Array.from({ length: 81 }, (_, k) => {
      const g = -4 + k / 10;
      return `${k ? "L" : "M"} ${px(g).toFixed(1)} ${py(f(g)).toFixed(1)}`;
    }).join(" ");
  const sweep = { duration: 12, repeat: Infinity, ease: "linear" as const };
  const still = 2;
  return (
    <Frame label="The MLP: the hidden state of 2,048 values is projected into a gate and an up vector of 5,632 values each; each output is silu of the gate times up, and the down projection takes the 5,632 outputs back to 2,048. Below, silu(g) = g × sigmoid(g) against g, and how much of an up value of 1 passes for that gate">
      <Label x={20} y={20}>{"SwiGLU in the MLP: two projections up, one value per pair, one projection down"}</Label>
      <Label x={xCol} y={top - 8}>x</Label>
      <Label x={gateCol} y={top - 8} className="fill-accent">gate</Label>
      <Label x={upCol} y={top - 8}>up</Label>
      <Label x={outCol} y={top - 8} className="fill-accent">{"silu(gate) × up"}</Label>
      <Label x={resCol} y={top - 8}>down</Label>
      {Array.from({ length: hiddenCells }, (_, k) => (
        <rect key={`x${k}`} x={xCol} y={top + ((ffCells - hiddenCells) / 2 + k) * pitch} width={colW} height={cellH} rx={2} className="fill-baseline/70" />
      ))}
      <line x1={xCol + colW + 6} x2={gateCol - 8} y1={mid} y2={mid} className="stroke-text" strokeWidth={1} />
      <path d={`M ${gateCol - 13} ${mid - 4} L ${gateCol - 8} ${mid} L ${gateCol - 13} ${mid + 4}`} className="fill-none stroke-text" strokeWidth={1} />
      <Label x={xCol + colW + 8} y={mid + 16}>{"× W_gate, W_up"}</Label>
      <motion.g {...on(0.04, 0.95)}>
        {aboutGates.map((g, j) => (
          <g key={`gu${j}`}>
            <rect x={gateCol} y={top + j * pitch} width={colW} height={cellH} rx={2} className="fill-accent" opacity={0.12 + (0.88 * Math.max(0, silu(g))) / maxSilu} />
            <rect x={upCol} y={top + j * pitch} width={colW} height={cellH} rx={2} className="fill-baseline" opacity={0.25 + (0.6 * aboutUps[j]) / 1.5} />
          </g>
        ))}
      </motion.g>
      <line x1={upCol + colW + 6} x2={outCol - 8} y1={mid} y2={mid} className="stroke-text" strokeWidth={1} />
      <path d={`M ${outCol - 13} ${mid - 4} L ${outCol - 8} ${mid} L ${outCol - 13} ${mid + 4}`} className="fill-none stroke-text" strokeWidth={1} />
      <Label x={upCol + colW + 8} y={mid + 16}>{"pair j → one value"}</Label>
      {aboutGates.map((g, j) => (
        <g key={`o${j}`}>
          <rect x={outCol} y={top + j * pitch} width={outMax} height={cellH} rx={2} className="fill-surface stroke-border" strokeWidth={0.5} />
          <motion.rect x={outCol} y={top + j * pitch} width={Math.max(1, (outMax * Math.max(0, silu(g) * aboutUps[j])) / maxOut)} height={cellH} rx={2} className="fill-accent" {...grow(0.16 + j * 0.035, 0.24 + j * 0.035)} />
        </g>
      ))}
      <rect x={gateCol - 4} y={top + aboutPicked * pitch - 2} width={outCol + outMax - gateCol + 8} height={cellH + 4} rx={3} className="fill-none stroke-text" strokeWidth={1} strokeDasharray="3 3" />
      <line x1={outCol + outMax + 6} x2={resCol - 8} y1={mid} y2={mid} className="stroke-text" strokeWidth={1} />
      <path d={`M ${resCol - 13} ${mid - 4} L ${resCol - 8} ${mid} L ${resCol - 13} ${mid + 4}`} className="fill-none stroke-text" strokeWidth={1} />
      <Label x={outCol + outMax + 8} y={mid + 16}>{"× W_down"}</Label>
      <motion.g {...on(0.62, 0.95)}>
        {Array.from({ length: hiddenCells }, (_, k) => (
          <rect key={`r${k}`} x={resCol} y={top + ((ffCells - hiddenCells) / 2 + k) * pitch} width={colW} height={cellH} rx={2} className="fill-baseline/70" />
        ))}
      </motion.g>
      <Label x={xCol} y={top + ffCells * pitch + 14}>{fmt(HIDDEN)}</Label>
      <Label x={gateCol} y={top + ffCells * pitch + 14}>{`${fmt(FF)} each`}</Label>
      <Label x={outCol} y={top + ffCells * pitch + 14}>{fmt(FF)}</Label>
      <Label x={resCol} y={top + ffCells * pitch + 14}>{fmt(HIDDEN)}</Label>
      <Label x={20} y={256} className="fill-accent">{"silu(g) = g × sigmoid(g)"}</Label>
      <Label x={20} y={272}>{"dashed: sigmoid(g), the first GLU's gate, 0 to 1"}</Label>
      <line x1={px(-4)} x2={px(4)} y1={py(0)} y2={py(0)} className="stroke-border" />
      <line x1={px(0)} x2={px(0)} y1={py(-0.4)} y2={py(4)} className="stroke-border" />
      <path d={curve(sigmoid)} className="fill-none stroke-muted" strokeWidth={1.25} strokeDasharray="4 3" />
      <path d={curve(silu)} className="fill-none stroke-accent" strokeWidth={1.75} />
      <Label x={px(4)} y={py(0) + 14} anchor="end">g</Label>
      <Label x={px(-4)} y={py(0) + 14}>−4</Label>
      <Label x={px(0) + 4} y={py(0) + 14}>0</Label>
      {reduced ? (
        <>
          <circle cx={px(still)} cy={py(sigmoid(still))} r={3.5} className="fill-muted" />
          <circle cx={px(still)} cy={py(silu(still))} r={4.5} className="fill-accent" />
        </>
      ) : (
        <>
          <motion.circle cx={0} cy={0} r={3.5} className="fill-muted" initial={{ x: px(0), y: py(sigmoid(0)) }} animate={{ x: samples.map(px), y: samples.map((g) => py(sigmoid(g))) }} transition={sweep} />
          <motion.circle cx={0} cy={0} r={4.5} className="fill-accent" initial={{ x: px(0), y: py(silu(0)) }} animate={{ x: samples.map(px), y: samples.map((g) => py(silu(g))) }} transition={sweep} />
        </>
      )}
      <Label x={bx} y={296}>{"an up value of 1"}</Label>
      <rect x={bx} y={302} width={unit} height={12} rx={3} className="fill-baseline" />
      <Label x={bx} y={334} className="fill-accent">{"what passes: silu(g) × up"}</Label>
      <rect x={bx} y={340} width={unit * 4} height={12} rx={3} className="fill-border" />
      {reduced ? (
        <rect x={bx} y={340} width={unit * silu(still)} height={12} rx={3} className="fill-accent" />
      ) : (
        <motion.rect x={bx} y={340} height={12} rx={3} className="fill-accent" initial={{ width: 0 }} animate={{ width: samples.map((g) => unit * Math.max(0.02, silu(g))) }} transition={sweep} />
      )}
      <line x1={bx + unit} x2={bx + unit} y1={336} y2={356} className="stroke-text" strokeWidth={1} strokeDasharray="2 2" />
      <Label x={bx} y={374}>{"negative gate: almost nothing passes"}</Label>
      <Label x={bx} y={390}>{"large gate: up passes, scaled by about g"}</Label>
    </Frame>
  );
}

export function SwigluAboutStory() {
  return <StickyStory kicker="the idea" steps={[swigluAbout]} caption={() => swigluAboutCaption} figure={() => <SwigluAboutFigure />} />;
}
