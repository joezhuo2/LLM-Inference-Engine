"use client";

import { motion } from "motion/react";
import { Frame, Label } from "../frame";
import { useLoop } from "../loop";
import { StickyStory, type StoryStep } from "../sticky-story";

export const gemmAbout: StoryStep = {
  id: "about",
  title: "what GEMM and GEMV mean",
  body: (
    <>
      <p>
        GEMM is short for general matrix multiply, and GEMV for general matrix-vector multiply. The names come from BLAS, the Basic Linear Algebra Subprograms, a Fortran library first published in 1979 whose routine names every GPU math library still uses: GE for a general matrix, with no special shape such as symmetric or triangular, MM for a matrix times a matrix, MV for a matrix times a vector.
      </p>
      <p>
        Both compute the same thing. Every output value is a dot product: take one row of the left matrix and one column of the right one, multiply them pair by pair and add up the products. An output of M rows and N columns with K pairs in each dot product costs M × N × K multiply-adds.
      </p>
      <p>
        In a transformer the left side is x, the activations, one row for every token being processed, and the right side is a weight matrix W learned in training. Decoding one token makes x a single row, and the product is a GEMV; sending several rows through at once makes it a GEMM. The formula is the same, but the work done for every byte read is not, and that is what the rest of this chapter is about.
      </p>
    </>
  ),
};

export const gemmAboutCaption = "illustrative: x is 4 values wide and W has 3 rows; TinyLlama's x is 2,048 or 5,632 wide and its W has 2,048 to 32,000 rows";

const cell = 26;
const gap = 4;
const pitch = cell + gap;
const K = 4;
const N = 3;
const B = 3;
const top = 120;
const xLeft = 40;
const wLeft = 225;
const yLeft = 400;
const wTop = top - pitch;
const column = (j: number) => 0.06 + j * 0.12;
const batched = (j: number) => 0.54 + j * 0.12;

export function GemmAboutFigure() {
  const { on } = useLoop(14);
  return (
    <Frame label="x times W transposed: each output value is the dot product of a row of x and a row of W. With one row of x the product is a GEMV; with three rows it is a GEMM, and each column of W transposed read once feeds three outputs">
      <Label x={20} y={24}>{"y = x · Wᵀ: every output is a dot product of a row of x and a row of W"}</Label>
      <motion.g {...on(0.02, 0.48, false)}>
        <Label x={20} y={48} className="fill-text">GEMV: x is one row, as when decoding one token</Label>
      </motion.g>
      <motion.g {...on(0.5, 0.97)}>
        <Label x={20} y={48} className="fill-text">GEMM: x has three rows, as in a batch or a prompt</Label>
      </motion.g>
      <Label x={xLeft} y={wTop - 12}>{`x [rows × ${K}]`}</Label>
      <Label x={wLeft} y={wTop - 12}>{`Wᵀ [${K} × ${N}]`}</Label>
      <Label x={yLeft} y={wTop - 12}>{`y [rows × ${N}]`}</Label>
      {Array.from({ length: B }, (_, r) =>
        Array.from({ length: K }, (_, k) => {
          const rect = <rect x={xLeft + k * pitch} y={top + r * pitch} width={cell} height={cell} rx={3} className="fill-accent/70" />;
          return r === 0 ? (
            <g key={`x${r}${k}`}>{rect}</g>
          ) : (
            <motion.g key={`x${r}${k}`} {...on(0.5, 0.97)}>
              {rect}
            </motion.g>
          );
        }),
      )}
      <circle cx={xLeft + K * pitch + 28} cy={top + cell / 2} r={3} className="fill-text" />
      {Array.from({ length: K }, (_, k) =>
        Array.from({ length: N }, (_, j) => <rect key={`w${k}${j}`} x={wLeft + j * pitch} y={wTop + k * pitch} width={cell} height={cell} rx={3} className="fill-baseline" />),
      )}
      {Array.from({ length: N }, (_, j) => (
        <g key={`c${j}`}>
          <motion.rect x={wLeft + j * pitch - 3} y={wTop - 3} width={cell + 6} height={K * pitch + 2} rx={4} className="fill-none stroke-accent" strokeWidth={2} {...on(column(j), column(j + 1), false)} />
          <motion.rect x={wLeft + j * pitch - 3} y={wTop - 3} width={cell + 6} height={K * pitch + 2} rx={4} className="fill-none stroke-accent" strokeWidth={2} {...on(batched(j), batched(j + 1), false)} />
        </g>
      ))}
      <Label x={wLeft + N * pitch + 22} y={top + cell / 2 + 4} className="fill-text">
        =
      </Label>
      {Array.from({ length: B }, (_, r) =>
        Array.from({ length: N }, (_, j) => (
          <g key={`y${r}${j}`}>
            <rect x={yLeft + j * pitch} y={top + r * pitch} width={cell} height={cell} rx={3} className="fill-surface stroke-border" strokeWidth={0.5} />
            {r === 0 && <motion.rect x={yLeft + j * pitch} y={top + r * pitch} width={cell} height={cell} rx={3} className="fill-accent" {...on(column(j + 1), 0.47, false)} />}
            <motion.rect x={yLeft + j * pitch} y={top + r * pitch} width={cell} height={cell} rx={3} className="fill-accent" {...on(batched(j + 1), 0.97)} />
          </g>
        )),
      )}
      <motion.g {...on(0.02, 0.48, false)}>
        <Label x={20} y={300}>{`1 row × ${N} outputs × ${K} pairs = ${N * K} multiply-adds`}</Label>
        <Label x={20} y={318} className="fill-accent">each value of W is read once and used once</Label>
        <Label x={20} y={346}>a matrix times a vector: GE (general) MV (matrix-vector)</Label>
      </motion.g>
      <motion.g {...on(0.5, 0.97)}>
        <Label x={20} y={300}>{`${B} rows × ${N} outputs × ${K} pairs = ${B * N * K} multiply-adds`}</Label>
        <Label x={20} y={318} className="fill-accent">{`each value of W is read once and used ${B} times`}</Label>
        <Label x={20} y={346}>a matrix times a matrix: GE (general) MM (matrix-matrix)</Label>
      </motion.g>
      <Label x={20} y={378}>the highlighted column of Wᵀ is one row of W, meeting every row of x</Label>
    </Frame>
  );
}

export function GemmAboutStory() {
  return <StickyStory kicker="the idea" steps={[gemmAbout]} caption={() => gemmAboutCaption} figure={() => <GemmAboutFigure />} />;
}
