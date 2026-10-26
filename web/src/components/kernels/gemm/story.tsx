"use client";

import NumberFlow from "@number-flow/react";
import { motion } from "motion/react";
import { useState } from "react";
import { gemmOps, roofline } from "@/content/kernels/gemm";
import { measured } from "@/content/kernels/types";
import { ceilings, cellNumber, decodeSteps } from "@/lib/data";
import { StickyStory, type StoryStep } from "../sticky-story";
import { Toggle } from "../toggle";
import { useLoop } from "../loop";
import { bytesOf, GemvFigure, type Projection, ProjectionsFigure, ReuseFigure, RoofBarsFigure, TransposeFigure } from "./figures";

const peak = ceilings();

const labels: Record<string, string> = { qkv: "QKV", o_proj: "output", gate_up: "gate/up", down: "down", lm_head: "LM head", attention: "attention" };

const projections: Projection[] = [
  { op: "qkv", label: "QKV", out: 2560, in: 2048, per: 22 },
  { op: "o_proj", label: "output", out: 2048, in: 2048, per: 22 },
  { op: "gate_up", label: "gate/up", out: 11264, in: 2048, per: 22 },
  { op: "down", label: "down", out: 2048, in: 5632, per: 22 },
  { op: "lm_head", label: "LM head", out: 32000, in: 2048, per: 1 },
].map((p) => ({ ...p, us: roofline(p.op, "1", "us/step") }));

const weightBytes = projections.reduce((s, p) => s + bytesOf(p), 0);

const share = (() => {
  const t = decodeSteps.b1;
  if (!t) return 0;
  const col = t.header.indexOf("% of kernel time");
  return t.rows.filter((r) => (gemmOps as readonly string[]).includes(r[0])).reduce((s, r) => s + (cellNumber(r[col]) ?? 0), 0);
})();

const step = (system: string, rows: string, column: string) => measured("roofline", { "whole decode step": system, rows }, column, 1, "results");

const tpotGraphs = step("engine + graphs", "1", "TPOT us");
const sum = (rows: string) => gemmOps.reduce((s, op) => s + roofline(op, rows, "us/step"), 0);

const roofRows = [...gemmOps, "attention"].map((op) => ({
  label: labels[op],
  gbs: roofline(op, "1", "GB/s"),
  pct: roofline(op, "1", "% of roof"),
  warn: op === "attention",
}));

const W = 560;
const H = 300;
const pad = { l: 46, r: 16, t: 14, b: 34 };
const xr = [0.5, 500];
const yr = [0.01, 100];
const lx = (v: number) => pad.l + ((Math.log10(v) - Math.log10(xr[0])) / (Math.log10(xr[1]) - Math.log10(xr[0]))) * (W - pad.l - pad.r);
const ly = (v: number) => H - pad.b - ((Math.log10(v) - Math.log10(yr[0])) / (Math.log10(yr[1]) - Math.log10(yr[0]))) * (H - pad.t - pad.b);
const ridge = (peak.tflops * 1e12) / (peak.gbs * 1e9);

function RooflinePicker() {
  const [rows, setRows] = useState("1");
  const { reduced } = useLoop(1);
  const ease = { duration: reduced ? 0 : 0.7, ease: "easeInOut" as const };
  const points = [
    ...gemmOps.map((op) => ({ id: op, x: roofline(op, rows, "FLOP/byte"), y: roofline(op, rows, "TFLOP/s"), cls: "fill-accent" })),
    { id: "attention", x: roofline("attention", rows, "FLOP/byte"), y: roofline("attention", rows, "TFLOP/s"), cls: "fill-warn" },
    { id: "step", x: step("engine + graphs", rows, "FLOP/byte"), y: step("engine + graphs", rows, "TFLOP/s"), cls: "fill-text" },
  ];
  const gu = points[2];
  const att = points[5];
  const whole = points[6];
  return (
    <div className="flex h-full flex-col gap-2 font-mono text-xs">
      <Toggle label="rows" choices={["1", "16"]} value={rows} onChange={setRows} />
      <svg viewBox={`0 0 ${W} ${H}`} className="min-h-0 w-full flex-1" role="img" aria-label={`Roofline at ${rows} rows: the five projections, attention and the whole decode step against the measured bandwidth and compute ceilings`}>
        {[0.01, 0.1, 1, 10, 100].map((v) => (
          <g key={v}>
            <line x1={pad.l} x2={W - pad.r} y1={ly(v)} y2={ly(v)} className="stroke-border" strokeWidth={0.5} />
            <text x={pad.l - 6} y={ly(v) + 3} textAnchor="end" className="fill-muted text-[10px]">
              {v}
            </text>
          </g>
        ))}
        {[1, 10, 100].map((v) => (
          <text key={v} x={lx(v)} y={H - pad.b + 14} textAnchor="middle" className="fill-muted text-[10px]">
            {v}
          </text>
        ))}
        <text x={W - pad.r} y={H - 6} textAnchor="end" className="fill-muted text-[10px]">
          FLOP per byte of DRAM traffic
        </text>
        <text x={pad.l} y={pad.t - 2} className="fill-muted text-[10px]">
          TFLOP/s
        </text>
        <path d={`M ${lx(xr[0])} ${ly((xr[0] * peak.gbs) / 1000)} L ${lx(ridge)} ${ly(peak.tflops)} L ${lx(xr[1])} ${ly(peak.tflops)}`} className="fill-none stroke-text" strokeWidth={1.25} />
        <text x={lx(2.2)} y={ly((2.2 * peak.gbs) / 1000) - 8} className="fill-text text-[10px]" transform={`rotate(-33 ${lx(2.2)} ${ly((2.2 * peak.gbs) / 1000) - 8})`}>
          {`${peak.gbs} GB/s`}
        </text>
        <text x={W - pad.r} y={ly(peak.tflops) - 6} textAnchor="end" className="fill-text text-[10px]">
          {`${peak.tflops} TFLOP/s, ridge at ${Math.round(ridge)}`}
        </text>
        {points.map((p) => (
          <motion.g key={p.id} initial={false} animate={{ x: lx(p.x), y: ly(p.y) }} transition={ease}>
            <circle r={p.id === "step" ? 5 : 4} className={p.cls} opacity={p.id === "step" ? 0.9 : 0.8} />
          </motion.g>
        ))}
        <motion.g initial={false} animate={{ x: Math.max(lx(gu.x) - 10, pad.l + 100), y: ly(gu.y) - (lx(gu.x) - 10 < pad.l + 100 ? 24 : 12) }} transition={ease}>
          <text textAnchor="end" className="fill-accent text-[10px]">
            five projections
          </text>
        </motion.g>
        <motion.g initial={false} animate={{ x: lx(att.x) + 8, y: ly(att.y) + 4 }} transition={ease}>
          <text className="fill-warn text-[10px]">attention</text>
        </motion.g>
        <motion.g initial={false} animate={{ x: lx(whole.x) + 8, y: ly(whole.y) + 16 }} transition={ease}>
          <text className="fill-text text-[10px]">whole step, graphs</text>
        </motion.g>
      </svg>
      <div className="mt-auto grid grid-cols-3 gap-x-4 border-t border-border pt-2 tabular-nums">
        {[
          { label: "gate/up, TFLOP/s", value: roofline("gate_up", rows, "TFLOP/s"), digits: 2, cls: "text-accent" },
          { label: "attention, TFLOP/s", value: roofline("attention", rows, "TFLOP/s"), digits: 2, cls: "text-warn" },
          { label: "whole step, % of roof", value: step("engine + graphs", rows, "% of roof"), digits: 0, cls: "text-text", suffix: "%" },
        ].map((r) => (
          <div key={r.label}>
            <div className="text-muted">{r.label}</div>
            <div className={r.cls}>
              <NumberFlow value={r.value} suffix={r.suffix} format={{ minimumFractionDigits: r.digits, maximumFractionDigits: r.digits }} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

const steps: StoryStep[] = [
  {
    id: "job",
    title: "the job",
    body: (
      <>
        <p>
          Most of a transformer is matrix multiplication. Each of TinyLlama&apos;s 22 layers multiplies the hidden state by
          four weight matrices: QKV (the three attention inputs, stored as one matrix), the attention output, gate/up (the
          MLP&apos;s two widening projections, also one matrix) and down. After the last layer the LM head turns the hidden
          state into 32,000 logits.
        </p>
        <p>
          The engine does not have its own kernel for any of them. All five go through one call to cuBLAS, NVIDIA&apos;s
          library, and this page is about why that is the right call here and how close it gets to the hardware.
        </p>
      </>
    ),
  },
  {
    id: "trick",
    title: "the row-major trick",
    body: (
      <>
        <p>
          The model stores W as [out, in] in row-major order, one output row after the other. cuBLAS, which comes from
          Fortran, reads every matrix column-major. Read column-major with a leading dimension of in, the same bytes are
          [in, out]: Wᵀ, with no copy.
        </p>
        <p>
          So the wrapper asks for the transposed problem. With the first operand transposed, cuBLAS computes W · Xᵀ, which
          is Yᵀ, and Yᵀ written column-major is exactly Y in row-major memory. One call, BF16 inputs and output, FP32
          accumulation.
        </p>
      </>
    ),
  },
  {
    id: "gemv",
    title: "batch 1 is a GEMV",
    body: (
      <>
        <p>
          Decoding one token means x has one row. A matrix times a vector reads every weight exactly once and uses it for
          exactly one multiply-add: 2 FLOPs for 2 bytes. The GPU can do over a hundred times more arithmetic than that per
          byte it reads, so the arithmetic is free and the time is the time it takes to read the weights.
        </p>
        <p>
          Every token reads all{" "}
          <span className="font-mono text-text">{`${(weightBytes / 1e9).toFixed(2)} GB`}</span>, which sets a floor of{" "}
          <span className="font-mono text-text">{`${(weightBytes / (peak.gbs * 1e6)).toFixed(2)} ms`}</span> per token on
          this GPU. cuBLAS sees the single row and runs a dedicated gemv kernel instead of a GEMM.
        </p>
      </>
    ),
  },
  {
    id: "roof",
    title: "at the roof",
    body: (
      <>
        <p>
          Dividing the bytes each projection reads by its time in the profile gives its bandwidth. All five reach{" "}
          <span className="font-mono text-accent">{`${Math.min(...roofRows.slice(0, 5).map((r) => r.pct))} to ${Math.max(...roofRows.slice(0, 5).map((r) => r.pct))}%`}</span>{" "}
          of the {peak.gbs} GB/s a plain copy kernel measures on this GPU. A hand-written GEMV could not do better: it has to
          read the same bytes through the same memory.
        </p>
        <p>
          That is why the engine&apos;s own kernels went elsewhere. The projections are{" "}
          <span className="font-mono text-text">{`${share.toFixed(1)}%`}</span> of a decode step&apos;s GPU time and already
          at the roof; the gains were in the launch gaps (CUDA Graphs) and in attention.
        </p>
      </>
    ),
  },
  {
    id: "batch",
    title: "sixteen rows",
    body: (
      <>
        <p>
          With 16 requests decoding together, x has 16 rows. Each weight row is still read once, but now it meets 16
          input rows: 16 FLOPs per byte. That is still far left of the ridge, so the GPU is still waiting on memory, and the
          projections take about the same time as for one row:{" "}
          <span className="font-mono text-accent">{`${Math.round(sum("16")).toLocaleString("en-US")} µs`}</span> against{" "}
          <span className="font-mono text-text">{`${Math.round(sum("1")).toLocaleString("en-US")} µs`}</span> per step.
        </p>
        <p>
          Sixteen tokens for the price of one is where batching gets its throughput. cuBLAS switches to tensor-core GEMM
          kernels at this size; the call is the same.
        </p>
      </>
    ),
  },
  {
    id: "roofline",
    title: "the roofline",
    body: (
      <>
        <p>
          A roofline puts each op at its FLOPs per byte and its speed, under two measured ceilings: {peak.gbs} GB/s on the
          slope, {peak.tflops} TFLOP/s from square cuBLAS GEMMs on the flat part. Switch between 1 and 16 rows on the right:
          the projections slide up the slope, attention stays where it is, because every row reads its own keys and values.
        </p>
        <p>
          Nothing in a decode step gets near the ridge at {Math.round(ridge)} FLOP/byte, so nothing here is compute bound.
          The way past the roof is fewer bytes per token: quantized weights, or more rows sharing each weight read.
        </p>
      </>
    ),
  },
];

const captions = [
  "bytes from the weight shapes in config.json; GPU time per op from the batch-1 Nsight Systems profile in results/decode_step.md",
  "a 3 × 4 W drawn; the engine's real call has out = 2,560 to 32,000 and in = 2,048 or 5,632",
  "12 of 2,560 QKV rows drawn; the floor is 2.07 GB ÷ the measured bandwidth; time per token from docs/results.md",
  "GB/s from scripts/roofline.py over the batch-1 profile; the peak from the bandwidth test in docs/results.md",
  "µs per step from the batch-1 and batch-16 profiles; TFLOP/s from scripts/roofline.py",
  "docs/results.md, Roofline: RTX 5060 Laptop GPU, ceilings measured on the same GPU",
];

export function GemmStory() {
  return (
    <StickyStory
      steps={steps}
      caption={(i) => captions[i]}
      figure={(i) => {
        switch (i) {
          case 0:
            return <ProjectionsFigure projections={projections} share={share} />;
          case 1:
            return <TransposeFigure />;
          case 2:
            return <GemvFigure weightBytes={weightBytes} peakGbs={peak.gbs} tpotUs={tpotGraphs} />;
          case 3:
            return <RoofBarsFigure rows={roofRows} peakGbs={peak.gbs} />;
          case 4:
            return <ReuseFigure us1={sum("1")} us16={sum("16")} tf1={roofline("gate_up", "1", "TFLOP/s")} tf16={roofline("gate_up", "16", "TFLOP/s")} />;
          default:
            return <RooflinePicker />;
        }
      }}
    />
  );
}
