"use client";

import NumberFlow from "@number-flow/react";
import Link from "next/link";
import { useState } from "react";
import { gemmOps, roofline } from "@/content/kernels/gemm";
import { measured } from "@/content/kernels/types";
import { ceilings, hardware, onchip, onchipRun } from "@/lib/data";
import { EasedBar } from "../eased-bar";
import { StickyStory, type StoryStep } from "../sticky-story";
import { Toggle } from "../toggle";
import { ContrastFigure, FusionFigure, HierarchyFigure, type Lane, MemoryFigure, RooflineFigure, type Tier } from "./figures";

const hw = hardware();
const peak = ceilings();

const blocks = 22;
const threadsPerBlock = 256;

const layers = 22;
const out = 11264;
const inn = 2048;
const weightBytes = out * inn * 2;
const flopsPerRow = 2 * out * inn;
const readUs = (weightBytes / (peak.gbs * 1e9)) * 1e6;
const mathUs = (rows: number) => ((rows * flopsPerRow) / (peak.tflops * 1e12)) * 1e6;
const ridge = (peak.tflops * 1e12) / (peak.gbs * 1e9);
const flopPerByte1 = roofline("gate_up", "1", "FLOP/byte");
const prefillRows = 512;
const rowChoices = ["1", "16", "64", "512"];
const measuredRows = ["1", "16"];

const wholeStep = (rows: string) => measured("roofline", { "whole decode step": "engine + graphs", rows }, "% of roof", 1, "results");
const gemvRoof = gemmOps.map((op) => roofline(op, "1", "% of roof"));
const measuredUs = (rows: string) => roofline("gate_up", rows, "us/step") / layers;

const lanes: Lane[] = [
  { title: "decode: one token, a matrix-vector product", rows: 1, flopPerByte: flopPerByte1, memUs: readUs, computeUs: mathUs(1), note: `the arithmetic needs ${((mathUs(1) / readUs) * 100).toFixed(1)}% of the read time` },
  { title: "prefill: a long prompt, a matrix product", rows: prefillRows, flopPerByte: flopPerByte1 * prefillRows, memUs: readUs, computeUs: mathUs(prefillRows), note: `the read needs ${((readUs / mathUs(prefillRows)) * 100).toFixed(0)}% of the arithmetic time` },
];

const fusedSection = "fused-add-rmsnorm-branch-30";
const fusionTokens = 64;
const fusion = {
  tokens: fusionTokens,
  unfusedKb: measured(fusedSection, { tokens: String(fusionTokens) }, "naive DRAM (KB)"),
  fusedKb: measured(fusedSection, { tokens: String(fusionTokens) }, "fast DRAM (KB)"),
};

const mb = (kb: number) => `${(kb / 1024).toLocaleString("en-US", { maximumFractionDigits: 1 })} MB`;

const run = onchipRun();
const whole = (n: number) => n.toLocaleString("en-US", { maximumFractionDigits: 0 });
const nanos = (n: number) => `${n.toFixed(n < 10 ? 1 : 0)} ns`;
const chip = {
  dramNs: onchip("dram_latency", "ns").median,
  dramCycles: onchip("dram_latency", "cycles").median,
  l2Gbs: onchip("l2_bandwidth", "GB/s").median,
  l2Ns: onchip("l2_latency", "ns").median,
  l2Cycles: onchip("l2_latency", "cycles").median,
  smem128Gbs: onchip("smem_bandwidth_128bit", "GB/s").median,
  smem32Gbs: onchip("smem_bandwidth_32bit", "GB/s").median,
  smemNs: onchip("smem_latency", "ns").median,
  smemCycles: onchip("smem_latency", "cycles").median,
  fmaCycles: onchip("fma_latency", "cycles").median,
  regKb: onchip("register_file", "KB/SM").median,
  regTotalKb: onchip("register_file", "KB").median,
  mhz: onchip("smem_bandwidth_128bit", "GB/s").mhz ?? 0,
};

const tiers: Tier[] = [
  { name: "global memory (DRAM)", size: `${hw.vram_gb} GB`, lines: ["off the chip, shared by every SM", `${peak.gbs} GB/s measured`, `load latency ${nanos(chip.dramNs)}`], box: { x: 10, y: 40, w: 280, h: 50 } },
  { name: "L2 cache", size: mb(hw.l2_kb), lines: ["on the chip, shared by every SM", `${whole(chip.l2Gbs)} GB/s read`, `hit latency ${nanos(chip.l2Ns)}`], box: { x: 35, y: 130, w: 230, h: 44 } },
  { name: "shared memory", size: `${hw.shared_kb_per_sm} KB per SM`, lines: [`one block's scratchpad, ${hw.sms} SMs`, `${whole(chip.smem128Gbs)} GB/s, 128-bit loads`, `load latency ${nanos(chip.smemNs)}`], box: { x: 60, y: 214, w: 180, h: 44 } },
  { name: "registers", size: `${whole(chip.regKb)} KB per SM`, lines: ["private to one thread", `${mb(chip.regTotalKb)} over ${hw.sms} SMs`, `FMA to FMA: ${Math.round(chip.fmaCycles)} cycles`], box: { x: 85, y: 298, w: 130, h: 44 } },
];

export function RowsPicker() {
  const [rows, setRows] = useState("1");
  const n = Number(rows);
  const bars = [
    { label: "memory", us: readUs, bound: readUs >= mathUs(n) },
    { label: "compute", us: mathUs(n), bound: mathUs(n) > readUs },
  ];
  const max = mathUs(Math.max(...rowChoices.map(Number)));
  const isMeasured = measuredRows.includes(rows);
  return (
    <div className="flex h-full flex-col gap-3 tabular-nums text-xs">
      <Toggle label="rows" choices={rowChoices} value={rows} onChange={setRows} />
      <div className="space-y-1.5 pt-1">
        <div className="text-muted">gate/up of one layer, µs at the ceilings</div>
        {bars.map((b) => (
          <div key={b.label} className="grid grid-cols-[5.5rem_1fr_4.5rem] items-center gap-2">
            <span className={b.bound ? "text-accent" : "text-muted"}>{b.label}</span>
            <div className="h-3 min-w-0 rounded-full bg-border">
              <EasedBar fraction={b.us / max} className={b.bound ? "bg-accent" : "bg-baseline/70"} />
            </div>
            <span className="text-right tabular-nums text-muted">
              <NumberFlow value={b.us} suffix=" µs" format={{ minimumFractionDigits: 1, maximumFractionDigits: 1 }} />
            </span>
          </div>
        ))}
      </div>
      <div className="space-y-1 leading-relaxed">
        <div>
          <span className="text-muted">FLOP per byte </span>
          <span className="text-text tabular-nums">{(flopPerByte1 * n).toFixed(1)}</span>
          <span className="text-muted">{` against a ridge of ${Math.round(ridge)}: `}</span>
          <span className="text-accent">{n < ridge ? "memory bound" : "compute bound"}</span>
        </div>
        <div className="text-muted">
          {isMeasured ? (
            <>
              {"measured: "}
              <span className="text-text tabular-nums">{`${measuredUs(rows).toFixed(1)} µs, ${roofline("gate_up", rows, "% of roof")}% of the bandwidth roof`}</span>
            </>
          ) : (
            "not measured: computed from the two ceilings"
          )}
        </div>
      </div>
      <div className="mt-auto space-y-1 border-t border-border pt-2 leading-relaxed">
        <div>
          <span className="text-muted">five GEMVs at batch 1: </span>
          <span className="text-accent tabular-nums">{`${Math.min(...gemvRoof)} to ${Math.max(...gemvRoof)}% of the roof`}</span>
        </div>
        <div>
          <span className="text-muted">whole decode step, CUDA Graphs: </span>
          <span className="text-accent tabular-nums">{`${wholeStep("1")}% of the roof`}</span>
        </div>
      </div>
    </div>
  );
}

const steps: StoryStep[] = [
  {
    id: "hierarchy",
    title: "grid, block, warp, thread",
    body: (
      <>
        <p>
          A CUDA kernel is one function that thousands of threads run at once, each on its own piece of the data. One launch
          is a grid. The grid is cut into blocks, every block has the same number of threads, and a block runs on one
          streaming multiprocessor (SM) from start to finish. This GPU has {hw.sms} SMs.
        </p>
        <p>
          Inside a block the hardware groups threads in warps of 32. A warp executes one instruction at a time for all 32
          threads, so a thread that wants a different branch than its neighbors makes the warp run both. SwiGLU&apos;s first
          draft launched {blocks} blocks of {threadsPerBlock} threads per row: 8 warps in each block.
        </p>
      </>
    ),
  },
  {
    id: "memory",
    title: "four places to keep a number",
    body: (
      <>
        <p>
          A thread can work on data in four places. Global memory is the {hw.vram_gb} GB of DRAM that holds the weights and
          the KV cache; its measured bandwidth on this GPU is{" "}
          <span className="text-accent">{peak.gbs} GB/s</span>. On a laptop GPU it is GDDR, not HBM, but the role
          is the same. The L2 cache ({mb(hw.l2_kb)}) sits in front of it and every SM shares it.
        </p>
        <p>
          Each SM also has {hw.shared_kb_per_sm} KB of shared memory, a scratchpad for the threads of a block, and{" "}
          {whole(chip.regKb)} KB of registers split among its threads: {mb(hw.sms * hw.shared_kb_per_sm)} and{" "}
          {mb(chip.regTotalKb)} over all {hw.sms} SMs.
        </p>
        <p>
          Each step closer to the thread is faster, measured on this GPU: a load waits{" "}
          <span className="text-text">{nanos(chip.dramNs)}</span> from DRAM,{" "}
          <span className="text-text">{nanos(chip.l2Ns)}</span> from L2 and{" "}
          <span className="text-text">{nanos(chip.smemNs)}</span> from shared memory, and an FMA&apos;s result
          feeds the next after {Math.round(chip.fmaCycles)} cycles. L2 streams {whole(chip.l2Gbs)} GB/s, shared memory{" "}
          <span className="text-accent">{whole(chip.smem128Gbs)} GB/s</span> with 128-bit loads and{" "}
          {whole(chip.smem32Gbs)} GB/s with 32-bit ones.
        </p>
      </>
    ),
  },
  {
    id: "bytes",
    title: "bytes against FLOPs",
    body: (
      <>
        <p>
          Every operation has to move some bytes from memory and do some arithmetic, and it takes as long as the slower of
          the two. Take one layer&apos;s gate/up projection: [{out.toLocaleString("en-US")} × {inn.toLocaleString("en-US")}]
          BF16 weights, {(weightBytes / 1e6).toFixed(0)} MB. Reading them at the measured {peak.gbs} GB/s takes{" "}
          <span className="text-text">{readUs.toFixed(1)} µs</span>, however many tokens use them.
        </p>
        <p>
          Decoding one token is a matrix times a vector: 2 FLOPs for every 2-byte weight. At {peak.tflops} TFLOP/s the
          arithmetic needs <span className="text-text">{mathUs(1).toFixed(1)} µs</span> and the GPU waits on
          memory: memory bound. Prefill with {prefillRows} prompt tokens uses each weight {prefillRows} times, needs{" "}
          <span className="text-accent">{mathUs(prefillRows).toFixed(0)} µs</span> of arithmetic and waits on
          nothing but the math: compute bound.
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
          A roofline plots speed against FLOPs per byte. Its slope is the memory bandwidth and its flat top the compute peak,
          and they meet at the ridge, {Math.round(ridge)} FLOP/byte on this GPU ({peak.tflops} TFLOP/s ÷ {peak.gbs} GB/s). Left
          of the ridge an operation is memory bound, right of it compute bound.
        </p>
        <p>
          Rows that share one read of the weights set the position: FLOP/byte is about the number of rows. The ring climbs the
          slope as the batch grows and flattens at the ridge. The dots are measured: gate/up at 1 and 16 rows, on the slope.
        </p>
      </>
    ),
  },
  {
    id: "measured",
    title: "decode sits on the slope",
    body: (
      <>
        <p>
          At batch 1 the five matrix-vector products of a decode step run at{" "}
          <span className="text-accent">{`${Math.min(...gemvRoof)} to ${Math.max(...gemvRoof)}%`}</span> of the
          bandwidth roof, and the whole step with CUDA Graphs at{" "}
          <span className="text-accent">{wholeStep("1")}%</span>. Nothing in a decode step is near the ridge, so a
          faster multiply would not make a token faster: moving fewer bytes would.
        </p>
        <p>
          That is the memory wall. Try the row counts on the right: the memory time never changes, and the compute time only
          overtakes it past the ridge. Only 1 and 16 rows were measured; the others come from the two ceilings.
        </p>
      </>
    ),
  },
  {
    id: "fusion",
    title: "fewer trips to memory",
    body: (
      <>
        <p>
          Each separate kernel starts by reading its inputs from global memory and ends by writing its outputs back, so two
          kernels in a row pay for the data between them twice. Fusing them keeps that intermediate in registers or shared
          memory. For add + RMSNorm that is the sum: the naive pair writes it and reads it back twice, the fused kernel does
          not, and the DRAM traffic at {fusion.tokens} tokens drops from{" "}
          <span className="text-text">{fusion.unfusedKb} KB</span> to{" "}
          <span className="text-accent">{fusion.fusedKb} KB</span>.
        </p>
        <p>
          The kernel stories that follow are this idea at work, each on one small operation of a layer:{" "}
          <Link href="/kernels/add-rmsnorm" transitionTypes={["nav-forward"]} className="link-underline text-accent">
            add + RMSNorm
          </Link>{" "}
          fuses two kernels into one,{" "}
          <Link href="/kernels/swiglu" transitionTypes={["nav-forward"]} className="link-underline text-accent">
            SwiGLU
          </Link>{" "}
          moves the gate/up row once with 16-byte loads, and{" "}
          <Link href="/kernels/rope" transitionTypes={["nav-forward"]} className="link-underline text-accent">
            RoPE
          </Link>{" "}
          touches only the Q and K slices. Each is under 2% of a decode step; the step is mostly the GEMVs, already at the
          roof.
        </p>
      </>
    ),
  },
];

const captions = [
  `all ${blocks} blocks drawn, one block and one warp zoomed; SM count from docs/hardware.md`,
  `${hw.gpu}: sizes from docs/hardware.md, DRAM bandwidth from the bandwidth test, the rest from scripts/onchip_bench.cu: median of ${run.runs} runs at about ${whole(chip.mhz)} MHz, commit ${run.commit}, ${run.date.slice(0, 10)}`,
  `bytes and FLOPs of the gate/up weights from config.json, ceilings from docs/results.md; ${prefillRows} rows is computed from the ceilings, not measured`,
  "docs/results.md, Roofline: gate/up at 1 and 16 rows measured; the ring follows the two ceilings",
  "docs/results.md, Roofline: RTX 5060 Laptop GPU, ceilings measured on the same GPU",
  `DRAM bytes per launch at ${fusion.tokens} tokens from Nsight Compute, docs/profiling.md; each packet stands for a whole tensor, not to scale`,
];

export const foundationsSteps = steps;
export const foundationsCaptions = captions;
export const foundationsNumbers = {
  sms: hw.sms,
  blocks,
  threadsPerBlock,
  tiers,
  gbs: peak.gbs,
  tflops: peak.tflops,
  ridge,
  out,
  inn,
  readUs,
  decodeUs: mathUs(1),
  prefillUs: mathUs(prefillRows),
  prefillRows,
  flopPerByte1,
  measured: measuredRows.map((rows) => ({ rows: Number(rows), x: roofline("gate_up", rows, "FLOP/byte"), y: roofline("gate_up", rows, "TFLOP/s") })),
  fusion,
};

export function FoundationsStory() {
  return (
    <StickyStory
      steps={steps}
      caption={(i) => captions[i]}
      figure={(i) => {
        switch (i) {
          case 0:
            return <HierarchyFigure sms={hw.sms} blocks={blocks} threadsPerBlock={threadsPerBlock} />;
          case 1:
            return <MemoryFigure tiers={tiers} footer={`latency in cycles: DRAM ${whole(chip.dramCycles)}, L2 ${whole(chip.l2Cycles)}, shared ${whole(chip.smemCycles)}, FMA ${Math.round(chip.fmaCycles)}`} />;
          case 2:
            return <ContrastFigure lanes={lanes} gpuLine={[`gate/up at 1 row measured: ${measuredUs("1").toFixed(1)} µs per layer, at the roof`, `${prefillRows} rows: computed from the ceilings, not measured`]} />;
          case 3:
            return (
              <RooflineFigure
                gbs={peak.gbs}
                tflops={peak.tflops}
                rowsPath={[1, 16, Math.round(ridge), prefillRows].map((rows) => ({ rows, flopPerByte: flopPerByte1 * rows }))}
                measured={[
                  { x: roofline("gate_up", "1", "FLOP/byte"), y: roofline("gate_up", "1", "TFLOP/s"), label: "gate/up, 1 row" },
                  { x: roofline("gate_up", "16", "FLOP/byte"), y: roofline("gate_up", "16", "TFLOP/s"), label: "gate/up, 16 rows" },
                ]}
              />
            );
          case 4:
            return <RowsPicker />;
          default:
            return <FusionFigure numbers={fusion} />;
        }
      }}
    />
  );
}
