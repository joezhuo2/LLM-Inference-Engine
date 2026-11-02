"use client";

import clsx from "clsx";
import { AnimatePresence, motion } from "motion/react";
import Link from "next/link";
import type { ReactNode } from "react";
import { StickyStory, type StoryStep } from "@/components/kernels/sticky-story";
import {
  batchSizes,
  benchRun,
  contextPrompts,
  contextRun,
  contextTpot,
  gemmRoofRange,
  nearTies,
  type Provenance,
  profileB16Run,
  profileRun,
  opRank,
  roofIntensity,
  roofline,
  roofPct,
  roofRows,
  scoreboard,
  segmentUs,
  splitsAt,
  stepRows,
  throughput,
  tpot,
  v2Blocks,
  valueAt,
  workload,
} from "@/content/results";
import { ceilings } from "@/lib/data";
import { usePrefersReducedMotion } from "@/lib/reduced-motion";
import { type Axis, Legend, LineChart, type Stat, Stats } from "./line-chart";
import { Scoreboard } from "./scoreboard";
import { GroupLegend, StepBars } from "./step-bars";

const peak = ceilings();
const ridge = (peak.tflops * 1000) / peak.gbs;

const tps = (id: string, b: number) => valueAt(throughput, id, b);
const ms = (id: string, b: number) => valueAt(tpot, id, b);
const ctx = (id: string, prompt: number) => {
  const v = contextTpot.find((s) => s.id === id)?.values[contextPrompts.indexOf(prompt)];
  if (v === undefined) throw new Error(`no ${id} TPOT at prompt ${prompt}`);
  return v;
};

const ordinals = ["", "first", "second", "third", "fourth", "fifth", "sixth", "seventh"];

const n = (v: number, digits = 0) => v.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
const longest = contextPrompts[contextPrompts.length - 1];
const { eager, graphs, b16 } = stepRows;

const batchAxis: Axis = { kind: "log", min: 0.85, max: 19, ticks: batchSizes, format: String, title: "max batch (concurrent requests)" };
const tpsAxis: Axis = { kind: "linear", min: 0, max: 1500, ticks: [0, 500, 1000, 1500], format: (v) => n(v), title: "output tokens/s" };
const tpotAxis: Axis = { kind: "linear", min: 0, max: 40, ticks: [0, 10, 20, 30, 40], format: String, title: "ms per output token (TPOT p50)" };
const contextAxis: Axis = { kind: "linear", min: 0, max: 2048, ticks: contextPrompts, format: (v) => n(v), title: "prompt tokens, batch 1" };
const contextY: Axis = { kind: "linear", min: 0, max: 16, ticks: [0, 4, 8, 12, 16], format: String, title: "ms per output token (TPOT p50, mean of 2 runs)" };
const intensityAxis: Axis = { kind: "log", min: 0.5, max: 200, ticks: [1, 10, 100], format: String, title: "FLOP per byte of DRAM traffic" };
const flopsAxis: Axis = { kind: "log", min: 0.01, max: 60, ticks: [0.01, 0.1, 1, 10], format: String, title: "TFLOP/s" };

function Roof({ sx, sy }: { sx: (v: number) => number; sy: (v: number) => number }) {
  const slope = (x: number) => (x * peak.gbs) / 1000;
  const labelX = 2;
  const angle = (Math.atan2(sy(slope(10)) - sy(slope(1)), sx(10) - sx(1)) * 180) / Math.PI;
  return (
    <g className="pointer-events-none">
      <path d={`M ${sx(0.5)} ${sy(slope(0.5))} L ${sx(ridge)} ${sy(peak.tflops)} L ${sx(200)} ${sy(peak.tflops)}`} className="fill-none stroke-text" strokeWidth={1.25} />
      <text x={sx(labelX)} y={sy(slope(labelX)) - 8} className="fill-muted text-[11px]" transform={`rotate(${angle.toFixed(1)} ${sx(labelX)} ${sy(slope(labelX)) - 8})`}>
        {`${peak.gbs} GB/s`}
      </text>
      <text x={sx(200)} y={sy(peak.tflops) - 6} textAnchor="end" className="fill-muted text-[11px]">
        {`${peak.tflops} TFLOP/s, ridge at ${Math.round(ridge)}`}
      </text>
    </g>
  );
}

const views = ["score", "bench", "bench", "context", "profile", "profile", "roof"] as const;

function benchStats(metric: "tps" | "tpot"): Stat[] {
  if (metric === "tps") {
    return [
      { label: "16 over 1", value: tps("engine_graphs", 16) / tps("engine_graphs", 1), digits: 1, suffix: "x" },
      { label: "over llama.cpp at 16", value: tps("engine_graphs", 16) / tps("llamacpp", 16), digits: 2, suffix: "x" },
      { label: "over HF at 16", value: tps("engine_graphs", 16) / tps("hf", 16), digits: 1, suffix: "x" },
    ];
  }
  return [
    { label: "graphs, batch 1", value: ms("engine_graphs", 1), digits: 2, suffix: " ms" },
    { label: "llama.cpp, batch 1", value: ms("llamacpp", 1), digits: 2, suffix: " ms" },
    { label: "HF, batch 1", value: ms("hf", 1), digits: 2, suffix: " ms" },
  ];
}

function profileStats(second: "graphs" | "b16"): Stat[] {
  if (second === "graphs") {
    return [
      { label: "span, eager", value: eager.span / 1000, digits: 2, suffix: " ms" },
      { label: "span, graphs", value: graphs.span / 1000, digits: 2, suffix: " ms" },
      { label: "idle, graphs", value: graphs.idlePct, digits: 1, suffix: "%" },
    ];
  }
  return [
    { label: "span, batch 16", value: b16.span / 1000, digits: 2, suffix: " ms" },
    { label: "attention, batch 16", value: segmentUs(b16, "attention") / 1000, digits: 2, suffix: " ms" },
    { label: "idle, batch 16", value: b16.idlePct, digits: 1, suffix: "%" },
  ];
}

function View({ step }: { step: number }) {
  switch (views[step]) {
    case "score":
      return <Scoreboard criteria={scoreboard} />;
    case "bench": {
      const metric = step === 1 ? "tps" : "tpot";
      const series = metric === "tps" ? throughput : tpot;
      return (
        <>
          <Legend series={series} />
          <LineChart
            view={metric}
            label={metric === "tps" ? "Output tokens per second against max batch for four systems" : "Median time per output token against max batch for four systems"}
            xs={batchSizes}
            xAxis={batchAxis}
            yAxis={metric === "tps" ? tpsAxis : tpotAxis}
            series={series}
            heading={(i) => `max batch ${batchSizes[i]}`}
            value={(v) => (metric === "tps" ? `${v.toFixed(1)} tok/s` : `${v.toFixed(2)} ms`)}
          />
          <Stats stats={benchStats(metric)} />
        </>
      );
    }
    case "context":
      return (
        <>
          <Legend series={contextTpot} />
          <LineChart
            view="context"
            label="Time per output token at batch 1 against prompt length, with and without split attention and CUDA Graphs"
            xs={contextPrompts}
            xAxis={contextAxis}
            yAxis={contextY}
            series={contextTpot}
            heading={(i) => `prompt ${n(contextPrompts[i])} tokens, ${splitsAt[i]} ${splitsAt[i] === 1 ? "split" : "splits"}`}
            value={(v) => `${v.toFixed(2)} ms`}
          />
          <Stats
            stats={[
              { label: `graphs, v2, ${n(longest)}`, value: ctx("graphs-v2", longest), digits: 2, suffix: " ms" },
              { label: `graphs, split, ${n(longest)}`, value: ctx("graphs-split", longest), digits: 2, suffix: " ms" },
              { label: "faster", value: ctx("graphs-v2", longest) / ctx("graphs-split", longest), digits: 1, suffix: "x" },
            ]}
          />
        </>
      );
    case "profile": {
      const second = step === 4 ? "graphs" : "b16";
      return (
        <>
          <GroupLegend groups={second === "graphs" ? ["gemm", "attention", "other", "graph", "idle"] : ["gemm", "attention", "other", "idle"]} />
          <StepBars rows={[eager, stepRows[second]]} />
          <Stats stats={profileStats(second)} />
        </>
      );
    }
    default:
      return (
        <>
          <Legend series={roofline} />
          <LineChart
            view="roof"
            label="Whole decode steps of each system on a roofline of the measured bandwidth and compute ceilings"
            xs={roofIntensity}
            xAxis={intensityAxis}
            yAxis={flopsAxis}
            series={roofline}
            heading={(i) => `${roofRows[i]} ${roofRows[i] === 1 ? "row" : "rows"}, ${roofIntensity[i].toFixed(1)} FLOP/byte`}
            value={(v) => `${v.toFixed(2)} TFLOP/s`}
            underlay={(sx, sy) => <Roof sx={sx} sy={sy} />}
          />
          <Stats
            stats={[
              { label: "graphs at 16, % of roof", value: roofPct("engine_graphs", 16), digits: 0, suffix: "%" },
              { label: "llama.cpp, at 16", value: roofPct("llamacpp", 16), digits: 0, suffix: "%" },
              { label: "HF, at 16", value: roofPct("hf", 16), digits: 0, suffix: "%" },
            ]}
          />
        </>
      );
  }
}

function ResultsFigure({ step }: { step: number }) {
  const reduced = usePrefersReducedMotion();
  return (
    <div className="grid h-full">
      <AnimatePresence>
        <motion.div
          key={views[step]}
          data-view={views[step]}
          className="col-start-1 row-start-1 flex min-h-0 min-w-0 flex-col gap-2 font-mono text-xs"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: reduced ? 0 : 0.35 }}
        >
          <View step={step} />
        </motion.div>
      </AnimatePresence>
    </div>
  );
}

function Source({ p }: { p: Provenance }) {
  const line = "overflow-x-auto whitespace-nowrap md:overflow-visible md:whitespace-normal";
  return (
    <>
      <div className={line}>{`${p.gpu} · commit ${p.commit} · ${p.date}`}</div>
      <div className={line}>{`source: ${p.source}`}</div>
      <div className={clsx(line, "mt-1")}>
        reproduce: <code className="text-text">{p.command}</code>
      </div>
    </>
  );
}

const sources = [benchRun, benchRun, benchRun, contextRun, profileRun, profileB16Run, { ...benchRun, source: "docs/results.md, Roofline (scripts/roofline.py)" }];

function Story({ to, children }: { to: string; children: ReactNode }) {
  return (
    <Link href={to} className="link-underline text-accent">
      {children}
    </Link>
  );
}

const Num = ({ children }: { children: ReactNode }) => <span className="font-mono text-text">{children}</span>;

const steps: StoryStep[] = [
  {
    id: "scoreboard",
    title: "the scoreboard",
    body: (
      <>
        <p>
          The engine had five criteria to meet before it counted as done: match HuggingFace&apos;s outputs, read the weights at
          a good share of the memory bandwidth at batch 1, gain at least 5x from serving 16 requests at once, explain every
          kernel with a before and after profile, and rebuild every number here from one command.
        </p>
        <p>
          All four measured criteria are met. The one with a footnote is free-running generation: {nearTies ? `${nearTies.exact} of ${nearTies.total}` : "most"} prompts match HuggingFace token for token, and the
          others part ways at a near tie, where HuggingFace&apos;s own top two logits are at most 2 BF16 ulps apart, so a
          rounding difference picks the other token and the rest of the text follows it.
        </p>
      </>
    ),
  },
  {
    id: "throughput",
    title: "tokens per second",
    body: (
      <>
        <p>
          The benchmark sends {workload.split(",")[0]} and lets each system run at most 1 to 16 of them at a time. At batch 1
          the engine with CUDA Graphs makes <Num>{n(tps("engine_graphs", 1), 1)}</Num> tokens/s; at 16 it makes{" "}
          <Num>{n(tps("engine_graphs", 16), 1)}</Num>, since one read of the weights now serves 16 tokens.
        </p>
        <p>
          llama.cpp is the baseline that matters, a mature engine running the same BF16 weights. It stays close up to
          batch 8 and falls behind at 16 (<Num>{n(tps("llamacpp", 16), 1)}</Num> tokens/s). HuggingFace transformers, a
          plain PyTorch decode loop, is far below both. The engine admits new requests into a running batch
          (see <Story to="/kernels/paged-kv">paged KV</Story>); the baselines run static batches, so this compares serving,
          not only kernels.
        </p>
      </>
    ),
  },
  {
    id: "tpot",
    title: "time per token",
    body: (
      <>
        <p>
          The same runs, read as the median time between two output tokens of one request. At batch 1 the engine with
          graphs takes <Num>{n(ms("engine_graphs", 1), 2)} ms</Num> against <Num>{n(ms("llamacpp", 1), 2)} ms</Num> for
          llama.cpp. Each token has to read every weight once, so this number is mostly memory bandwidth.
        </p>
        <p>
          It barely moves with the batch: <Num>{n(ms("engine_graphs", 16), 2)} ms</Num> at 16, because the extra rows add
          arithmetic, not weight reads. CUDA Graphs take <Num>{n(ms("engine", 1) - ms("engine_graphs", 1), 2)} ms</Num> off every token at batch 1 by removing the gaps
          between kernel launches (<Story to="/kernels/cuda-graphs">CUDA Graphs</Story>).
        </p>
      </>
    ),
  },
  {
    id: "context",
    title: "long prompts",
    body: (
      <>
        <p>
          The benchmark above uses 64-token prompts, where attention is a small part of the step. At a{" "}
          {n(longest)}-token prompt it is not: with one block per attention head the GPU runs {v2Blocks} blocks and leaves most of
          its SMs idle, and the time per token rises to <Num>{n(ctx("eager-v2", longest), 2)} ms</Num>.
        </p>
        <p>
          Split attention cuts each row&apos;s context into up to {Math.max(...splitsAt)} parts that run side by side and merges them (
          <Story to="/kernels/split-attention">split attention</Story>). With one CUDA graph captured per split count, the
          time per token at {n(longest)} tokens drops from <Num>{n(ctx("graphs-v2", longest), 2)}</Num> to{" "}
          <Num>{n(ctx("graphs-split", longest), 2)} ms</Num>, close to the short-prompt figure. These runs come from the
          profiling notes, not from the benchmark script.
        </p>
      </>
    ),
  },
  {
    id: "step",
    title: "inside a batch-1 step",
    body: (
      <>
        <p>
          Nsight Systems splits a decode step by op. The five cuBLAS matrix products take most of it; gate/up alone is{" "}
          <Num>{n(segmentUs(eager, "gate_up"), 1)} µs</Num>. They already read memory at {gemmRoofRange.min} to {gemmRoofRange.max}% of the measured peak (
          <Story to="/kernels/gemm">GEMM</Story>), so the place to gain was not the arithmetic.
        </p>
        <p>
          Launched one by one, the GPU sits idle for <Num>{eager.idlePct}%</Num> of the step, in {eager.gaps} gaps over {eager.gapUs} µs.
          Replayed from a CUDA graph, the same {graphs.kernels} kernels run with{" "}
          <Num>{graphs.idlePct}%</Num> idle, and the step shrinks from <Num>{n(eager.span)}</Num> to{" "}
          <Num>{n(graphs.span)} µs</Num>.
        </p>
      </>
    ),
  },
  {
    id: "batch16",
    title: "sixteen rows in one step",
    body: (
      <>
        <p>
          With 16 requests in the step, the matrix products take almost the same time as for one:{" "}
          <Num>{n(segmentUs(b16, "gate_up"), 1)} µs</Num> for gate/up against <Num>{n(segmentUs(eager, "gate_up"), 1)}</Num>. Each
          weight byte is read once and used 16 times.
        </p>
        <p>
          Attention is different: every row reads its own keys and values, so it grows from{" "}
          <Num>{n(segmentUs(eager, "attention"), 1)}</Num> to <Num>{n(segmentUs(b16, "attention"), 1)} µs</Num> and becomes the{" "}
          {ordinals[opRank(b16, "attention")] ?? `number ${opRank(b16, "attention")}`} biggest op, against the{" "}
          {ordinals[opRank(eager, "attention")] ?? `number ${opRank(eager, "attention")}`} at batch 1. That makes it the place with the most headroom at batch 16 (
          <Story to="/kernels/attention-v2">attention v2</Story>).
        </p>
      </>
    ),
  },
  {
    id: "roofline",
    title: "against the roof",
    body: (
      <>
        <p>
          A roofline places each whole decode step by FLOPs per byte of memory traffic and by speed, under two ceilings
          measured on this GPU: {peak.gbs} GB/s of bandwidth and {peak.tflops} TFLOP/s of BF16 matrix math. Every system sits
          on the bandwidth slope, far left of the ridge at {Math.round(ridge)} FLOP/byte, so none of them is limited by
          arithmetic.
        </p>
        <p>
          The question is how close each gets to the slope. With graphs the engine reaches{" "}
          <Num>{roofPct("engine_graphs", 1)}%</Num> of it at batch 1 and <Num>{roofPct("engine_graphs", 16)}%</Num> at 16,
          against <Num>{roofPct("llamacpp", 16)}%</Num> for llama.cpp and <Num>{roofPct("hf", 16)}%</Num> for HuggingFace.
          The op by op version is on the <Story to="/kernels/gemm">GEMM</Story> page.
        </p>
      </>
    ),
  },
];

export function ResultsStory() {
  return <StickyStory morph steps={steps} figure={(i) => <ResultsFigure step={i} />} caption={(i) => <Source p={sources[i]} />} />;
}
