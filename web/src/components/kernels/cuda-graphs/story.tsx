"use client";

import NumberFlow from "@number-flow/react";
import { motion } from "motion/react";
import { useState } from "react";
import { longPrompt, twoRuns } from "@/content/kernels/cuda-graphs";
import { measured } from "@/content/kernels/types";
import { launchGaps } from "@/lib/data";
import { StickyStory, type StoryStep } from "../sticky-story";
import { Toggle } from "../toggle";
import { useLoop } from "../loop";
import trace from "@/content/kernels/cuda-graphs-trace.json";
import { CaptureFigure, EagerTimelineFigure, eagerLayer, GapFigure, graphLayer, GraphGridFigure, layerIndex, StageFigure } from "./figures";

const eager = launchGaps("eager");
const replay = launchGaps("graphs");

const splitGraphs = "split-attention-in-cuda-graphs-branch-43";
const longRun = {
  before: twoRuns(splitGraphs, longPrompt, "main graphs"),
  after: twoRuns(splitGraphs, longPrompt, "this branch graphs"),
  eager: twoRuns(splitGraphs, longPrompt, "eager"),
};

const tpot = (system: string, batch: string) => measured("throughput-and-latency", { system, "max batch": batch }, "TPOT p50 ms", 0, "results");

const layerUs = (ks: typeof eagerLayer) => ks[ks.length - 1].end;

const W = 560;
const H = 150;
const x0 = 20;
const lane = 520;
const total = Math.ceil(layerUs(eagerLayer) / 10) * 10 + 10;
const px = (t: number) => (lane * t) / total;
const gemm = new Set(["qkv", "o_proj", "gate_up", "down"]);

function ReplayPicker() {
  const [mode, setMode] = useState("eager");
  const [batch, setBatch] = useState("1");
  const { reduced } = useLoop(1);
  const ease = { duration: reduced ? 0 : 0.7, ease: "easeInOut" as const };
  const ks = mode === "eager" ? eagerLayer : graphLayer;
  const end = layerUs(ks);
  const stats = [
    { label: `layer ${layerIndex}, µs`, value: end, digits: 0, cls: mode === "eager" ? "text-warn" : "text-accent" },
    { label: "median gap, µs", value: trace[mode === "eager" ? "eager" : "graphs"].gap_us.p50, digits: 2, cls: mode === "eager" ? "text-warn" : "text-accent" },
    { label: "GPU idle, % of step", value: mode === "eager" ? eager.idle_pct : replay.idle_pct, digits: 1, cls: mode === "eager" ? "text-warn" : "text-accent", suffix: "%" },
  ];
  const e = tpot("engine", batch);
  const g = tpot("engine + graphs", batch);
  return (
    <div className="flex h-full flex-col gap-2 font-mono text-xs">
      <div className="flex flex-wrap gap-x-6 gap-y-2">
        <Toggle label="step" choices={["eager", "graph"]} value={mode} onChange={setMode} />
        <Toggle label="batch" choices={["1", "4", "16"]} value={batch} onChange={setBatch} />
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={`Layer ${layerIndex} of a batch-1 decode step, ${mode === "eager" ? "launched kernel by kernel" : "replayed from a CUDA graph"}: ${Math.round(end)} µs`}>
        <text x={x0} y={16} className="fill-muted text-[11px]">{`layer ${layerIndex} of 22, the same ten kernels, from the two profiles`}</text>
        {ks.map((k, i) => {
          const w = Math.max(1.5, px(eagerLayer[i].end - eagerLayer[i].start));
          return (
            <motion.rect
              key={i}
              x={0}
              y={40}
              width={w}
              height={26}
              rx={2}
              className={gemm.has(k.op) ? "fill-baseline" : "fill-accent"}
              style={{ transformBox: "fill-box", originX: 0 }}
              initial={false}
              animate={{ x: x0 + px(k.start), scaleX: Math.max(1.5, px(k.end - k.start)) / w }}
              transition={ease}
            />
          );
        })}
        {eagerLayer.slice(0, -1).map((k, i) => (
          <motion.rect
            key={`g${i}`}
            x={x0 + px(k.end)}
            y={70}
            width={Math.max(1.5, px(eagerLayer[i + 1].start - k.end))}
            height={5}
            className="fill-warn"
            initial={false}
            animate={{ opacity: mode === "eager" ? 1 : 0 }}
            transition={ease}
          />
        ))}
        <motion.g initial={false} animate={{ x: x0 + px(end) }} transition={ease}>
          <line x1={0} x2={0} y1={32} y2={84} className={mode === "eager" ? "stroke-warn" : "stroke-accent"} strokeWidth={1.25} />
        </motion.g>
        <line x1={x0 + px(layerUs(graphLayer))} x2={x0 + px(layerUs(graphLayer))} y1={88} y2={96} className="stroke-accent" />
        <line x1={x0 + px(layerUs(eagerLayer))} x2={x0 + px(layerUs(eagerLayer))} y1={88} y2={96} className="stroke-warn" />
        <text x={x0 + px(layerUs(graphLayer)) - 4} y={110} textAnchor="end" className="fill-accent text-[10px]">
          graph
        </text>
        <text x={x0 + px(layerUs(eagerLayer)) + 4} y={110} className="fill-warn text-[10px]">
          eager
        </text>
        {[0, 100, 200, 300].map((t) => (
          <text key={t} x={x0 + px(t)} y={136} textAnchor={t === 0 ? undefined : "middle"} className="fill-muted text-[10px]">
            {`${t} µs`}
          </text>
        ))}
      </svg>
      <div className="grid grid-cols-3 gap-x-4 border-t border-border pt-2 tabular-nums">
        {stats.map((r) => (
          <div key={r.label}>
            <div className="text-muted">{r.label}</div>
            <div className={r.cls}>
              <NumberFlow value={r.value} suffix={r.suffix} format={{ minimumFractionDigits: r.digits, maximumFractionDigits: r.digits }} />
            </div>
          </div>
        ))}
      </div>
      <div className="mt-auto grid grid-cols-3 gap-x-4 border-t border-border pt-2 tabular-nums">
        {[
          { label: `batch ${batch} eager, ms`, value: e, cls: "text-text" },
          { label: `batch ${batch} graphs, ms`, value: g, cls: "text-accent" },
          { label: "saved per token, ms", value: e - g, cls: "text-accent" },
        ].map((r) => (
          <div key={r.label}>
            <div className="text-muted">{r.label}</div>
            <div className={r.cls}>
              <NumberFlow value={r.value} format={{ minimumFractionDigits: 2, maximumFractionDigits: 2 }} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

const fmt = (v: number, d = 0) => v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
const saved = ["1", "4", "16"].map((b) => tpot("engine", b) - tpot("engine + graphs", b));
const savedShare = Math.round((100 * saved[0]) / tpot("engine", "1"));
const layersAhead = Math.round(trace.eager.cpu_ahead_us / layerUs(eagerLayer));
const shortKernels = eagerLayer.filter((k) => k.end - k.start < 3).length;

const steps: StoryStep[] = [
  {
    id: "launches",
    title: "one launch per kernel",
    body: (
      <>
        <p>
          A decode step of TinyLlama is {eager.kernels} kernels: ten per layer for 22 layers, plus the embedding, the last
          norm and the LM head. Run eagerly, the CPU hands each one to the GPU with its own cudaLaunchKernel, which takes{" "}
          <span className="font-mono text-text">{`${fmt(trace.eager.launches.cudaLaunchKernel.median_us)} µs`}</span> on the
          CPU under the profiler. The launches queue up on a stream and the GPU runs them in order.
        </p>
        <p>
          The CPU keeps up easily: in the middle of a step it is{" "}
          <span className="font-mono text-text">{`${fmt(trace.eager.cpu_ahead_us / 1000, 1)} ms`}</span> ahead, launching
          kernels {layersAhead} layers further on. And still the GPU is idle for{" "}
          <span className="font-mono text-warn">{`${eager.idle_pct}%`}</span> of the step.
        </p>
      </>
    ),
  },
  {
    id: "gaps",
    title: "the gap between kernels",
    body: (
      <>
        <p>
          Zoomed in, the idle time is a few microseconds before every kernel. Between two kernels launched one by one on a
          stream, the GPU lets the first finish completely, then takes the next launch from the queue and sets up its grid
          before its first block runs. The median gap is{" "}
          <span className="font-mono text-warn">{`${trace.eager.gap_us.p50} µs`}</span>, whatever the CPU is doing.
        </p>
        <p>
          A few microseconds is nothing next to a 140 µs gate/up GEMV, but a step has {eager.kernels - 1} of these
          boundaries, and {shortKernels} of every ten kernels run for under 3 µs. Together they come to{" "}
          <span className="font-mono text-warn">{`${fmt(eager.span_us - eager.busy_us)} µs`}</span> per step. Faster kernels
          would not touch it; fewer launches would.
        </p>
      </>
    ),
  },
  {
    id: "capture",
    title: "record once",
    body: (
      <>
        <p>
          A CUDA Graph is the whole sequence of launches recorded once and handed to the GPU as one unit. DecodeGraphs
          records it with stream capture: between cudaStreamBeginCapture and cudaStreamEndCapture, the kernels that
          run_staged launches are not run but added to a graph, each one after the one before it.
        </p>
        <p>
          Instantiating the graph checks it and prepares it for launch, once, at startup. From then on a decode step is
          one cudaGraphLaunch. Capture has rules: nothing may allocate memory or wait for the GPU, which is why every
          graph is captured right after one eager run of the same step, when cuBLAS has already picked and loaded its
          kernels; its workspace is memory the engine gave it up front.
        </p>
      </>
    ),
  },
  {
    id: "stage",
    title: "same addresses, new values",
    body: (
      <>
        <p>
          A graph stores each kernel with the arguments it was captured with: the same pointers, the same grid. A replay
          cannot be told &quot;these are this step&apos;s tokens&quot;. So the step&apos;s inputs go where the graph already
          looks: stage copies the token ids, positions, KV slots, context lengths and block tables into one buffer of the
          runner, outside the graph, and the captured kernels read them there.
        </p>
        <p>
          The five arrays sit at offsets that depend on the number of rows, and so does every GEMM&apos;s shape. A batch of
          3 is padded to the 4-row bucket with rows that write nowhere and attend to nothing, so it replays the 4-row graph.
          Five buckets, 1 to 16 rows, cover every decode batch.
        </p>
      </>
    ),
  },
  {
    id: "splits",
    title: "one graph per split count",
    body: (
      <>
        <p>
          The frozen grid caught up with the engine once. Split attention cuts a long context across 2, 4 or 8 thread blocks
          per head, and the split count is part of the attention kernel&apos;s grid. The graphs were captured on an empty
          dummy batch, so every replay ran the one-split kernel: at a long prompt eager became faster than the graphs.
        </p>
        <p>
          Now the constructor walks the contexts up to 2,048 tokens and captures again whenever a bucket&apos;s split count
          changes: 11 graphs instead of 5. A replay picks its graph from the batch&apos;s longest context, and at a
          1,984-token prompt the time per token goes from{" "}
          <span className="font-mono text-text">{`${longRun.before.toFixed(1)}`}</span> to{" "}
          <span className="font-mono text-accent">{`${longRun.after.toFixed(1)} ms`}</span>.
        </p>
      </>
    ),
  },
  {
    id: "replay",
    title: "the replay",
    body: (
      <>
        <p>
          Switch the step on the right: the same ten kernels, from the two profiles, close up. The median gap drops from{" "}
          {trace.eager.gap_us.p50} to <span className="font-mono text-accent">{`${trace.graphs.gap_us.p50} µs`}</span>,
          and the GPU is idle for {replay.idle_pct}% of the step instead of {eager.idle_pct}%. The kernels themselves take
          the same time.
        </p>
        <p>
          Without the profiler the saving is{" "}
          <span className="font-mono text-accent">{`${saved.map((s) => s.toFixed(2)).join(", ")} ms`}</span> per token at
          batch 1, 4 and 16: roughly a fixed amount per step, since every step has the same {eager.kernels} boundaries. It
          is {savedShare}% of a batch-1 token and less of a longer batch-16 step. Prefill steps are not graphed.
        </p>
      </>
    ),
  },
];

const captions = [
  `one decode step from the batch-1 Nsight Systems profile behind results/decode_step.md (commit ${trace.commit}): real start and end times, layer ${layerIndex} drawn`,
  "gap = start of a kernel minus the end of the one before it, over every decode step of both profiles (web/scripts/step-timeline.py)",
  "one layer of the 224 nodes drawn; the code is capture() in src/runtime/decode_graphs.cpp",
  "the token ids, positions and slots are made up; the layout is PagedRunner::stage's",
  "graph count and split rule from docs/profiling.md, Split attention in CUDA Graphs; ms per token from engine bench, mean of two runs",
  "layer from the two profiles; idle share from results/decode_step*.md; ms per token from engine bench in docs/results.md (TPOT p50, 3 runs)",
];

export function CudaGraphsStory() {
  return (
    <StickyStory
      steps={steps}
      caption={(i) => captions[i]}
      figure={(i) => {
        switch (i) {
          case 0:
            return <EagerTimelineFigure idlePct={eager.idle_pct} kernels={eager.kernels} spanUs={eager.span_us} busyUs={eager.busy_us} />;
          case 1:
            return <GapFigure boundaries={eager.kernels - 1} gapsOver={eager.gaps} idleUs={eager.span_us - eager.busy_us} />;
          case 2:
            return <CaptureFigure kernels={replay.kernels} />;
          case 3:
            return <StageFigure />;
          case 4:
            return <GraphGridFigure {...longRun} />;
          default:
            return <ReplayPicker />;
        }
      }}
    />
  );
}
