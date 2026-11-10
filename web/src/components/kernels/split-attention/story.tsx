"use client";

import NumberFlow from "@number-flow/react";
import { useState } from "react";
import { measured } from "@/content/kernels/types";
import { modelShape, onchip } from "@/lib/data";
import { EasedBar } from "../eased-bar";
import { StickyStory, type StoryStep } from "../sticky-story";
import { Toggle } from "../toggle";
import { CutFigure, type GraphKey, GraphsFigure, MergeFigure, type Occupancy, OccupancyFigure, ScratchFigure, WaitingFigure, WaveFigure } from "./figures";
import { expectAnswer, type Prediction, Predict } from "../predict";

const split = "split-context-attention-branch-40";
const wired = "split-attention-in-the-forward-pass-branch-42";
const graphs = "split-attention-in-cuda-graphs-branch-43";
const occupancy = "split-attention-occupancy-branch-66";
const v2Section = "paged-attention-v2-branch-37";

const perSm = 6;
const sms = 26;
const slots = perSm * sms;
const headDim = 64;
const maxBlocks = 256;
const minTokens = 256;
const maxSplits = 8;
const fullRows = 16;
const fullBlocks = fullRows * modelShape().heads;
const cutContext = 2048;
const cutSplits = 8;

function occ(kernel: string, splits: number): Occupancy {
  const m = { kernel, splits: String(splits) };
  return {
    splits,
    blocks: measured(occupancy, m, "grid blocks"),
    waves: measured(occupancy, m, "waves per SM"),
    warps: measured(occupancy, m, "warps active"),
    us: measured(occupancy, m, "time (us)"),
  };
}

const v2 = occ("v2", 1);
const waves = [1, 2, 4, 8].map((s) => occ("split partial", s));
const dramCycles = Math.round(onchip("dram_latency", "cycles").median).toLocaleString("en-US");
const l2Cycles = Math.round(onchip("l2_latency", "cycles").median);
const v2Nsight = { "rows x context": "1 x 2,048", kernel: "v2" };
const smThroughput = measured(v2Section, v2Nsight, "SM throughput", 1);
const dram = measured(v2Section, v2Nsight, "DRAM throughput", 1);

const at2048 = { rows: "1", context: "2,048" };
const loopV2 = measured(split, at2048, "v2");
const loop8 = measured(split, at2048, "8");

const longPrompt = { "max batch": "1", prompt: "1,984" };
const firstRun = (section: string, column: string) => measured(section, longPrompt, column);
const graphBars = [
  { label: "v2, eager", ms: firstRun(wired, "v2 eager") },
  { label: "split, eager", ms: firstRun(wired, "split eager") },
  { label: "split, old graphs", ms: firstRun(graphs, "main graphs") },
  { label: "one graph per split", ms: firstRun(graphs, "this branch graphs"), accent: true },
];

export function attentionSplits(rows: number, context: number, heads = 32) {
  let s = 1;
  while (s < maxSplits && 2 * s * rows * heads <= maxBlocks && context >= 2 * s * minTokens) s *= 2;
  return s;
}

const graphKeys: GraphKey[] = [1, 2, 4, 8, 16].map((bucket) => ({
  bucket,
  splits: [...new Set(Array.from({ length: 2049 }, (_, c) => attentionSplits(bucket, c)))],
}));

const scratchBytes = maxBlocks * (headDim + 2) * 4;

const rowsChoices = ["1", "16"];
const contextChoices = ["128", "512", "2,048", "8,192"];
const options = [
  { column: "v2", label: "v2" },
  { column: "1 split", label: "1 split" },
  { column: "2", label: "2 splits" },
  { column: "4", label: "4 splits" },
  { column: "8", label: "8 splits" },
  { column: "16", label: "16 splits" },
];

const v2At128 = measured(split, { rows: "1", context: "128" }, "v2");
const eightAt128 = measured(split, { rows: "1", context: "128" }, "8");
const prediction: Prediction = expectAnswer({
  question: "One row at only 128 tokens: do 8 splits beat v2?",
  choices: ["yes", "no"],
  answer: eightAt128 < v2At128 ? "yes" : "no",
  why: `8 splits take ${eightAt128.toFixed(1)} µs, v2 ${v2At128.toFixed(1)} µs. Each split gets only 16 keys, too few to pay for the partial results and the second launch of the merge. Set the context to 128 to see it.`,
}, "no");

export function SplitPicker() {
  const [rows, setRows] = useState("1");
  const [context, setContext] = useState("2,048");
  const times = options.map((o) => ({ ...o, us: measured(split, { rows, context }, o.column) }));
  const max = Math.max(...times.map((t) => t.us));
  const best = times.reduce((a, b) => (b.us < a.us ? b : a));
  const picked = attentionSplits(Number(rows), Number(context.replace(",", "")));
  const pickedColumn = picked === 1 ? "v2" : String(picked);
  return (
    <div className="flex h-full flex-col gap-3 tabular-nums text-xs">
      <Predict {...prediction} />
      <div className="flex flex-wrap gap-x-5 gap-y-2">
        <Toggle label="rows" choices={rowsChoices} value={rows} onChange={setRows} />
        <Toggle label="context" choices={contextChoices} value={context} onChange={setContext} />
      </div>
      <div className="flex-1 space-y-1.5 pt-1">
        {times.map((t) => (
          <div key={t.column} className="grid grid-cols-[4.5rem_1fr_4.5rem] items-center gap-2">
            <span className={t.column === pickedColumn ? "text-accent" : "text-muted"}>{t.label}</span>
            <div className="h-3 min-w-0 rounded-full bg-border">
              <EasedBar fraction={t.us / max} className={t.column === pickedColumn ? "bg-accent" : "bg-baseline/70"} />
            </div>
            <span className="text-right tabular-nums text-muted">
              <NumberFlow value={t.us} suffix=" µs" format={{ minimumFractionDigits: 1, maximumFractionDigits: 1 }} />
            </span>
          </div>
        ))}
      </div>
      <div className="space-y-1 border-t border-border pt-2 leading-relaxed">
        <div>
          <span className="text-muted">grid </span>
          <span className="text-text">{`${Number(rows) * 32 * picked} blocks`}</span>
          <span className="text-muted">{` for ${slots} slots`}</span>
        </div>
        <div>
          <span className="text-muted">engine picks </span>
          <span className="text-accent">{picked === 1 ? "v2 (no split)" : `${picked} splits`}</span>
        </div>
        <div>
          <span className="text-muted">fastest measured </span>
          <span className="text-text">{best.label}</span>
          <span className="text-muted">{` (${best.us.toFixed(1)} µs)`}</span>
        </div>
      </div>
    </div>
  );
}

const steps: StoryStep[] = [
  {
    id: "few",
    title: "one block per head",
    body: (
      <>
        <p>
          v2 launches one thread block per row and query head. At batch 1 that is {v2.blocks} blocks. Each SM of this GPU
          holds only a few of them at once (registers and warps both stop it there), so the SMs together have a fixed
          number of slots.
        </p>
        <p>
          That fills only <span className="text-accent">{v2.waves}</span> of a wave. Every block
          walks its head&apos;s whole context alone, and most of the GPU has nothing to do.
        </p>
      </>
    ),
  },
  {
    id: "waiting",
    title: "waiting, not reading",
    body: (
      <>
        <p>
          A long context sounds like a lot of memory to read, but Nsight Compute shows DRAM at{" "}
          <span className="text-text">{dram}%</span> of its peak. The kernel is not short of bandwidth.
        </p>
        <p>
          It is short of warps. A load that has to go to DRAM waits several hundred cycles on this GPU, much longer than one
          that hits L2, and a GPU hides that wait by running other warps in the meantime. With <span className="text-accent">{v2.warps}%</span> of warp slots filled, there are
          too few to switch to.
        </p>
      </>
    ),
  },
  {
    id: "cut",
    title: "cut the context",
    body: (
      <>
        <p>
          The split kernel adds a grid dimension: blocks are (split, head, row). Each split takes a range of whole KV
          blocks, so no 16-token block is shared by two ranges, and runs the same online-softmax loop as v2 over it.
        </p>
        <p>One head&apos;s long context becomes <span className="text-accent">8</span> ranges, worked on at the same time.</p>
      </>
    ),
  },
  {
    id: "wave",
    title: "fill one wave",
    body: (
      <>
        <p>
          Each doubling of the split count doubles the grid. While the whole grid fits in one wave, the time halves each
          time.
        </p>
        <p>
          At the last doubling the grid is <span className="text-accent">{waves[3].waves}</span> waves. The first wave&apos;s worth of blocks run together and the rest wait for a second, mostly empty wave, so each block has half the work and the
          time barely moves.
        </p>
      </>
    ),
  },
  {
    id: "scratch",
    title: "partial results",
    body: (
      <>
        <p>
          A split cannot write the output, because it only saw part of the context. It writes three things to a float
          scratch buffer: its largest score m, its sum of exponentials l, and its {headDim} weighted values, not yet divided.
        </p>
        <p>
          The buffer was a new argument to the kernel, which is why split attention needed its own signature. The runner
          owns <span className="text-accent">{scratchBytes.toLocaleString("en-US")}</span> bytes of it.
        </p>
      </>
    ),
  },
  {
    id: "merge",
    title: "merge in a fixed order",
    body: (
      <>
        <p>
          A second, small kernel, one block per row and head, finds the largest partial maximum, rescales every partial
          result to it and adds them up in split order.
        </p>
        <p>
          Splits finish in whatever order the GPU runs them, but the merge always adds them the same way, so the same
          input gives the same bits on every run. One row at a long context now takes <span className="text-accent">{loop8} µs</span> per call, merge included.
        </p>
      </>
    ),
  },
  {
    id: "graphs",
    title: "the graphs still ran v2",
    body: (
      <>
        <p>
          Wired into the decode step, splitting made a long prompt faster per token in eager mode. With CUDA Graphs nothing moved: each graph had been captured on a dummy batch with no
          context, where the rule says one split, so every replay launched v2&apos;s grid.
        </p>
        <p>
          The fix captures one graph per batch size and split count, more graphs than before, and replays the one that
          matches the batch: <span className="text-accent">{graphBars[3].ms} ms</span>.
        </p>
      </>
    ),
  },
  {
    id: "choose",
    title: "when splitting loses",
    body: (
      <>
        <p>
          With many rows v2 already launches <span className="text-accent">{fullBlocks}</span> blocks, enough to fill the GPU several times over, and splitting only adds the partial writes
          and the merge. At a very short context a split holds too few keys to pay for its second launch.
        </p>
        <p>
          So the rule doubles the split count only while the grid stays at or under {maxBlocks} blocks and each split keeps
          enough tokens to pay for itself. Try the measured times on the right.
        </p>
      </>
    ),
  },
];

const captions = [
  `${perSm} blocks per SM and ${v2.warps}% warps active are measured (Nsight Compute, 1 row × 2,048 tokens); ${sms} SMs hold ${slots} blocks at once, calculated`,
  `from the v2 Nsight Compute table in docs/profiling.md; ${smThroughput}% SM throughput; a DRAM load waits about ${dramCycles} cycles and an L2 hit ${l2Cycles}, both measured with a pointer chase (scripts/onchip_bench.cu)`,
  `one row, one head: ${cutContext.toLocaleString("en-US")} tokens become ${cutSplits} ranges of ${cutContext / cutSplits}; the other ${modelShape().heads - 1} heads are cut the same way`,
  `measured with Nsight Compute: ${waves.map((w) => w.us.toFixed(1)).join(", ")} µs for ${waves.map((w) => w.splits).join(", ")} splits; up to ${waves[2].splits} splits (${waves[2].blocks} blocks) the grid fits one wave; at ${waves[3].splits} splits ${slots} blocks run first and ${waves[3].blocks - slots} wait; slot placement is a sketch`,
  "the layout of the scratch buffer; the record contents are illustrative",
  `${loopV2} µs → ${loop8} µs per call, one row at 2,048 tokens, v2 against 8 splits, merge included; example values; the same arithmetic as one pass over the whole context`,
  `TPOT p50 of the first of two runs, RTX 5060 Laptop GPU, 64 output tokens; eager at a 1,984-token prompt: v2 ${graphBars[0].ms} → split ${graphBars[1].ms} ms per token; ${graphKeys.reduce((a, k) => a + k.splits.length, 0)} graphs instead of ${graphKeys.length} for the batch sizes ${graphKeys.map((k) => k.bucket).join(", ")}, measured with those sizes; the CUDA Graphs chapter adds batch sizes in between`,
  `µs per call from the bench loop; the engine never picks 16; ${fullRows} rows × ${modelShape().heads} heads = ${fullBlocks} blocks against ${slots} slots; the short context is 128 tokens; the rule needs at least ${minTokens} tokens per split, at most ${maxBlocks} blocks and ${maxSplits} splits`,
];

export const splitSteps = steps;
export const splitCaptions = captions;
export const splitNumbers = { perSm, sms, headDim, v2, waves, dram, smThroughput, dramCycles, l2Cycles, loopV2, loop8, graphBars, graphKeys, scratchBytes, replaySplits: attentionSplits(1, 1984) };
export type SplitNumbers = typeof splitNumbers;

export function SplitAttentionStory() {
  return (
    <StickyStory
      steps={steps}
      caption={(i) => captions[i]}
      figure={(i) => {
        switch (i) {
          case 0:
            return <OccupancyFigure perSm={perSm} v2={v2} />;
          case 1:
            return <WaitingFigure warps={v2.warps} sm={smThroughput} dram={dram} />;
          case 2:
            return <CutFigure splits={8} contextBlocks={128} />;
          case 3:
            return <WaveFigure perSm={perSm} rows={waves} />;
          case 4:
            return <ScratchFigure splits={8} bytes={scratchBytes} />;
          case 5:
            return <MergeFigure />;
          case 6:
            return <GraphsFigure bars={graphBars} keys={graphKeys} />;
          default:
            return <SplitPicker />;
        }
      }}
    />
  );
}
