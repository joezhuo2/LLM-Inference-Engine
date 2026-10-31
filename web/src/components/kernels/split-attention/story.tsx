"use client";

import NumberFlow from "@number-flow/react";
import { useState } from "react";
import { measured } from "@/content/kernels/types";
import { EasedBar } from "../eased-bar";
import { StickyStory, type StoryStep } from "../sticky-story";
import { Toggle } from "../toggle";
import { CutFigure, type GraphKey, GraphsFigure, MergeFigure, type Occupancy, OccupancyFigure, ScratchFigure, WaitingFigure, WaveFigure } from "./figures";

const split = "split-context-attention-branch-40";
const wired = "split-attention-in-the-forward-pass-branch-42";
const graphs = "split-attention-in-cuda-graphs-branch-43";
const occupancy = "split-attention-occupancy-branch-66";
const v2Section = "paged-attention-v2-branch-37";

const perSm = 6;
const headDim = 64;
const maxBlocks = 256;

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
  while (s < 8 && 2 * s * rows * heads <= maxBlocks && context >= 2 * s * 256) s *= 2;
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

function SplitPicker() {
  const [rows, setRows] = useState("1");
  const [context, setContext] = useState("2,048");
  const times = options.map((o) => ({ ...o, us: measured(split, { rows, context }, o.column) }));
  const max = Math.max(...times.map((t) => t.us));
  const best = times.reduce((a, b) => (b.us < a.us ? b : a));
  const picked = attentionSplits(Number(rows), Number(context.replace(",", "")));
  const pickedColumn = picked === 1 ? "v2" : String(picked);
  return (
    <div className="flex h-full flex-col gap-3 font-mono text-xs">
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
          <span className="text-muted">{` for ${perSm * 26} slots`}</span>
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
          holds at most {perSm} of them at once (registers and warps both stop it there), so 26 SMs have {perSm * 26}{" "}
          slots.
        </p>
        <p>
          {v2.blocks} of {perSm * 26} is <span className="font-mono text-text">{v2.waves}</span> of a wave. Every block
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
          <span className="font-mono text-text">{dram}%</span> of its peak. The kernel is not short of bandwidth.
        </p>
        <p>
          It is short of warps. Each key load waits hundreds of cycles, and a GPU hides that wait by running other warps
          in the meantime. With <span className="font-mono text-text">{v2.warps}%</span> of warp slots filled, there are
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
        <p>One head&apos;s 2,048 tokens become 8 ranges of 256, worked on at the same time.</p>
      </>
    ),
  },
  {
    id: "wave",
    title: "fill one wave",
    body: (
      <>
        <p>
          Each doubling of the split count doubles the grid. Up to 4 splits (128 blocks) everything still fits in one
          wave, and the time halves each time: {waves.map((w) => w.us.toFixed(1)).join(", ")} µs for 1, 2, 4 and 8 splits.
        </p>
        <p>
          At 8 splits the grid is {waves[3].waves} waves. The first {perSm * 26} blocks run together and the last{" "}
          {waves[3].blocks - perSm * 26} wait for a second, mostly empty wave, so each block has half the work and the
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
          scratch buffer: its largest score m, its sum of exponentials l, and its 64 weighted values, not yet divided.
        </p>
        <p>
          The buffer was a new argument to the kernel, which is why split attention needed its own signature. The runner
          owns {scratchBytes.toLocaleString("en-US")} bytes of it.
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
          input gives the same bits on every run. One row at 2,048 tokens: <span className="font-mono text-text">{loopV2}</span>{" "}
          to <span className="font-mono text-accent">{loop8} µs</span> per call, merge included.
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
          Wired into the decode step, splitting took a 1,984-token prompt from {graphBars[0].ms} to {graphBars[1].ms} ms
          per token in eager mode. With CUDA Graphs nothing moved: each graph had been captured on a dummy batch with no
          context, where the rule says 1 split, so every replay launched v2&apos;s grid.
        </p>
        <p>
          The fix captures one graph per batch size and split count, {graphKeys.reduce((a, k) => a + k.splits.length, 0)}{" "}
          instead of 5, and replays the one that matches the batch: <span className="font-mono text-accent">{graphBars[3].ms} ms</span>.
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
          With 16 rows v2 already launches 512 blocks, more than three waves, and splitting only adds the partial writes
          and the merge. At 128 tokens a split holds too few keys to pay for its second launch.
        </p>
        <p>
          So the rule doubles the split count only while the grid stays at or under 256 blocks and every split keeps at
          least 256 tokens, up to 8. Try the measured times on the right.
        </p>
      </>
    ),
  },
];

const captions = [
  `${perSm} blocks per SM and ${v2.warps}% warps active are measured (Nsight Compute, 1 row × 2,048 tokens)`,
  `from the v2 Nsight Compute table in docs/profiling.md; ${smThroughput}% SM throughput`,
  "one row, one head; the other 31 heads are cut the same way",
  "block counts, waves, warps and times measured with Nsight Compute; slot placement is a sketch",
  "the layout of the scratch buffer; the record contents are illustrative",
  "example values; the same arithmetic as one pass over the whole context",
  "TPOT p50 of the first of two runs, RTX 5060 Laptop GPU, 64 output tokens",
  "µs per call from the bench loop; the engine never picks 16",
];

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
