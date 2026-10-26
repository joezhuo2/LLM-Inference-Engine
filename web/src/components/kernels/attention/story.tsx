"use client";

import NumberFlow from "@number-flow/react";
import clsx from "clsx";
import { useState } from "react";
import { measured } from "@/content/kernels/types";
import { traces } from "@/lib/data";
import { StickyStory, type StoryStep } from "../sticky-story";
import { Toggle } from "../toggle";
import { BlocksFigure, IdleFigure, LanesFigure, MergeFigure, OnlineSoftmaxFigure, OverviewFigure, SplitFigure, V1Figure } from "./figures";

const split = "split-context-attention-branch-40";
const v2Section = "paged-attention-v2-branch-37";
const at2048 = { rows: "1", context: "2,048" };
const v1At2048 = measured(v2Section, at2048, "v1 (us)");
const v2At2048 = measured(v2Section, at2048, "v2 (us)");
const v2Loop = measured(split, at2048, "v2");
const split8 = measured(split, at2048, "8");
const warps = (() => {
  const v = measured(v2Section, { "rows x context": "1 x 2,048", kernel: "v2" }, "warps active", 1);
  return `${v}%`;
})();

const poolBlocks = Number((traces.find((t) => t.meta.prompts && t.meta.prompts !== "random") ?? traces[0])?.meta.kv_blocks);

function attentionSplits(rows: number, context: number, heads = 32) {
  let s = 1;
  while (s < 8 && 2 * s * rows * heads <= 256 && context >= 2 * s * 256) s *= 2;
  return s;
}

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
            <div className="h-3 min-w-0 rounded-sm bg-border">
              <div
                className={clsx("bar-ease h-full origin-left rounded-sm", t.column === pickedColumn ? "bg-accent" : "bg-baseline/70")}
                style={{ transform: `scaleX(${t.us / max})` }}
              />
            </div>
            <span className="text-right tabular-nums text-muted">
              <NumberFlow value={t.us} suffix=" µs" format={{ minimumFractionDigits: 1, maximumFractionDigits: 1 }} />
            </span>
          </div>
        ))}
      </div>
      <div className="space-y-1 border-t border-border pt-2 leading-relaxed">
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
    id: "job",
    title: "the job",
    body: (
      <>
        <p>
          Each decode step adds one token to every sequence. Attention compares that token&apos;s query with the key of
          every token before it, turns the scores into weights with a softmax, and sums the values with those weights.
        </p>
        <p>32 query heads, 64 dimensions each, against a context that grows by one token every step.</p>
      </>
    ),
  },
  {
    id: "blocks",
    title: "keys live in blocks",
    body: (
      <>
        <p>
          The KV cache is one pool of 16-token blocks shared by every request. A sequence owns a list of blocks, its
          block table, and they can sit anywhere in the pool.
        </p>
        <p>So the kernel cannot read a context as one contiguous array: every key goes through the table first.</p>
      </>
    ),
  },
  {
    id: "v1",
    title: "v1: every score in shared memory",
    body: (
      <>
        <p>
          One thread block per row and head. It writes a score for every token into shared memory, finds the maximum,
          takes the exponentials and their sum, then multiplies by the values: three passes.
        </p>
        <p>
          Shared memory has a fixed size, so the context does too: v1 stops at 2,048 tokens. One row at 2,048 tokens:{" "}
          <span className="font-mono text-text">{v1At2048} µs</span>.
        </p>
      </>
    ),
  },
  {
    id: "v2",
    title: "v2: one chunk at a time",
    body: (
      <>
        <p>
          The online softmax keeps three things between chunks: the largest score so far, the sum of exponentials, and
          the weighted values. When a chunk brings a larger score, the old sum and values are scaled down by
          exp(m_old − m_new) and the new chunk is added.
        </p>
        <p>
          The answer is the same as the three-pass version, but only one 256-token chunk is ever in shared memory, so
          contexts of 8,192 tokens run.
        </p>
      </>
    ),
  },
  {
    id: "lanes",
    title: "eight lanes per key",
    body: (
      <>
        <p>
          In v1 each thread computed a whole dot product alone. In v2 a group of 8 lanes shares one key: at every step
          they read 8 neighbouring dimensions together, one 16-byte read, and three shuffles add their 8 partial sums.
        </p>
        <p>
          With the softmax change and every thread helping with the values, one row at 2,048 tokens went from{" "}
          <span className="font-mono text-text">{v1At2048}</span> to{" "}
          <span className="font-mono text-accent">{v2At2048} µs</span>.
        </p>
      </>
    ),
  },
  {
    id: "idle",
    title: "one row leaves the GPU idle",
    body: (
      <>
        <p>
          v2 launches one block per row and head. With a single sequence that is 32 blocks, each walking its whole
          context alone, while the GPU could keep many more warps in flight.
        </p>
        <p>
          Nsight Compute measured <span className="font-mono text-text">{warps}</span> of warps active and DRAM
          throughput under 2%: the kernel waits on latency, not bandwidth.
        </p>
      </>
    ),
  },
  {
    id: "split",
    title: "split the context",
    body: (
      <>
        <p>
          The split kernel cuts each context into up to 8 ranges on block boundaries. Each range is its own thread block
          running the v2 loop and writing a partial result: its maximum, its sum and its 64 weighted values.
        </p>
        <p>
          One row at 2,048 tokens: <span className="font-mono text-text">{v2Loop}</span> to{" "}
          <span className="font-mono text-accent">{split8} µs</span> with 8 splits, merge included.
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
          A second, small kernel finds the largest of the partial maxima, rescales every partial result to it, and adds
          them up in split order.
        </p>
        <p>Because the order never changes, the same input gives the same bits on every run, which the tests check.</p>
      </>
    ),
  },
  {
    id: "choose",
    title: "how many splits?",
    body: (
      <>
        <p>
          Splitting only pays when there are few rows and a long context. With 16 rows there are already 512 blocks, and
          extra splits add work. Try the measured times on the right.
        </p>
        <p>
          The engine&apos;s rule doubles the split count while the grid stays at or under 256 blocks, every split keeps at
          least 256 tokens, and there are at most 8. CUDA Graphs capture one graph per split count.
        </p>
      </>
    ),
  },
];

const captions = [
  "decode attention for one sequence and one head; the bars are illustrative",
  Number.isFinite(poolBlocks) && poolBlocks > 0
    ? `example block table; the pool held ${poolBlocks.toLocaleString("en-US")} blocks in the recorded run`
    : "example block table",
  `one cell of the context = 64 tokens; ${v1At2048} µs for one row at 2,048 tokens`,
  "example scores; m only ever rises, and every rise rescales what came before",
  "lane j reads dims j, j + 8, j + 16 … so each step's 8 reads are side by side in memory",
  `warps active ${warps} is measured (Nsight Compute, 1 row × 2,048); dots are an illustration, not slots to scale`,
  `${v2Loop} µs → ${split8} µs, one row at 2,048 tokens, from docs/profiling.md`,
  "example values; the same arithmetic as one pass over the whole context",
  "µs per call, measured on the RTX 5060 Laptop GPU; the engine never picks 16",
];

export function PagedAttentionStory() {
  return (
    <StickyStory
      steps={steps}
      caption={(i) => captions[i]}
      figure={(i) => {
        switch (i) {
          case 0:
            return <OverviewFigure />;
          case 1:
            return <BlocksFigure />;
          case 2:
            return <V1Figure />;
          case 3:
            return <OnlineSoftmaxFigure />;
          case 4:
            return <LanesFigure />;
          case 5:
            return <IdleFigure v2={v2Loop} warps={warps} />;
          case 6:
            return <SplitFigure v2={v2Loop} split={split8} />;
          case 7:
            return <MergeFigure />;
          default:
            return <SplitPicker />;
        }
      }}
    />
  );
}
