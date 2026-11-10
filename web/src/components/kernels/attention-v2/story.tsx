"use client";

import NumberFlow from "@number-flow/react";
import Link from "next/link";
import { useState } from "react";
import { measured } from "@/content/kernels/types";
import { profilingTable } from "@/lib/data";
import { EasedBar } from "../eased-bar";
import { StickyStory, type StoryStep } from "../sticky-story";
import { Toggle } from "../toggle";
import { LanesFigure, OnlineSoftmaxFigure, type Pair, ProfileFigure, type SharedBytes, SharedMemoryFigure, StepFigure, ThreePassFigure } from "./figures";
import { expectAnswer, type Prediction, Predict } from "../predict";

const v2Section = "paged-attention-v2-branch-37";
const wired = "paged-attention-v2-in-the-forward-pass-branch-38";
const shared = "attention-shared-memory-branch-67";

const at2048 = { rows: "1", context: "2,048" };
const v1At2048 = measured(v2Section, at2048, "v1 (us)");
const v2At2048 = measured(v2Section, at2048, "v2 (us)");

const ncu = (shape: string, kernel: string, column: string) => measured(v2Section, { "rows x context": shape, kernel }, column, 1);
const profilePairs: Pair[] = [
  { label: "duration, 1 row × 2,048", v1: ncu("1 x 2,048", "v1", "duration (us)"), v2: ncu("1 x 2,048", "v2", "duration (us)"), unit: "µs" },
  { label: "warps active, 16 rows × 512", v1: ncu("16 x 512", "v1", "warps active"), v2: ncu("16 x 512", "v2", "warps active"), unit: "%" },
  { label: "SM throughput, 16 rows × 512", v1: ncu("16 x 512", "v1", "SM throughput"), v2: ncu("16 x 512", "v2", "SM throughput"), unit: "%" },
];
const dram = Math.max(...["v1", "v2"].flatMap((k) => ["1 x 2,048", "16 x 512"].map((s) => ncu(s, k, "DRAM throughput"))));

const stepRow = (label: string) => ({ "": label });
const attentionRow = "attention per decode step (eager profile, context at most 64)";
const stepBefore = measured(wired, stepRow(attentionRow), "before (e07c022)");
const stepAfter = measured(wired, stepRow(attentionRow), "after (79c1f91)");
const tpotRow = "batch-1 TPOT with graphs";
const tokensRow = "bench tokens/s at max batch 1 / 4 / 16 with graphs";
const batch16 = (column: string) => {
  const { header, rows } = profilingTable(wired, 0);
  const cell = rows.find((r) => r[0] === tokensRow)?.[header.indexOf(column)] ?? "";
  const value = Number(cell.split("/").pop()?.trim());
  if (!Number.isFinite(value)) throw new Error(`${wired}: no batch-16 tokens/s in "${column}"`);
  return value;
};
const stepPairs: Pair[] = [
  { label: "attention per decode step", v1: stepBefore, v2: stepAfter, unit: "µs" },
  { label: "batch-1 ms per token, CUDA Graphs", v1: measured(wired, stepRow(tpotRow), "before (e07c022)"), v2: measured(wired, stepRow(tpotRow), "after (79c1f91)"), unit: "ms" },
  { label: "batch-16 tokens/s, CUDA Graphs", v1: batch16("before (e07c022)"), v2: batch16("after (79c1f91)"), unit: "tok/s" },
];

const sharedPoints: SharedBytes[] = ["128", "512", "2,048", "8,192"].map((c) => {
  const row = { context: c };
  return { context: Number(c.replace(",", "")), v1: c === "8,192" ? undefined : measured(shared, row, "v1 bytes"), v2: measured(shared, row, "v2 bytes") };
});
const cap = 2048;
const limit = 48 * 1024;

const rowsChoices = ["1", "16"];
const contextChoices = ["128", "512", "2,048", "8,192"];

const v1At512 = measured(v2Section, { rows: "16", context: "512" }, "v1 (us)");
const v2At512 = measured(v2Section, { rows: "16", context: "512" }, "v2 (us)");
const prediction: Prediction = expectAnswer({
  question: "16 rows at 512 tokens: how many times faster is v2 than v1?",
  choices: ["under 2x", "2x to 5x", "over 5x"],
  answer: v1At512 / v2At512 < 2 ? "under 2x" : v1At512 / v2At512 <= 5 ? "2x to 5x" : "over 5x",
  why: `${(v1At512 / v2At512).toFixed(1)}x: ${v1At512.toFixed(1)} µs against ${v2At512.toFixed(1)} µs. Neither kernel is near the memory limit; v2 wins by keeping more warps busy and letting eight lanes share each key. Set rows 16 and context 512 to check.`,
}, "2x to 5x");

export function V1V2Picker() {
  const [rows, setRows] = useState("1");
  const [context, setContext] = useState("2,048");
  const { header, rows: table } = profilingTable(v2Section, 0);
  const row = table.find((r) => r[0] === rows && r[1] === context);
  const v1Cell = row?.[header.indexOf("v1 (us)")] ?? "";
  const v1 = Number(v1Cell.replace(",", ""));
  const v2 = measured(v2Section, { rows, context }, "v2 (us)");
  const rejected = !Number.isFinite(v1) || v1Cell === "n/a";
  const max = Math.max(v2, rejected ? 0 : v1);
  return (
    <div className="flex h-full flex-col gap-3 tabular-nums text-xs">
      <Predict {...prediction} />
      <div className="flex flex-wrap gap-x-5 gap-y-2">
        <Toggle label="rows" choices={rowsChoices} value={rows} onChange={setRows} />
        <Toggle label="context" choices={contextChoices} value={context} onChange={setContext} />
      </div>
      <div className="flex-1 space-y-3 pt-2">
        <div className="grid grid-cols-[3rem_1fr_5.5rem] items-center gap-2">
          <span className="text-muted">v1</span>
          <div className="h-3 min-w-0 rounded-full bg-border">
            <EasedBar fraction={rejected ? 0 : v1 / max} className="bg-baseline/70" />
          </div>
          <span className="text-right tabular-nums text-muted">{rejected ? "rejected" : <NumberFlow value={v1} suffix=" µs" format={{ minimumFractionDigits: 1, maximumFractionDigits: 1 }} />}</span>
        </div>
        <div className="grid grid-cols-[3rem_1fr_5.5rem] items-center gap-2">
          <span className="text-accent">v2</span>
          <div className="h-3 min-w-0 rounded-full bg-border">
            <EasedBar fraction={v2 / max} className="bg-accent" />
          </div>
          <span className="text-right tabular-nums text-accent">
            <NumberFlow value={v2} suffix=" µs" format={{ minimumFractionDigits: 1, maximumFractionDigits: 1 }} />
          </span>
        </div>
      </div>
      <div className="space-y-1 border-t border-border pt-2 leading-relaxed">
        <div>
          <span className="text-muted">v1 / v2 </span>
          <span className="text-text">{rejected ? `n/a: past the ${cap.toLocaleString("en-US")}-token cap` : `${(v1 / v2).toFixed(1)}x`}</span>
        </div>
      </div>
    </div>
  );
}

const steps: StoryStep[] = [
  {
    id: "v1",
    title: "v1: three passes",
    body: (
      <>
        <p>
          v1 gives each row and query head one thread block. Pass one writes a score for every token of the
          context into shared memory. Pass two finds the maximum, then the exponentials and their sum. Pass three
          multiplies by the values, one thread per dimension: only 64 of the block&apos;s threads work and the rest wait.
        </p>
        <p>
          Each thread also computes a whole dot product alone. One row at a long context:{" "}
          <span className="text-accent">{v1At2048} µs</span>.
        </p>
      </>
    ),
  },
  {
    id: "cap",
    title: "the cap",
    body: (
      <>
        <p>
          Keeping every score means shared memory grows with the context, one float per token. The host code sizes it from
          the block table and refuses any table longer than {cap.toLocaleString("en-US")} tokens, TinyLlama&apos;s full
          context. That is why the longest rows of the benchmark read n/a for v1.
        </p>
        <p>
          The memory itself was not the wall: at the cap v1 uses{" "}
          <span className="text-accent">{(sharedPoints[2].v1 ?? 0).toLocaleString("en-US")} bytes</span> per block,
          a fifth of the default limit. The cap was a design choice that a longer model would have hit.
        </p>
      </>
    ),
  },
  {
    id: "online",
    title: "one chunk at a time",
    body: (
      <>
        <p>
          v2 keeps three things between chunks of keys: the largest score so far, the sum of exponentials, and the
          weighted values. When a chunk brings a larger score, the old sum and values are scaled down by
          exp(m_old − m_new) and the new chunk is added.
        </p>
        <p>
          The result is the same softmax, in one pass, and shared memory holds one chunk:{" "}
          <span className="text-accent">{sharedPoints[0].v2.toLocaleString("en-US")} bytes</span> at any length.
          The cap is gone.
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
          In v1 neighbouring threads read keys a whole key apart. In v2 a group of <span className="text-accent">8</span> lanes shares one key: at every step
          they read neighbouring dimensions together, in one wide read, and a few shuffles add their partial sums.
        </p>
        <p>The values step also uses every thread instead of 64.</p>
      </>
    ),
  },
  {
    id: "profile",
    title: "what the profile says",
    body: (
      <>
        <p>
          Both kernels read far less than the memory system could deliver, so the gain is not bandwidth. v2 keeps more
          warps busy and gives them less to wait on.
        </p>
        <p>
          One thing that did not help: replacing v2&apos;s shared-memory reductions with warp shuffles was 15 to 20%
          slower, so it was dropped.
        </p>
      </>
    ),
  },
  {
    id: "step",
    title: "in the decode step",
    body: (
      <>
        <p>
          Wired into the runner, attention per decode step fell to <span className="text-accent">{stepAfter} µs</span> on
          short contexts. The figure shows v1&apos;s time beside it.
        </p>
        <p>
          Batch 16 gained the most, because attention work grows with the number of rows while the weight reads of the
          GEMVs do not.
        </p>
      </>
    ),
  },
  {
    id: "compare",
    title: "every size",
    body: (
      <>
        <p>
          From a context of a few hundred tokens up v2 is about <span className="text-accent">3x</span> faster than v1; at the
          shortest context with one row there is little to overlap and the gain is smaller. Try the measured times
          on the right.
        </p>
        <p>
          With a single row v2 still launches only 32 blocks, which is the next story:{" "}
          <Link href="/kernels/split-attention" className="link-underline text-accent">
            split-context attention
          </Link>
          .
        </p>
      </>
    ),
  },
];

const captions = [
  `one row at 2,048 tokens, ${v1At2048} µs per launch from the bench loop; a block has 256 threads, so 64 work and 192 wait in pass three; the threads and cells are a sketch`,
  "bytes from the launch code of each kernel (docs/profiling.md): 4 bytes per token for v1, and the benchmark's longest rows are 8,192 tokens; 48 KB is CUDA's default per-block limit",
  "example scores; m only ever rises, and every rise rescales what came before; a chunk is 256 tokens",
  "lane j reads dims j, j + 8, j + 16 … so each step's 8 reads are side by side in memory: one 16-byte read, three shuffles, against v1's keys 128 bytes apart",
  "Nsight Compute, clocks locked and caches cold, so slower than the bench loop",
  `attention per decode step ${stepBefore} → ${stepAfter} µs with contexts of at most 64 tokens (eager profile); before e07c022, after 79c1f91, scripts/reproduce.sh, RTX 5060 Laptop GPU`,
  `µs per launch from the bench loop, every row at the full context; v2 is ${(v1At2048 / v2At2048).toFixed(1)}x faster than v1 at one row and 2,048 tokens, and about 3x from 512 tokens up; the shortest context is 128 tokens`,
];

export const attentionV2Steps = steps;
export const attentionV2Captions = captions;
export const attentionV2Numbers = { v1At2048, v2At2048, sharedPoints, cap, limit };

export function AttentionV2Profile() {
  return <ProfileFigure pairs={profilePairs} dram={dram} />;
}

export function AttentionV2Step() {
  return <StepFigure pairs={stepPairs} />;
}

export function AttentionV2Story() {
  return (
    <StickyStory
      steps={steps}
      caption={(i) => captions[i]}
      figure={(i) => {
        switch (i) {
          case 0:
            return <ThreePassFigure us={v1At2048} />;
          case 1:
            return <SharedMemoryFigure points={sharedPoints} cap={cap} limit={limit} />;
          case 2:
            return <OnlineSoftmaxFigure />;
          case 3:
            return <LanesFigure />;
          case 4:
            return <AttentionV2Profile />;
          case 5:
            return <AttentionV2Step />;
          default:
            return <V1V2Picker />;
        }
      }}
    />
  );
}
