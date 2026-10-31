"use client";

import NumberFlow from "@number-flow/react";
import { useState } from "react";
import { measured } from "@/content/kernels/types";
import { profilingTable } from "@/lib/data";
import { EasedBar } from "../eased-bar";
import { StickyStory, type StoryStep } from "../sticky-story";
import { Toggle } from "../toggle";
import { ArgmaxFigure, CostFigure, Fp64Figure, GumbelFigure, type Mode, PhiloxFigure, TopKFigure, TopPFigure } from "./figures";

const section = "sampling-branch-65";

const columns = [
  { column: "greedy", label: "greedy" },
  { column: "top-k 50", label: "top-k 50" },
  { column: "top-k 50 + top-p 0.9", label: "top-k + top-p" },
  { column: "full sort alone", label: "sort alone", warn: true },
  { column: "top-p 0.9", label: "top-p 0.9" },
  { column: "temperature", label: "temperature only" },
];

const at = (rows: string, column: string) => measured(section, { rows }, column);
const one = (column: string) => at("1", column);

const fp64 = (metric: string) => measured(section, { metric }, "value", 1);
const busy = fp64("FP64 pipe busy");
const share = fp64("FP64 share of instructions");
const floatRuns = (() => {
  const { header, rows } = profilingTable(section, 2);
  const row = rows.find((r) => r[0] === "float `u` and `__logf`");
  const runs = ["1 row, temperature (us)", "16 rows, temperature (us)"].flatMap((c) => (row?.[header.indexOf(c)] ?? "").split("/").map((v) => Number(v.trim())));
  if (runs.length === 0 || runs.some((v) => !Number.isFinite(v))) throw new Error(`${section}: no float gumbel runs`);
  return runs;
})();

const costModes: Mode[] = columns.map((c) => ({ label: c.label, us: one(c.column), accent: c.column === "top-k 50", warn: c.warn }));

function ModePicker() {
  const [rows, setRows] = useState("1");
  const times = columns.map((c) => ({ ...c, us: at(rows, c.column) }));
  const max = Math.max(...times.map((t) => t.us));
  const sort = at(rows, "full sort alone");
  return (
    <div className="flex h-full flex-col gap-3 font-mono text-xs">
      <Toggle label="rows" choices={["1", "16"]} value={rows} onChange={setRows} />
      <div className="flex-1 space-y-2 pt-1">
        {times.map((t) => (
          <div key={t.column} className="grid grid-cols-[8rem_1fr_4.5rem] items-center gap-2">
            <span className={t.warn ? "text-warn" : t.us < sort ? "text-accent" : "text-muted"}>{t.label}</span>
            <div className="h-3 min-w-0 rounded-full bg-border">
              <EasedBar fraction={t.us / max} className={t.warn ? "bg-warn/70" : t.us < sort ? "bg-accent" : "bg-baseline/70"} />
            </div>
            <span className="text-right tabular-nums text-muted">
              <NumberFlow value={t.us} suffix=" µs" format={{ minimumFractionDigits: 1, maximumFractionDigits: 1 }} />
            </span>
          </div>
        ))}
      </div>
      <div className="space-y-1 border-t border-border pt-2 leading-relaxed text-muted">
        <div>{`${rows === "1" ? "1 row uses 1 of 26 SMs" : "16 rows run side by side on 16 SMs"}, so the times barely move`}</div>
      </div>
    </div>
  );
}

const steps: StoryStep[] = [
  {
    id: "greedy",
    title: "one id out of 32,000",
    body: (
      <>
        <p>
          After the LM head, each sequence has 32,000 logits and needs one token id. Greedy decoding takes the largest:
          1,024 threads stride across the row, each keeping the best value and index it saw, and a tree halves them to
          one.
        </p>
        <p>
          Ties go to the lowest index, so the result never depends on thread timing:{" "}
          <span className="font-mono text-text">{one("greedy")} µs</span> per call.
        </p>
      </>
    ),
  },
  {
    id: "gumbel",
    title: "sampling is an argmax too",
    body: (
      <>
        <p>
          Sampling from softmax(logits / T) usually means exponentials, a sum, a running total and a search. The
          Gumbel-max trick skips all of that: add independent Gumbel noise, −log(−log u), to every scaled logit and take
          the argmax. The winner is distributed exactly as the softmax.
        </p>
        <p>
          So temperature sampling is the greedy kernel with noise added. The figure draws a sample every 0.7 s; the
          picks settle onto the softmax outline.
        </p>
      </>
    ),
  },
  {
    id: "philox",
    title: "noise you can replay",
    body: (
      <>
        <p>
          The noise comes from Philox, a counter-based generator: token i at decode step s gets the number at offset
          s × 32,000 + i of the request&apos;s seed. Nothing is stored between calls.
        </p>
        <p>
          The same seed repeats a whole answer, and a row draws the same token whether it runs alone or in a batch of
          16, which the tests check. Each row carries its own seed, step and temperature, so greedy and sampled requests
          share one launch.
        </p>
      </>
    ),
  },
  {
    id: "topk",
    title: "top-k without a sort",
    body: (
      <>
        <p>
          Top-k keeps the k largest logits. Sorting 32,000 values per row to find them is the usual way. Here the
          kernel binary-searches a threshold instead: each pass every thread counts its logits at or above the cut, one
          block sum, and the cut moves up or down.
        </p>
        <p>
          Logits become 32-bit keys that sort like the floats, so 32 passes find the exact cut, and every token tied
          with the k-th largest stays in.
        </p>
      </>
    ),
  },
  {
    id: "topp",
    title: "top-p: search the mass",
    body: (
      <>
        <p>
          Top-p keeps the smallest set of top tokens whose probability reaches p. The same search works on mass instead
          of a count: each pass adds exp(s − max) over the logits at or above the cut and compares it with p times the
          total.
        </p>
        <p>The search starts from top-k&apos;s cut when both are set, as in Hugging Face, which applies top-k first.</p>
      </>
    ),
  },
  {
    id: "cost",
    title: "what it costs",
    body: (
      <>
        <p>
          Top-k 50 takes <span className="font-mono text-accent">{one("top-k 50")} µs</span>. Sorting the row alone, before
          any prefix sum or draw, takes <span className="font-mono text-text">{one("full sort alone")} µs</span>, so the
          search wins by about 2x.
        </p>
        <p>
          Two columns break the pattern: top-p alone, and plain temperature sampling at{" "}
          <span className="font-mono text-warn">{one("temperature")} µs</span>, the slowest mode of all.
        </p>
      </>
    ),
  },
  {
    id: "fp64",
    title: "the double-precision log",
    body: (
      <>
        <p>
          The noise uses double-precision log. A laptop GPU runs FP64 at a small fraction of its FP32 rate, and Nsight
          Compute shows the FP64 pipe busy {busy}% of the time for {share}% of the instructions. With temperature alone
          every one of 32,000 tokens gets noise; with top-k only the 50 kept ones do.
        </p>
        <p>
          A throwaway float version ran in {Math.min(...floatRuns).toFixed(0)} to {Math.max(...floatRuns).toFixed(0)} µs.
          It changes which token a fixed seed draws, so it is written up for a decision rather than merged.
        </p>
      </>
    ),
  },
  {
    id: "rows",
    title: "one block per row",
    body: (
      <>
        <p>
          Every mode launches one block per row, so one row uses one of the 26 SMs and 16 rows use 16. That is why the
          times barely change with the batch: the rows run side by side.
        </p>
        <p>Try both row counts on the right.</p>
      </>
    ),
  },
];

const captions = [
  `${one("greedy")} µs per call from scripts/sample_bench.cu; 12 of 32,000 logits shown`,
  "a live simulation of Gumbel-max on these 12 logits at T = 1, not engine output",
  "the offsets are the kernel's; the cells are a sketch",
  "example keys; the kernel searches the 32-bit key of every logit, 32 passes",
  "example keys; the bar is the kept share of the probability mass",
  "µs per call, first of two runs, RTX 5060 Laptop GPU (docs/profiling.md)",
  "Nsight Compute on one temperature-only launch; float times from a throwaway copy",
  "µs per call, first of two runs; green is faster than sorting the row alone",
];

export function SamplingStory() {
  return (
    <StickyStory
      steps={steps}
      caption={(i) => captions[i]}
      figure={(i) => {
        switch (i) {
          case 0:
            return <ArgmaxFigure us={one("greedy")} />;
          case 1:
            return <GumbelFigure />;
          case 2:
            return <PhiloxFigure />;
          case 3:
            return <TopKFigure k={6} />;
          case 4:
            return <TopPFigure p={0.9} />;
          case 5:
            return <CostFigure modes={costModes} />;
          case 6:
            return <Fp64Figure busy={busy} share={share} now={one("temperature")} float={[Math.min(...floatRuns), Math.max(...floatRuns)]} />;
          default:
            return <ModePicker />;
        }
      }}
    />
  );
}
