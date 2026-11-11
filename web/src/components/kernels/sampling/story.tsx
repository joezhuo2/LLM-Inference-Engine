"use client";

import NumberFlow from "@number-flow/react";
import { useState } from "react";
import { measured } from "@/content/kernels/types";
import { hardware, modelShape, profilingTable } from "@/lib/data";
import { EasedBar } from "../eased-bar";
import { StickyStory, type StoryStep } from "../sticky-story";
import { Toggle } from "../toggle";
import { ArgmaxFigure, CostFigure, Fp64Figure, GumbelFigure, type Mode, PhiloxFigure, TopKFigure, TopPFigure } from "./figures";
import { expectAnswer, type Prediction, Predict } from "../predict";

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
const vocab = modelShape().vocab.toLocaleString("en-US");
const sms = hardware().sms;

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

const greedyRatio = at("16", "greedy") / at("1", "greedy");
const prediction: Prediction = expectAnswer({
  question: "From 1 row to 16, how much longer does greedy decoding take?",
  choices: ["about 16 times", "a few times", "barely longer"],
  answer: greedyRatio > 8 ? "about 16 times" : greedyRatio > 1.6 ? "a few times" : "barely longer",
  why: `Greedy takes ${greedyRatio.toPrecision(2)}x as long at 16 rows as at 1 row (calculated from the two bars). Each row gets its own block, so the rows run side by side on different SMs.`,
}, "barely longer");

export function ModePicker() {
  const [rows, setRows] = useState("1");
  const times = columns.map((c) => ({ ...c, us: at(rows, c.column) }));
  const max = Math.max(...times.map((t) => t.us));
  const sort = at(rows, "full sort alone");
  return (
    <div className="flex h-full flex-col gap-3 tabular-nums text-xs">
      <Predict {...prediction} />
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
        <div>{`${rows === "1" ? `1 row uses 1 of ${sms} SMs` : "16 rows run side by side on 16 SMs"}, so the times barely move`}</div>
      </div>
    </div>
  );
}

const steps: StoryStep[] = [
  {
    id: "greedy",
    title: `greedy is an argmax over ${vocab} logits`,
    body: (
      <>
        <p>
          After the LM head, each sequence has {vocab} logits and needs one token id. Greedy decoding takes the largest:
          a block of threads strides across the row, each keeping the best value and index it saw, and a tree halves them to
          one.
        </p>
        <p>
          Ties go to the lowest index, so the result never depends on thread timing:{" "}
          <span className="text-accent">{one("greedy")} µs</span> per call.
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
          So temperature sampling is the greedy kernel with noise added. The figure draws repeated samples; the
          picks settle onto the softmax outline.
        </p>
      </>
    ),
  },
  {
    id: "philox",
    title: "counter-based noise replays exactly",
    body: (
      <>
        <p>
          The noise comes from Philox, a counter-based generator: token i at decode step s gets the number at offset
          s × <span className="text-accent">{vocab}</span> + i of the request&apos;s seed. Nothing is stored between calls.
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
    title: "top-k: search for the cut, no sort",
    body: (
      <>
        <p>
          Top-k keeps the k largest logits. Sorting {vocab} values per row to find them is the usual way. Here the
          kernel binary-searches a threshold instead: each pass every thread counts its logits at or above the cut, one
          block sum, and the cut moves up or down.
        </p>
        <p>
          Logits become keys that sort like the floats, and each pass settles one bit of the key, so <span className="text-accent">32 passes</span> find the exact cut, and every token tied
          with the k-th largest stays in.
        </p>
      </>
    ),
  },
  {
    id: "topp",
    title: "top-p: the same search, on mass",
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
    title: `top-k runs ${(one("full sort alone") / one("top-k 50")).toFixed(1)}x faster than a bare sort`,
    group: "measured",
    body: (
      <>
        <p>
          Top-k takes <span className="text-accent">{one("top-k 50")} µs</span>. Sorting the row alone, before
          any prefix sum or draw, takes about twice as long, so the search wins comfortably.
        </p>
        <p>
          Two columns break the pattern: top-p alone, and plain temperature sampling at{" "}
          <span className="text-warn">{one("temperature")} µs</span>, the slowest mode of all.
        </p>
      </>
    ),
  },
  {
    id: "fp64",
    title: "FP64 logs make temperature slowest",
    group: "measured",
    body: (
      <>
        <p>
          The noise uses double-precision log. A laptop GPU runs FP64 at a small fraction of its FP32 rate, and Nsight
          Compute shows the FP64 pipe busy <span className="text-accent">{busy}%</span> of the time for a small share of the instructions. With temperature alone
          every token gets noise; with top-k only the kept ones do.
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
    title: "rows run side by side, one block each",
    group: "measured",
    body: (
      <>
        <p>
          Every mode launches one block per row, so a single row uses one of the <span className="text-accent">{sms} SMs</span> and each extra row uses another. That is why the
          times barely change with the batch: the rows run side by side.
        </p>
        <p>Try both row counts on the right.</p>
      </>
    ),
  },
];

const captions = [
  `${one("greedy")} µs per call from scripts/sample_bench.cu, 1,024 threads striding across the row; 12 of ${vocab} logits shown`,
  "a live simulation of Gumbel-max on these 12 logits at T = 1, not engine output",
  "the offsets are the kernel's; the cells are a sketch",
  "example keys; the kernel searches the 32-bit key of every logit, 32 passes",
  "example keys; the bar is the kept share of the probability mass",
  `µs per call at 1 row, first of two runs, RTX 5060 Laptop GPU (docs/profiling.md); top-k 50 ${one("top-k 50")}, sorting the row alone ${one("full sort alone")}`,
  `Nsight Compute on one temperature-only launch: the FP64 instructions are ${share}% of all instructions, and ${vocab} tokens get noise where top-k 50 computes it for 50; float times from a throwaway copy`,
  "µs per call, first of two runs; blue is faster than sorting the row alone",
];

export const samplingSteps = steps;
export const samplingCaptions = captions;
export const samplingNumbers = {
  modes: columns.map((c) => ({ column: c.column, label: c.label, sort: c.column === "full sort alone", us: { "1": one(c.column), "16": at("16", c.column) } })),
  greedy: one("greedy"),
  temperature: one("temperature"),
  topK: one("top-k 50"),
  sort: one("full sort alone"),
  busy,
  share,
  float: [Math.min(...floatRuns), Math.max(...floatRuns)] as [number, number],
};

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
