"use client";

import NumberFlow from "@number-flow/react";
import { useState } from "react";
import { measured } from "@/content/kernels/types";
import { cellNumber, decodeSteps, modelShape } from "@/lib/data";
import { EasedBar } from "../eased-bar";
import { StickyStory, type StoryStep } from "../sticky-story";
import { Toggle } from "../toggle";
import { BitsFigure, LayoutFigure, ParallelFigure, RotateFigure, SerialFigure } from "./figures";
import { Author } from "../authorship";
import { expectAnswer, type Prediction, Predict } from "../predict";

const fast = "fast-rope-branch-31";

const shape = modelShape();
const headDim = shape.hidden / shape.heads;
const pairs = headDim / 2;
const qkHeads = shape.heads + shape.kv_heads;
const qkvWidth = (shape.heads + 2 * shape.kv_heads) * headDim;
const draftThreads = 128;
const groups = 8;
const fastThreads = groups * pairs;
const draftTrig = qkHeads * pairs;

const tokenChoices = ["1", "16", "64", "512"];
const at = (tokens: string) => ({ tokens });

const naive1 = measured(fast, at("1"), "naive (us)");
const draft1 = measured(fast, at("1"), "first draft (us)");
const fast1 = measured(fast, at("1"), "fast (us)");
const naive512 = measured(fast, at("512"), "naive (us)");
const draft512 = measured(fast, at("512"), "first draft (us)");
const fast512 = measured(fast, at("512"), "fast (us)");
const draftFlat = `${tokenChoices.map((t) => measured(fast, at(t), "first draft (us)").toFixed(1)).join(" / ")} µs`;

const stepOp = (() => {
  const t = decodeSteps.b1;
  const row = t?.rows.find((r) => r[0] === "rope");
  if (!t || !row) return undefined;
  const us = cellNumber(row[t.header.indexOf("GPU us/step")]);
  const launches = cellNumber(row[t.header.indexOf("kernels/step")]);
  return us !== undefined && launches !== undefined ? { us, launches } : undefined;
})();

const versions = [
  { column: "naive (us)", label: "naive" },
  { column: "first draft (us)", label: "first draft" },
  { column: "fast (us)", label: "fast" },
];

const fastOne = measured(fast, at("1"), "fast (us)");
const fast64 = measured(fast, at("64"), "fast (us)");
const prediction: Prediction = expectAnswer({
  question: "From 1 token to 64, how much longer does the fast kernel take?",
  choices: ["about 64 times", "a few times", "barely longer"],
  answer: fast64 / fastOne > 20 ? "about 64 times" : fast64 / fastOne > 2 ? "a few times" : "barely longer",
  why: `${fastOne.toFixed(1)} µs at 1 token, ${fast64.toFixed(1)} µs at 64. A few hundred KB at most is far from the memory limit, so the launch and one chain of load, sine and store set the time. Try 512.`,
}, "barely longer");

export function TokensPicker() {
  const [tokens, setTokens] = useState("1");
  const times = versions.map((v) => ({ ...v, us: measured(fast, at(tokens), v.column) }));
  const max = Math.max(...times.map((t) => t.us));
  const kb = measured(fast, at(tokens), "DRAM, fast (KB)");
  const fastTime = times[2].us;
  return (
    <div className="flex h-full flex-col gap-3 tabular-nums text-xs">
      <Predict {...prediction} />
      <Toggle label="tokens" choices={tokenChoices} value={tokens} onChange={setTokens} />
      <div className="space-y-1.5 pt-1">
        <div className="text-muted">µs per launch, 32 + 4 heads of 64</div>
        {times.map((t, i) => (
          <div key={t.column} className="grid grid-cols-[5.5rem_1fr_4.5rem] items-center gap-2">
            <span className={i === 2 ? "text-accent" : "text-muted"}>{t.label}</span>
            <div className="h-3 min-w-0 rounded-full bg-border">
              <EasedBar fraction={t.us / max} className={i === 2 ? "bg-accent" : "bg-baseline/70"} />
            </div>
            <span className="text-right tabular-nums text-muted">
              <NumberFlow value={t.us} suffix=" µs" format={{ minimumFractionDigits: 1, maximumFractionDigits: 1 }} />
            </span>
          </div>
        ))}
      </div>
      <div className="space-y-1">
        <div className="text-muted">DRAM bytes per launch, fast</div>
        <div className="tabular-nums text-text">
          <NumberFlow value={kb} suffix=" KB" format={{ maximumFractionDigits: 1 }} />
        </div>
      </div>
      <div className="mt-auto space-y-1 border-t border-border pt-2 leading-relaxed">
        <div>
          <span className="text-muted">fast against naive </span>
          <span className="text-accent tabular-nums">{`${(times[0].us / fastTime).toFixed(1)}x faster`}</span>
        </div>
        {stepOp && (
          <div>
            <span className="text-muted">batch-1 decode step: </span>
            <span className="text-text tabular-nums">{`${stepOp.us.toFixed(1)} µs, ${stepOp.launches} launches`}</span>
          </div>
        )}
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
          Attention compares every query with every key, and on its own it has no idea which token came first. RoPE gives
          it the order by turning the query and key vectors, not by adding anything to them.
        </p>
        <p>
          Each head&apos;s values are read as <span className="text-accent">{pairs} pairs</span>, and pair i is turned by the angle position · θᵢ. A query at
          position p and a key at position p′ then meet at an angle that depends only on p − p′. The first pairs turn
          fast and tell nearby tokens apart; the last ones barely move and carry the long distances.
        </p>
      </>
    ),
  },
  {
    id: "layout",
    title: "which values turn",
    body: (
      <>
        <p>
          The QKV projection leaves one row of values per token: the query heads, then the key heads, then the value
          heads. RoPE turns the Q and K heads in place, right before the keys are written to the cache, and leaves V
          alone.
        </p>
        <p>
          Inside a head, value i pairs with value i + <span className="text-accent">{pairs}</span>, not with its neighbour; that is how Hugging Face&apos;s
          rotate_half lays it out, and the weights were trained that way. Each pair costs one cosine, one sine, four
          products and two sums.
        </p>
      </>
    ),
  },
  {
    id: "bits",
    title: "the same bits as Hugging Face",
    body: (
      <>
        <p>
          The naive kernel copies PyTorch&apos;s arithmetic exactly: cos and sin in FP32, rounded to BF16, then each
          product rounded on its own. The fast kernel has to match the naive one bit for bit.
        </p>
        <p>
          The frequencies were the trap. A device powf differs from the host&apos;s in the last bit for a couple of the frequencies, the
          angle multiplies that error by the position, and near <span className="text-accent">2,047</span> a sine lands on the other side of a BF16 rounding
          boundary. The subtraction then turns one ulp into dozens. Both kernels now take the frequencies from a table the
          host computes.
        </p>
      </>
    ),
  },
  {
    id: "serial",
    title: "one thread per head",
    body: (
      <>
        <p>
          <Author story="Joe's first draft" book="The first draft" /> gave each head one thread, so most of a token&apos;s threads had nothing to do and each busy one walked its
          pairs one after another, computing every sine and cosine again for every head.
        </p>
        <p>
          Correct once the frequencies came from the host, but flat at about{" "}
          <span className="text-accent">{draft1.toFixed(0)} µs</span> at every token count, slower than the naive kernel at one token. The time
          was the chain of pairs each thread walked, not the memory: only at the largest token count did the naive kernel get slower.
        </p>
      </>
    ),
  },
  {
    id: "parallel",
    title: "every pair at once",
    body: (
      <>
        <p>
          The fast kernel turns the mapping around: one thread per pair in each head group, and a few groups,{" "}
          <span className="text-accent">{fastThreads} threads</span> per token. Each thread computes the cosine and sine of its pair once and applies them to every head of its group, so the
          heads no longer wait for each other.
        </p>
        <p>
          That is far fewer sines and cosines per token, and{" "}
          <span className="text-accent">{fast1.toFixed(1)} µs</span> for one token. Neighbouring threads read
          neighbouring pairs, so a warp&apos;s narrow loads already coalesce into a few contiguous sectors; the wide vector loads of the
          spec were never needed.
        </p>
      </>
    ),
  },
  {
    id: "measured",
    title: "measured",
    body: (
      <>
        <p>
          For small token counts the fast kernel stays near its one-token time: a few hundred KB at most, far from the memory
          limit, so the launch and one chain of load, sine and store set the time. At the largest count it is{" "}
          <span className="text-accent">{fast512.toFixed(1)} µs</span>, half the naive kernel&apos;s. Try the
          token counts on the right.
        </p>
        <p>
          In a decode step RoPE is under 1% of the time, so the gain is small in absolute terms. The point was to stop being
          slower than the kernel it replaced, and to stay bit-identical while doing it.
        </p>
      </>
    ),
  },
];

const captions = [
  `8 of the ${pairs} pairs of one head of ${headDim} values, at their real rates; the vector lengths are illustrative`,
  `TinyLlama's layout: a row of ${qkvWidth.toLocaleString("en-US")} values per token, ${shape.heads} query and ${shape.kv_heads} key/value heads of ${headDim}; RoPE turns the ${qkHeads} Q and K heads`,
  `the chain of the bug found while writing the naive kernel: the device powf differed in the last bit for 2 of the ${pairs} frequencies; bit patterns illustrative`,
  `µs per launch, Nsight Compute: first draft ${draftFlat} at ${tokenChoices.join(" / ")} tokens, naive ${naive1.toFixed(1)} µs at one token and ${naive512.toFixed(1)} at 512, draft ${draft512.toFixed(1)}; the draft ran ${draftThreads} threads per token, ${qkHeads} with a head, ${pairs} pairs each; the pair segments are an illustration, not measured`,
  `µs per launch for one token, Nsight Compute; ${pairs} threads per head group, ${groups} groups, each thread on ${Math.floor(qkHeads / groups)} or ${Math.ceil(qkHeads / groups)} heads; sin and cos counts from the code, ${fastThreads} per token instead of ${draftTrig.toLocaleString("en-US")}; the loads are 2 bytes, where the spec asked for 128-bit`,
  `Nsight Compute, RTX 5060 Laptop GPU, one launch after ten warm-ups; the fast kernel takes ${fast1.toFixed(1)} µs at one token and ${fast512.toFixed(1)} at 512, the naive ${naive512.toFixed(1)}`,
];

export const ropeSteps = steps;
export const ropeCaptions = captions;
export const ropeNumbers = { naive: naive1, draft: draft1, fast: fast1 };

export function RopeStory() {
  return (
    <StickyStory
      steps={steps}
      caption={(i) => captions[i]}
      figure={(i) => {
        switch (i) {
          case 0:
            return <RotateFigure />;
          case 1:
            return <LayoutFigure />;
          case 2:
            return <BitsFigure />;
          case 3:
            return <SerialFigure naive={naive1} draft={draft1} />;
          case 4:
            return <ParallelFigure naive={naive1} fast={fast1} />;
          default:
            return <TokensPicker />;
        }
      }}
    />
  );
}
