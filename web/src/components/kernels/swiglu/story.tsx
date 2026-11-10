"use client";

import NumberFlow from "@number-flow/react";
import { useState } from "react";
import { chapterNumber } from "@/content/book-order";
import { measured } from "@/content/kernels/types";
import { cellNumber, decodeSteps, hardware, modelShape } from "@/lib/data";
import { EasedBar } from "../eased-bar";
import { StickyStory, type StoryStep } from "../sticky-story";
import { Toggle } from "../toggle";
import { ChainFigure, GridFigure, peakGbs, SmFigure, VectorFigure, ZipFigure } from "./figures";
import { Author } from "../authorship";
import { expectAnswer, type Prediction, Predict } from "../predict";

const fast = "fast-swiglu-branch-32";
const small = "small-batches-branch-33";

const tokenChoices = ["1", "16", "64", "512", "4096"];
const at = (tokens: string) => ({ tokens });

const shape = modelShape();
const bf16 = 2;
const gateUpRowBytes = 2 * shape.intermediate * bf16;
const gateToUpBytes = (shape.intermediate * bf16).toLocaleString("en-US");
const gateUpWeightBytes = 2 * shape.hidden * shape.intermediate * bf16;
const roundTripKb = ((2 * gateUpRowBytes) / 1024).toFixed(1);
const roundTripShare = (((2 * gateUpRowBytes) / gateUpWeightBytes) * 100).toFixed(1);
const weightsMb = (gateUpWeightBytes / 1024 ** 2).toFixed(1);
const fmtInt = (v: number) => v.toLocaleString("en-US");
const ffText = fmtInt(shape.intermediate);
const hiddenText = fmtInt(shape.hidden);
const rowValues = fmtInt(2 * shape.intermediate);
const vecWidth = 16 / bf16;
const scalarInstructions = 3 * vecWidth;
const vectors = shape.intermediate / vecWidth;
const threads = 256;
const rowBlocks = shape.intermediate / threads;
const sms = hardware().sms;
const smallTokens = Math.floor((sms * (threads / 2) - 1) / vectors);
const smallBlocks = Math.ceil(vectors / threads);

const gbs = (tokens: string, column: string) => (measured(fast, at(tokens), "DRAM, fast (KB)") * 1024) / (measured(fast, at(tokens), column) * 1000);

const naive1 = measured(fast, at("1"), "naive (us)");
const draft1 = measured(fast, at("1"), "first draft (us)");
const naiveBig = measured(fast, at("4096"), "naive (us)");
const vectorBig = measured(fast, at("4096"), "fast (us)");
const smallNaive = measured(small, at("1"), "naive (us)");
const smallVector = measured(small, at("1"), "previous (us)");
const smallFast = measured(small, at("1"), "fast (us)");

const versions = [
  { column: "naive (us)", label: "naive" },
  { column: "first draft (us)", label: "first draft" },
  { column: "fast (us)", label: "vector" },
];

const stepOp = (() => {
  const t = decodeSteps.b1;
  const row = t?.rows.find((r) => r[0] === "swiglu");
  if (!t || !row) return undefined;
  const us = cellNumber(row[t.header.indexOf("GPU us/step")]);
  const launches = cellNumber(row[t.header.indexOf("kernels/step")]);
  return us !== undefined && launches !== undefined ? { us, launches } : undefined;
})();

const vector1 = measured(fast, at("1"), "fast (us)");
const gainTokens = ["64", "512", "4096"];
const gains = gainTokens.map((t) => measured(fast, at(t), "naive (us)") / measured(fast, at(t), "fast (us)"));
const gainLow = Math.min(...gains).toFixed(1);
const gainHigh = Math.max(...gains).toFixed(1);
const smallKb = ["1", "16"].map((t) => fmtInt(measured(fast, at(t), "DRAM, fast (KB)")));
const stepShare = (() => {
  const t = decodeSteps.b1;
  const row = t?.rows.find((r) => r[0] === "swiglu");
  const value = t && row ? cellNumber(row[t.header.indexOf("% of kernel time")]) : undefined;
  if (value === undefined) throw new Error("swiglu: no share of kernel time in the batch-1 decode step");
  return value;
})();
const prediction: Prediction = expectAnswer({
  question: "At 1 token, is the vector kernel, with its 16-byte loads, faster than naive?",
  choices: ["faster", "about the same", "slower"],
  answer: vector1 / naive1 < 0.9 ? "faster" : vector1 / naive1 <= 1.1 ? "about the same" : "slower",
  why: `At 1 token the vector kernel takes ${(vector1 / naive1).toPrecision(2)}x of naive's time (calculated from the two bars). The row makes too few vectors to fill the GPU, so wide loads have little to save. Switch to the largest size to see where they pay.`,
}, "about the same");

export function TokensPicker() {
  const [tokens, setTokens] = useState("4096");
  const times = versions.map((v) => ({ ...v, us: measured(fast, at(tokens), v.column) }));
  const max = Math.max(...times.map((t) => t.us));
  const bandwidth = gbs(tokens, "fast (us)");
  return (
    <div className="flex h-full flex-col gap-3 tabular-nums text-xs">
      <Predict {...prediction} />
      <Toggle label="tokens" choices={tokenChoices} value={tokens} onChange={setTokens} />
      <div className="space-y-1.5 pt-1">
        <div className="text-muted">µs per launch, ff 5,632</div>
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
      <div className="space-y-1.5">
        <div className="text-muted">{`vector kernel, DRAM bytes ÷ time, against the ${peakGbs} GB/s peak`}</div>
        <div className="grid grid-cols-[1fr_4.5rem] items-center gap-2">
          <div className="h-3 min-w-0 rounded-full bg-border">
            <EasedBar fraction={Math.min(1, bandwidth / peakGbs)} className="bg-accent" />
          </div>
          <span className="text-right tabular-nums text-accent">
            <NumberFlow value={Math.round(bandwidth)} suffix=" GB/s" />
          </span>
        </div>
      </div>
      <div className="mt-auto space-y-1 border-t border-border pt-2 leading-relaxed">
        <div>
          <span className="text-muted">vector against naive </span>
          <span className="text-accent tabular-nums">{`${(times[0].us / times[2].us).toFixed(2)}x faster`}</span>
        </div>
        {Number(tokens) < 5 && (
          <div className="text-muted">{`below 5 tokens the kernel now runs the scalar path: ${(smallFast / smallNaive).toFixed(2)}x of naive time`}</div>
        )}
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
          Every layer&apos;s MLP widens the hidden state twice, with a gate projection and an up
          projection, mixes the two, and narrows it back down. The engine runs both projections as one GEMM, so each token
          comes out as a single row of <span className="text-accent">{rowValues} values</span>: the gate half, then the up half.
        </p>
        <p>
          SwiGLU is the mixing. Output j is silu(gate j) times up j. silu is close to zero for a negative gate and close to
          the gate itself for a large one, so the gate decides, value by value, how much of up gets through.
        </p>
      </>
    ),
  },
  {
    id: "bits",
    title: "two roundings",
    body: (
      <>
        <p>
          PyTorch computes this in BF16 as two operations, act_fn(gate) and then the product, so the silu is rounded to BF16
          before it meets up. The naive kernel does exactly that, with an accurate expf and a true divide, and reproduces
          Hugging Face&apos;s act tensor bit for bit on every traced layer.
        </p>
        <p>
          This is already a small case of the fusion in chapter {chapterNumber("add-rmsnorm")}, &quot;fusion, the pattern&quot;: silu(gate) stays in a
          register, and the BF16 rounding that writing it out as a tensor would have done is still done there. Every
          faster version had to keep that chain. Skipping the first rounding looks harmless and gives a different
          answer: in the example on the right the output moves by <span className="text-accent">one BF16 step</span>.
        </p>
      </>
    ),
  },
  {
    id: "draft",
    title: "one thread per value",
    body: (
      <>
        <p>
          <Author story="Joe's first draft" book="The first draft" /> kept the naive kernel&apos;s work split, one thread per output value, on a 2D grid: blocks of threads along a row, and the row on blockIdx.y. It was bit-identical from the start and a
          little faster than naive at every size, <span className="text-accent">{draft1.toFixed(1)} µs</span> at one token.
        </p>
        <p>
          The 2D grid had a ceiling: gridDim.y stops at 65,535, so a batch with more rows would not launch. The engine never
          sends that many rows, but the spec promised any shape, and the tests now launch more rows than that. Both kernels now run on one flat grid
          and work out the row from the index.
        </p>
      </>
    ),
  },
  {
    id: "vector",
    title: "eight values per load",
    body: (
      <>
        <p>
          A thread that loads one value at a time issues a memory instruction for every value it touches. The
          vector kernel reads a vector of gate values and a vector of up values with two wide loads and writes the matching outputs with one wide
          store: an eighth of the instructions for the same bytes, so the threads spend their time waiting on memory and not issuing loads.
          The two loads are half a row apart. The one GEMM writes each row as the gate half and then the up half, so up
          value j sits half a row after gate value j, and because half a row is a whole number of vectors both loads stay aligned.
        </p>
        <p>
          With enough tokens that is what the kernel is limited by. At the largest size it takes{" "}
          <span className="text-accent">{vectorBig.toFixed(0)} µs</span>, well under naive&apos;s time, which is the GPU&apos;s measured
          peak bandwidth: there is no faster way to move those bytes.
        </p>
      </>
    ),
  },
  {
    id: "small",
    title: "too few blocks",
    body: (
      <>
        <p>
          A decode step has one token per request. At one token the row makes only a few hundred vectors, {smallBlocks} blocks on a GPU
          with many SMs, and each thread runs its exp and divide chains one after another. The vector kernel was no faster
          than naive there.
        </p>
        <p>
          So the kernel counts its work first: below half a block per SM, a handful of tokens here, it takes the scalar path,
          whose blocks spread the same chains over most of the GPU. At one token that is{" "}
          <span className="text-accent">{smallFast.toFixed(2)} µs</span>, faster than the vector kernel in the same session.
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
          At the larger token counts the vector kernel is <span className="text-accent">{gainLow}x to {gainHigh}x</span> faster than naive, and the bandwidth line shows why the gain
          stops there. At the smallest token counts the work is small and the launch itself sets the time. Try the token
          counts on the right.
        </p>
        <p>
          In a decode step SwiGLU is about <span className="text-accent">{stepShare}%</span> of the kernel time, so none of this moves the token rate. Folding it
          into the gate/up GEMM&apos;s epilogue would be the fusion of chapter {chapterNumber("add-rmsnorm")} again. Writing that row and reading it back
          is a small share of the gate and up weights the GEMM reads, so at decode it would save a launch more than bytes; that is not done.
        </p>
      </>
    ),
  },
];

const captions = [
  `the hidden state has ${hiddenText} values and each half of the row ${ffText}; 16 of ${ffText} values per half drawn; the silu curve is exact`,
  `the example is computed with the kernel's arithmetic; any g and u with the same roundings would do; the naive kernel matched Hugging Face's act tensor on all ${shape.layers} layers`,
  `µs per launch for one token, Nsight Compute: naive ${naive1.toFixed(1)}, first draft ${draft1.toFixed(1)}; ${rowBlocks} blocks of ${threads} threads cover a row of ${ffText}, and the tests launch 70,000 rows; 4 of the rows drawn`,
  `µs and DRAM bytes from Nsight Compute at 4,096 tokens: vector ${vectorBig.toFixed(0)}, naive ${naiveBig.toFixed(0)}; each value is ${bf16} bytes, so the scalar path issues ${scalarInstructions} memory instructions for ${vecWidth} outputs and the vector path 3; up sits ${ffText} values (${gateToUpBytes} bytes) after gate, a multiple of ${vecWidth}; the peak from the bandwidth test in docs/results.md`,
  `µs per launch for one token, Nsight Compute, five runs: the vector kernel took ${smallVector.toFixed(2)} µs; ${ffText} values make ${vectors} vectors, ${smallBlocks} blocks on ${sms} SMs with ${vecWidth} chains per thread; the scalar path runs ${rowBlocks} blocks and takes ${smallTokens} tokens or fewer; chain bars illustrative`,
  `Nsight Compute, RTX 5060 Laptop GPU, median of three launches after thirty warm-ups; ${gainLow}x to ${gainHigh}x is naive time over vector time at ${gainTokens.map((t) => fmtInt(Number(t))).join(", ")} tokens (calculated); DRAM bytes of the vector kernel at 1 and 16 tokens: ${smallKb.join(" KB and ")} KB (measured); ${stepShare}% is the share of kernel time in a batch-1 decode step (measured); the row written and read back is ${roundTripKb} KB per layer at one token, ${roundTripShare}% of the ${weightsMb} MB of gate and up weights (calculated from config.json)`,
];

export const swigluSteps = steps;
export const swigluCaptions = captions;
export const swigluNumbers = {
  naive1,
  draft1,
  naiveBig,
  vectorBig,
  smallNaive,
  smallVector,
  smallFast,
  peakGbs,
  bandwidth: versions.map((v, k) => ({ label: v.label === "first draft" ? "draft" : v.label, gbs: gbs("4096", v.column), accent: k === 2 })),
};

export function SwigluStory() {
  return (
    <StickyStory
      steps={steps}
      caption={(i) => captions[i]}
      figure={(i) => {
        switch (i) {
          case 0:
            return <ZipFigure />;
          case 1:
            return <ChainFigure />;
          case 2:
            return <GridFigure naive={naive1} draft={draft1} />;
          case 3:
            return (
              <VectorFigure
                bars={swigluNumbers.bandwidth}
              />
            );
          case 4:
            return <SmFigure naive={smallNaive} vector={smallVector} dispatch={smallFast} />;
          default:
            return <TokensPicker />;
        }
      }}
    />
  );
}
