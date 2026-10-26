"use client";

import NumberFlow from "@number-flow/react";
import { useState } from "react";
import { measured } from "@/content/kernels/types";
import { cellNumber, decodeSteps } from "@/lib/data";
import { EasedBar } from "../eased-bar";
import { StickyStory, type StoryStep } from "../sticky-story";
import { Toggle } from "../toggle";
import { ChainFigure, GridFigure, peakGbs, SmFigure, VectorFigure, ZipFigure } from "./figures";

const fast = "fast-swiglu-branch-32";
const small = "small-batches-branch-33";

const tokenChoices = ["1", "16", "64", "512", "4096"];
const at = (tokens: string) => ({ tokens });

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

function TokensPicker() {
  const [tokens, setTokens] = useState("4096");
  const times = versions.map((v) => ({ ...v, us: measured(fast, at(tokens), v.column) }));
  const max = Math.max(...times.map((t) => t.us));
  const bandwidth = gbs(tokens, "fast (us)");
  return (
    <div className="flex h-full flex-col gap-3 font-mono text-xs">
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
          Every layer&apos;s MLP widens the hidden state from 2,048 to 5,632 values twice, with a gate projection and an up
          projection, mixes the two, and narrows it back down. The engine runs both projections as one GEMM, so each token
          comes out as a single row of 11,264 values: the gate half, then the up half.
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
          Hugging Face&apos;s act tensor bit for bit on all 22 layers.
        </p>
        <p>
          Every faster version had to keep that chain. Skipping the first rounding looks harmless and gives a different
          answer: in the example on the right the output moves by one BF16 step.
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
          Joe&apos;s first draft kept the naive kernel&apos;s work split, one thread per output value, on a 2D grid: 22
          blocks of 256 threads along a row of 5,632, and the row on blockIdx.y. It was bit-identical from the start and a
          little faster than naive at every size,{" "}
          <span className="font-mono text-text">{draft1.toFixed(1)} µs</span> against{" "}
          <span className="font-mono text-text">{naive1.toFixed(1)} µs</span> at one token.
        </p>
        <p>
          The 2D grid had a ceiling: gridDim.y stops at 65,535, so a batch with more rows would not launch. The engine never
          sends that many rows, but the spec promised any shape, and the tests now launch 70,000. Both kernels now run on one flat grid
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
          Each value is 2 bytes, and a thread that loads one at a time issues 24 memory instructions for 8 outputs. The
          vector kernel reads 8 gate values and 8 up values with two 16-byte loads and writes 8 outputs with one 16-byte
          store: 3 instructions for the same bytes, so the threads spend their time waiting on memory and not issuing loads.
        </p>
        <p>
          With enough tokens that is what the kernel is limited by. At 4,096 tokens it takes{" "}
          <span className="font-mono text-accent">{vectorBig.toFixed(0)} µs</span> against{" "}
          <span className="font-mono text-text">{naiveBig.toFixed(0)} µs</span> for naive, which is the GPU&apos;s measured
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
          A decode step has one token per request. At one token the 5,632 values make only 704 vectors, 3 blocks on a GPU
          with 26 SMs, and each thread runs its 8 exp and divide chains one after another. The vector kernel was no faster
          than naive there.
        </p>
        <p>
          So the kernel counts its work first: below half a block per SM, 4 tokens or fewer here, it takes the scalar path,
          whose 22 blocks spread the same chains over most of the GPU. At one token that is{" "}
          <span className="font-mono text-accent">{smallFast.toFixed(2)} µs</span> against{" "}
          <span className="font-mono text-text">{smallVector.toFixed(2)} µs</span> for the vector kernel in the same session.
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
          From 64 tokens up the vector kernel is 1.6x to 2x faster than naive, and the bandwidth line shows why the gain
          stops there. At 1 and 16 tokens the work is a few hundred KB and the launch itself sets the time. Try the token
          counts on the right.
        </p>
        <p>
          In a decode step SwiGLU is about 0.4% of the time, so none of this moves the token rate. The real fix would be to
          fold it into the gate/up GEMM&apos;s epilogue, where it would disappear as a kernel altogether; that is not done.
        </p>
      </>
    ),
  },
];

const captions = [
  "16 of 5,632 values per half drawn; the silu curve is exact",
  "the example is computed with the kernel's arithmetic; any g and u with the same roundings would do",
  "µs per launch for one token, Nsight Compute; 4 of the rows drawn",
  "µs and DRAM bytes from Nsight Compute at 4,096 tokens; the peak from the bandwidth test in docs/results.md",
  "µs per launch for one token, Nsight Compute, five runs; chain bars illustrative",
  "Nsight Compute, RTX 5060 Laptop GPU, median of three launches after thirty warm-ups",
];

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
                bars={versions.map((v, k) => ({ label: v.label === "first draft" ? "draft" : v.label, gbs: gbs("4096", v.column), accent: k === 2 }))}
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
