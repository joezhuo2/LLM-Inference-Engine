"use client";

import NumberFlow from "@number-flow/react";
import Link from "next/link";
import { useState } from "react";
import { chapterNumber } from "@/content/book-order";
import { measured } from "@/content/kernels/types";
import { ceilings, cellNumber, decodeSteps, modelShape } from "@/lib/data";
import { StickyStory, type StoryStep } from "../sticky-story";
import { EasedBar } from "../eased-bar";
import { Toggle } from "../toggle";
import { FirstFusedFigure, fusedKb64, fusionTokens, JobFigure, naiveKb64, PatternFigure, RoundingFigure, RoundTripFigure, ShuffleFigure, VectorFigure } from "./figures";
import { expectAnswer, type Prediction, Predict } from "../predict";

const fused = "fused-add-rmsnorm-branch-30";
const tuned = "tuning-add-rmsnorm-branch-39";
const wired = "fast-kernels-in-the-forward-pass-branch-36";

const tokenChoices = ["1", "16", "64"];
const at = (tokens: string) => ({ tokens });

const pair1 = measured(fused, at("1"), "naive total (us)");
const first1 = measured(fused, at("1"), "fast (us)");
const firstFlat = `${tokenChoices.map((t) => measured(fused, at(t), "fast (us)").toFixed(1)).join(" / ")} µs`;
const tunedPair1 = measured(tuned, at("1"), "naive pair (us)");
const tuned1 = measured(tuned, at("1"), "tuned fused (us)");

const kbPerToken = (kb: number) => (kb / fusionTokens).toFixed(1);
const stepKernels = (column: string) => measured(wired, { "decode step at batch 1": "kernels per step" }, column);
const kernelsNaive = stepKernels("naive kernels, eager");
const kernelsFast = stepKernels("fast kernels, eager");
const shape = modelShape();
const peak = ceilings();
const memoryUs = (measured(fused, at(String(fusionTokens)), "fast DRAM (KB)") * 1024 * 1e6) / (peak.gbs * 1e9);

const stepOps = (() => {
  const t = decodeSteps.b1;
  if (!t) return undefined;
  const rows = t.rows.filter((r) => r[0] === "add1_norm2" || r[0] === "add2_norm1");
  if (rows.length !== 2) return undefined;
  const us = rows.reduce((a, r) => a + (cellNumber(r[t.header.indexOf("GPU us/step")]) ?? Number.NaN), 0);
  const launches = rows.reduce((a, r) => a + (cellNumber(r[t.header.indexOf("kernels/step")]) ?? Number.NaN), 0);
  return Number.isFinite(us) && Number.isFinite(launches) ? { us, launches } : undefined;
})();

const versions = [
  { column: "naive pair (us)", label: "naive pair" },
  { column: "first fused (us)", label: "first fused" },
  { column: "tuned fused (us)", label: "tuned" },
];

const firstPair = measured(tuned, at("1"), "naive pair (us)");
const firstFused = measured(tuned, at("1"), "first fused (us)");
const fewer1 = Math.round((1 - measured(fused, at("1"), "fast DRAM (KB)") / measured(fused, at("1"), "naive DRAM (KB)")) * 100);
const prediction: Prediction = expectAnswer({
  question: `At 1 token the first fused kernel moves ${fewer1}% fewer DRAM bytes than the naive pair. How does its time compare?`,
  choices: ["about that much faster", "about the same", "slower"],
  answer: firstFused / firstPair < 0.85 ? "about that much faster" : firstFused / firstPair <= 1.1 ? "about the same" : "slower",
  why: `At 1 token it takes ${firstFused.toFixed(1)} µs against the pair's ${firstPair.toFixed(1)} µs. A few KB is nothing to DRAM, so dependent loads and two barriers set the time, not bytes. Compare the tuned kernel below it.`,
}, "about the same");

export function TokensPicker() {
  const [tokens, setTokens] = useState("1");
  const times = versions.map((v) => ({ ...v, us: measured(tuned, at(tokens), v.column) }));
  const max = Math.max(...times.map((t) => t.us));
  const bytes = [
    { label: "naive pair", kb: measured(fused, at(tokens), "naive DRAM (KB)") },
    { label: "fused", kb: measured(fused, at(tokens), "fast DRAM (KB)") },
  ];
  const maxKb = Math.max(...bytes.map((b) => b.kb));
  const tunedTime = times[2].us;
  return (
    <div className="flex h-full flex-col gap-3 tabular-nums text-xs">
      <Predict {...prediction} />
      <Toggle label="tokens" choices={tokenChoices} value={tokens} onChange={setTokens} />
      <div className="space-y-1.5 pt-1">
        <div className="text-muted">µs per launch, hidden 2048</div>
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
        <div className="text-muted">DRAM bytes per launch</div>
        {bytes.map((b, i) => (
          <div key={b.label} className="grid grid-cols-[5.5rem_1fr_4.5rem] items-center gap-2">
            <span className={i === 1 ? "text-accent" : "text-muted"}>{b.label}</span>
            <div className="h-3 min-w-0 rounded-full bg-border">
              <EasedBar fraction={b.kb / maxKb} className={i === 1 ? "bg-accent" : "bg-baseline/70"} />
            </div>
            <span className="text-right tabular-nums text-muted">
              <NumberFlow value={b.kb} suffix=" KB" format={{ minimumFractionDigits: 1, maximumFractionDigits: 1 }} />
            </span>
          </div>
        ))}
      </div>
      <div className="mt-auto space-y-1 border-t border-border pt-2 leading-relaxed">
        <div>
          <span className="text-muted">tuned against the pair </span>
          <span className="text-accent tabular-nums">{`${(times[0].us / tunedTime).toFixed(1)}x faster`}</span>
        </div>
        {stepOps && (
          <div>
            <span className="text-muted">in a batch-1 decode step now </span>
            <span className="text-text tabular-nums">{`${stepOps.us.toFixed(1)} µs over ${stepOps.launches} launches`}</span>
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
          A transformer layer never replaces its input; it adds to it. After attention, and again after the MLP, the
          layer&apos;s output is added to the residual stream, and that sum is what the next block reads.
        </p>
        <p>
          The next block first normalizes it with RMSNorm: divide the row by the root of its mean square, then scale by a
          learned weight per dimension. So every add is followed by a norm of the same row.
        </p>
      </>
    ),
  },
  {
    id: "fusion",
    title: "fusion, the pattern",
    body: (
      <>
        <p>
          Every separate elementwise kernel reads its inputs from DRAM, does a little arithmetic, writes its output back
          and pays a launch. Two in a row send the value between them out to DRAM and read it straight back.
        </p>
        <p>
          Fusing the two keeps that value in registers. The bytes of the round trip and one launch disappear, and the
          arithmetic is the same. At decode the math was never the cost, so what fusion saves is bytes and the gap before a
          launch, not arithmetic.
        </p>
        <p>
          Here the fused kernel moves <span className="text-accent">{kbPerToken(fusedKb64)} KB</span> per token where the pair moves
          more (calculated from the bytes per launch), and a decode step runs fewer kernels (measured). Fewer bytes still did not make
          the first fused kernel faster: at one token it took about the same time as the pair. The errata entry &quot;
          <Link href="/kernels/dead-ends" className="link-underline text-accent">
            Fusing two kernels saved bytes, not time
          </Link>
          &quot; tells that story.
        </p>
        <p>
          The same pattern returns in SwiGLU, chapter {chapterNumber("swiglu")}. CUDA Graphs, a later chapter, remove the
          launch part of the cost for every kernel at once.
        </p>
      </>
    ),
  },
  {
    id: "round-trip",
    title: "two kernels, one round trip",
    body: (
      <>
        <p>
          The naive version runs them as two kernels. add writes the sum to memory; rmsnorm then reads it back, twice,
          once for the squares and once to scale it.
        </p>
        <p>
          A fused kernel reads x, the residual and the weight once, writes the sum and the norm once, and keeps the sum on
          the chip in between. The DRAM traffic falls to <span className="text-accent">{fusedKb64} KB</span> per launch, and one launch replaces two.
        </p>
      </>
    ),
  },
  {
    id: "first",
    title: "fused, and no faster",
    body: (
      <>
        <p>
          The first fused kernel runs one block of 256 threads per row. Each thread adds and squares a few values, the block
          reduces the squares, one thread computes rstd and shares it, and a second pass reads the sums back and writes the
          norm.
        </p>
        <p>
          It moves a third fewer bytes, but takes the same time as the pair:{" "}
          <span className="text-accent">{first1.toFixed(1)} µs</span> for one token. The memory traffic is a small part of
          that; the rest is waiting on dependent loads and barriers.
        </p>
      </>
    ),
  },
  {
    id: "rounding",
    title: "the same bits as the naive pair",
    body: (
      <>
        <p>
          The spec asks for a sum bit-identical to the naive add, and a norm equal to the naive RMSNorm except for the
          order the squares are added in. That pins down where every rounding to BF16 happens.
        </p>
        <p>
          Two review fixes came from it: square the sum after rounding it, since the naive kernel only ever sees the
          rounded value, and round the normalized value before multiplying by the weight, as the naive kernel does.
        </p>
      </>
    ),
  },
  {
    id: "vectors",
    title: "16 bytes at a time",
    body: (
      <>
        <p>
          A row is split into pieces of 16 bytes, one per thread. The tuned kernel loads each piece with one{" "}
          <span className="text-accent">128-bit</span> instruction instead of a separate small load for every value, for x and the residual alike.
        </p>
        <p>
          It also keeps the rounded sums in registers, so the second pass never goes back to memory for them, and writes
          the sum and the norm with 128-bit stores.
        </p>
      </>
    ),
  },
  {
    id: "shuffles",
    title: "every thread gets the total",
    body: (
      <>
        <p>
          Adding up the squares used to end with the total in one thread, which then shared rstd through shared memory
          behind a second barrier.
        </p>
        <p>
          With xor shuffles, each lane swaps partial sums with the lane whose index differs in one bit. After a few swaps
          all the lanes of a warp hold the warp&apos;s total; after <span className="text-accent">one barrier</span> and one more round over the warps, every thread holds
          the row&apos;s total and computes rstd itself.
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
          The tuned kernel takes <span className="text-accent">{tuned1.toFixed(1)} µs</span> for one token, about a third of the pair&apos;s time, and stays near that up to the largest token count measured.
          Try the token counts on the right.
        </p>
        <p>
          In a decode step, the add + RMSNorm launches are now about 1% of the time. The gain only became real once the
          kernel was faster than the pair it replaced, not just one launch fewer.
        </p>
      </>
    ),
  },
];

const captions = [
  "one row; the values and the square bars are illustrative",
  `an illustration of any two elementwise kernels in a row; per-token bytes are calculated from the Nsight Compute bytes per launch at ${fusionTokens} tokens: the pair moves ${kbPerToken(naiveKb64)} KB per token against ${kbPerToken(fusedKb64)} KB; kernels per step from the nsys profile: ${kernelsFast} instead of ${kernelsNaive}; at one token the first fused kernel takes ${first1.toFixed(1)} µs against the pair's ${pair1.toFixed(1)} µs`,
  `DRAM bytes per launch at 64 tokens from Nsight Compute: ${naiveKb64} KB → ${fusedKb64} KB`,
  `µs per launch for one token (Nsight Compute): the pair ${pair1.toFixed(1)}, the first fused kernel ${first1.toFixed(1)}; each of the 256 threads handles 8 values; ${fusionTokens} rows move ${fusedKb64} KB, ${memoryUs.toFixed(1)} µs of DRAM time at the measured ${peak.gbs} GB/s (calculated); phase widths are illustrative, not measured`,
  "the order of operations required by the spec and checked by the tests",
  `a row of ${shape.hidden.toLocaleString("en-US")} BF16 values is ${(shape.hidden * 2) / 16} pieces of 16 bytes; 8 threads of 256 shown; one load = 8 values = 16 bytes, where 2-byte loads would take 8`,
  "8 lanes and 3 swaps shown; a warp has 32 lanes and does 5; a block of 256 threads is 8 warps",
  `Nsight Compute, hidden 2048, RTX 5060 Laptop GPU; times and bytes from two separate runs; at one token the pair takes ${tunedPair1.toFixed(1)} µs, and the tuned kernel stays near ${tuned1.toFixed(1)} µs up to 64 rows`,
];

export const addRmsnormSteps = steps;
export const addRmsnormCaptions = captions;
export const addRmsnormNumbers = { pair: pair1, first: first1, firstFlat, naiveKb: naiveKb64, fusedKb: fusedKb64 };

export function AddRmsnormStory() {
  return (
    <StickyStory
      steps={steps}
      caption={(i) => captions[i]}
      figure={(i) => {
        switch (i) {
          case 0:
            return <JobFigure />;
          case 1:
            return <PatternFigure />;
          case 2:
            return <RoundTripFigure naiveKb={naiveKb64} fusedKb={fusedKb64} />;
          case 3:
            return <FirstFusedFigure pair={pair1} first={first1} flat={firstFlat} />;
          case 4:
            return <RoundingFigure />;
          case 5:
            return <VectorFigure />;
          case 6:
            return <ShuffleFigure />;
          default:
            return <TokensPicker />;
        }
      }}
    />
  );
}
