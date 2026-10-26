"use client";

import NumberFlow from "@number-flow/react";
import { useState } from "react";
import { measured } from "@/content/kernels/types";
import { cellNumber, decodeSteps } from "@/lib/data";
import { EasedBar } from "../eased-bar";
import { StickyStory, type StoryStep } from "../sticky-story";
import { Toggle } from "../toggle";
import { BitsFigure, LayoutFigure, ParallelFigure, RotateFigure, SerialFigure } from "./figures";

const fast = "fast-rope-branch-31";

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

function TokensPicker() {
  const [tokens, setTokens] = useState("1");
  const times = versions.map((v) => ({ ...v, us: measured(fast, at(tokens), v.column) }));
  const max = Math.max(...times.map((t) => t.us));
  const kb = measured(fast, at(tokens), "DRAM, fast (KB)");
  const fastTime = times[2].us;
  return (
    <div className="flex h-full flex-col gap-3 font-mono text-xs">
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
          Each head&apos;s 64 values are read as 32 pairs, and pair i is turned by the angle position · θᵢ. A query at
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
          The QKV projection leaves one row of 2,560 values per token: 32 query heads, then 4 key heads, then 4 value
          heads. RoPE turns the 36 Q and K heads in place, right before the keys are written to the cache, and leaves V
          alone.
        </p>
        <p>
          Inside a head, value i pairs with value i + 32, not with its neighbour; that is how Hugging Face&apos;s
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
          The frequencies were the trap. A device powf differs from the host&apos;s in the last bit for 2 of the 32, the
          angle multiplies that error by the position, and near 2,047 a sine lands on the other side of a BF16 rounding
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
          Joe&apos;s first draft gave each head one thread: 128 threads per token, of which 36 had a head, each walking its
          32 pairs one after another and computing every sine and cosine again for every head.
        </p>
        <p>
          Correct once the frequencies came from the host, but a flat{" "}
          <span className="font-mono text-text">{draftFlat}</span> at 1 to 512 tokens, against{" "}
          <span className="font-mono text-text">{naive1.toFixed(1)} µs</span> for the naive kernel at one token. The time
          was the chain of 32 pairs per thread, not the memory: only at 512 tokens did the naive kernel get slower (
          <span className="font-mono text-text">{naive512.toFixed(1)}</span> against{" "}
          <span className="font-mono text-text">{draft512.toFixed(1)} µs</span>).
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
          The fast kernel turns the mapping around: 32 threads per head group, one per pair, and 8 groups, 256 threads per
          token. Each thread computes the cosine and sine of its pair once and applies them to 4 or 5 heads, so the 36
          heads no longer wait for each other.
        </p>
        <p>
          That is 256 sines and cosines per token instead of 1,152, and{" "}
          <span className="font-mono text-accent">{fast1.toFixed(1)} µs</span> for one token. Neighbouring threads read
          neighbouring pairs, so a warp&apos;s 2-byte loads already coalesce into a few contiguous sectors; the 128-bit loads of the
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
          Up to 64 tokens the fast kernel stays near {fast1.toFixed(1)} µs: a few hundred KB at most, far from the memory
          limit, so the launch and one chain of load, sine and store set the time. At 512 tokens it is{" "}
          <span className="font-mono text-accent">{fast512.toFixed(1)} µs</span>, half the naive kernel&apos;s. Try the
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
  "8 of the 32 pairs of one head, at their real rates; the vector lengths are illustrative",
  "TinyLlama's layout: 32 query and 4 key/value heads of 64",
  "the chain of the bug found while writing the naive kernel; bit patterns illustrative",
  `µs per launch for one token, Nsight Compute; the pair segments are an illustration, not measured`,
  `µs per launch for one token, Nsight Compute; sin and cos counts from the code`,
  "Nsight Compute, RTX 5060 Laptop GPU, one launch after ten warm-ups",
];

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
