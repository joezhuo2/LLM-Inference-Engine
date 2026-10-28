"use client";

import NumberFlow from "@number-flow/react";
import { useState } from "react";
import { measured } from "@/content/kernels/types";
import { cellNumber, decodeSteps } from "@/lib/data";
import { StickyStory, type StoryStep } from "../sticky-story";
import { EasedBar } from "../eased-bar";
import { Toggle } from "../toggle";
import { FirstFusedFigure, fusedKb64, JobFigure, naiveKb64, RoundingFigure, RoundTripFigure, ShuffleFigure, VectorFigure } from "./figures";

const fused = "fused-add-rmsnorm-branch-30";
const tuned = "tuning-add-rmsnorm-branch-39";

const tokenChoices = ["1", "16", "64"];
const at = (tokens: string) => ({ tokens });

const pair1 = measured(fused, at("1"), "naive total (us)");
const first1 = measured(fused, at("1"), "fast (us)");
const firstFlat = `${tokenChoices.map((t) => measured(fused, at(t), "fast (us)").toFixed(1)).join(" / ")} µs`;
const tunedPair1 = measured(tuned, at("1"), "naive pair (us)");
const tuned1 = measured(tuned, at("1"), "tuned fused (us)");

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

function TokensPicker() {
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
    <div className="flex h-full flex-col gap-3 font-mono text-xs">
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
          the chip in between. At 64 tokens that is <span className="font-mono text-text">{naiveKb64}</span> to{" "}
          <span className="font-mono text-accent">{fusedKb64} KB</span> of DRAM traffic, and one launch instead of two.
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
          The first fused kernel runs one block of 256 threads per row. Each thread adds and squares 8 values, the block
          reduces the squares, thread 0 computes rstd and shares it, and a second pass reads the sums back and writes the
          norm.
        </p>
        <p>
          It moves a third fewer bytes, but takes the same time as the pair:{" "}
          <span className="font-mono text-text">{pair1.toFixed(1)}</span> against{" "}
          <span className="font-mono text-text">{first1.toFixed(1)} µs</span> for one token. 64 rows of 4 KB are under 2 µs of
          memory traffic; the rest is waiting on dependent loads and two barriers.
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
          A row of 2,048 BF16 values is 256 pieces of 16 bytes, one per thread. The tuned kernel loads each piece with one
          128-bit instruction instead of 8 separate 2-byte loads, for x and the residual alike.
        </p>
        <p>
          It also keeps the 8 rounded sums in registers, so the second pass never goes back to memory for them, and writes
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
          With xor shuffles, each lane swaps partial sums with the lane whose index differs in one bit. After five swaps
          all 32 lanes hold the warp&apos;s total; after one barrier and one more round over the 8 warps, every thread holds
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
          The tuned kernel takes <span className="font-mono text-accent">{tuned1.toFixed(1)} µs</span> for one token against{" "}
          <span className="font-mono text-text">{tunedPair1.toFixed(1)} µs</span> for the pair, and stays near that up to 64 rows.
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
  `DRAM bytes per launch at 64 tokens from Nsight Compute: ${naiveKb64} KB → ${fusedKb64} KB`,
  `µs per launch for one token (Nsight Compute); phase widths are illustrative, not measured`,
  "the order of operations required by the spec and checked by the tests",
  "8 threads of 256 shown; one load = 8 values = 16 bytes",
  "8 lanes and 3 swaps shown; a warp has 32 lanes and does 5",
  "Nsight Compute, hidden 2048, RTX 5060 Laptop GPU; times and bytes from two separate runs",
];

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
            return <RoundTripFigure naiveKb={naiveKb64} fusedKb={fusedKb64} />;
          case 2:
            return <FirstFusedFigure pair={pair1} first={first1} flat={firstFlat} />;
          case 3:
            return <RoundingFigure />;
          case 4:
            return <VectorFigure />;
          case 5:
            return <ShuffleFigure />;
          default:
            return <TokensPicker />;
        }
      }}
    />
  );
}
