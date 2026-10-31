"use client";

import Link from "next/link";
import { measured } from "@/content/kernels/types";
import { traces } from "@/lib/data";
import { StickyStory, type StoryStep } from "../sticky-story";
import { V2LinkFigure } from "../attention-v2/figures";
import { BlocksFigure, OverviewFigure, SplitFigure } from "./figures";

const split = "split-context-attention-branch-40";
const at2048 = { rows: "1", context: "2,048" };
const wired = "paged-attention-v2-in-the-forward-pass-branch-38";
const attentionRow = { "": "attention per decode step (eager profile, context at most 64)" };
const stepV1 = measured(wired, attentionRow, "before (e07c022)");
const stepV2 = measured(wired, attentionRow, "after (79c1f91)");
const v2Loop = measured(split, at2048, "v2");
const split8 = measured(split, at2048, "8");
const poolBlocks = Number((traces.find((t) => t.meta.prompts && t.meta.prompts !== "random") ?? traces[0])?.meta.kv_blocks);

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
    id: "v2",
    title: "v1 → v2",
    body: (
      <>
        <p>
          The first version kept every score of the context in shared memory and made three passes over it, with a
          context cap of 2,048 tokens. The second streams keys in chunks with an online softmax, shares each key between
          eight lanes, and runs 8,192 tokens.
        </p>
        <p>
          Attention per decode step: <span className="font-mono text-text">{stepV1}</span> to{" "}
          <span className="font-mono text-accent">{stepV2} µs</span>. The whole before and after is its own story:{" "}
          <Link href="/kernels/attention-v2" className="link-underline text-accent">
            attention v1 → v2
          </Link>
          .
        </p>
      </>
    ),
  },
  {
    id: "split",
    title: "then: split the context",
    body: (
      <>
        <p>
          v2 still launches one block per row and head, 32 blocks at batch 1, a fifth of what the GPU holds at once. Cutting
          each context into ranges that run side by side, then merging them, took one row at 2,048 tokens from{" "}
          <span className="font-mono text-text">{v2Loop}</span> to <span className="font-mono text-accent">{split8} µs</span>.
        </p>
        <p>
          That is its own story:{" "}
          <Link href="/kernels/split-attention" className="link-underline text-accent">
            split-context attention
          </Link>
          , with the occupancy measurements, the merge and the CUDA Graphs fix.
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
  `${stepV1} → ${stepV2} µs per decode step (eager profile, contexts of at most 64 tokens), from docs/profiling.md`,
  `${v2Loop} µs → ${split8} µs, one row at 2,048 tokens, from docs/profiling.md; the busy SMs are an illustration`,
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
            return <V2LinkFigure v1={stepV1} v2={stepV2} />;
          default:
            return <SplitFigure v2={v2Loop} split={split8} />;
        }
      }}
    />
  );
}
