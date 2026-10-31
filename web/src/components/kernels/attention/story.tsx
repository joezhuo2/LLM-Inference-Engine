"use client";

import Link from "next/link";
import { measured } from "@/content/kernels/types";
import { traces } from "@/lib/data";
import { StickyStory, type StoryStep } from "../sticky-story";
import { BlocksFigure, LanesFigure, OnlineSoftmaxFigure, OverviewFigure, SplitFigure, V1Figure } from "./figures";

const split = "split-context-attention-branch-40";
const v2Section = "paged-attention-v2-branch-37";
const at2048 = { rows: "1", context: "2,048" };
const v1At2048 = measured(v2Section, at2048, "v1 (us)");
const v2At2048 = measured(v2Section, at2048, "v2 (us)");
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
    id: "v1",
    title: "v1: every score in shared memory",
    body: (
      <>
        <p>
          One thread block per row and head. It writes a score for every token into shared memory, finds the maximum,
          takes the exponentials and their sum, then multiplies by the values: three passes.
        </p>
        <p>
          Shared memory has a fixed size, so the context does too: v1 stops at 2,048 tokens. One row at 2,048 tokens:{" "}
          <span className="font-mono text-text">{v1At2048} µs</span>.
        </p>
      </>
    ),
  },
  {
    id: "v2",
    title: "v2: one chunk at a time",
    body: (
      <>
        <p>
          The online softmax keeps three things between chunks: the largest score so far, the sum of exponentials, and
          the weighted values. When a chunk brings a larger score, the old sum and values are scaled down by
          exp(m_old − m_new) and the new chunk is added.
        </p>
        <p>
          The answer is the same as the three-pass version, but only one 256-token chunk is ever in shared memory, so
          contexts of 8,192 tokens run.
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
          In v1 each thread computed a whole dot product alone. In v2 a group of 8 lanes shares one key: at every step
          they read 8 neighbouring dimensions together, one 16-byte read, and three shuffles add their 8 partial sums.
        </p>
        <p>
          With the softmax change and every thread helping with the values, one row at 2,048 tokens went from{" "}
          <span className="font-mono text-text">{v1At2048}</span> to{" "}
          <span className="font-mono text-accent">{v2At2048} µs</span>.
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
  `one cell of the context = 64 tokens; ${v1At2048} µs for one row at 2,048 tokens`,
  "example scores; m only ever rises, and every rise rescales what came before",
  "lane j reads dims j, j + 8, j + 16 … so each step's 8 reads are side by side in memory",
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
            return <V1Figure />;
          case 3:
            return <OnlineSoftmaxFigure />;
          case 4:
            return <LanesFigure />;
          default:
            return <SplitFigure v2={v2Loop} split={split8} />;
        }
      }}
    />
  );
}
