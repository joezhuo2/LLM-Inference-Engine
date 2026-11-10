"use client";

import Link from "next/link";
import { measured } from "@/content/kernels/types";
import { modelShape, traces } from "@/lib/data";
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
const shape = modelShape();
const headDim = shape.hidden / shape.heads;

const steps: StoryStep[] = [
  {
    id: "job",
    title: "one query against every cached key, every step",
    body: (
      <>
        <p>
          Each decode step adds one token to every sequence. Attention compares that token&apos;s query with the key of
          every token before it, turns the scores into weights with a softmax, and sums the values with those weights.
        </p>
        <p><span className="text-accent">{shape.heads}</span> query heads against a context that grows by one token every step.</p>
      </>
    ),
  },
  {
    id: "blocks",
    title: "every key is found through the block table",
    body: (
      <>
        <p>
          The KV cache is one pool of <span className="text-accent">16</span>-token blocks shared by every request. A sequence owns a list of blocks, its
          block table, and they can sit anywhere in the pool.
        </p>
        <p>So the kernel cannot read a context as one contiguous array: every key goes through the table first.</p>
      </>
    ),
  },
  {
    id: "reads",
    title: "pages move the reads, not the math",
    body: (
      <>
        <p>
          Inside a block, one KV head&apos;s 16 keys sit side by side, {headDim} values each, so a key is one run of{" "}
          <span className="text-accent">{headDim * 2}</span> bytes and the reads only jump at a block boundary. The table adds one read per key, an integer
          that the 16 keys of a block share.
        </p>
        <p>
          The scores, the softmax and the weighted sum are the arithmetic of ordinary attention, which is why the tests can
          check scattered block tables against a CPU reference. What the lookup costs in time was not measured on its own.
        </p>
      </>
    ),
  },
  {
    id: "v2",
    title: "an online softmax removed v1's context cap",
    group: "how it got built",
    body: (
      <>
        <p>
          The first kernel, v1, kept every score of the context in shared memory and made three passes over it, which capped
          the context at 2,048 tokens. The second, v2, streams keys in chunks with an online softmax, shares each key between
          a group of lanes and has no cap; its tests reach <span className="text-accent">8,192</span> tokens.
        </p>
        <p>
          Attention per decode step: <span className="text-accent">{stepV2} µs</span>, down from v1&apos;s time in the caption. How the online softmax works is the chapter{" "}
          <Link href="/kernels/attention-v2" className="link-underline text-accent">
            attention v2
          </Link>
          .
        </p>
      </>
    ),
  },
  {
    id: "split",
    title: "splitting the context filled an idle GPU",
    group: "how it got built",
    body: (
      <>
        <p>
          v2 still launches one block per row and query head, which at one row is a fifth of what the GPU holds at once. Cutting
          each context into ranges that run side by side, then merging them, took one row at a long context down to{" "}
          <span className="text-accent">{split8} µs</span>. The runner picks a split count every step and runs v2 when it picks one.
        </p>
        <p>
          The chapter{" "}
          <Link href="/kernels/split-attention" className="link-underline text-accent">
            split attention
          </Link>{" "}
          has the occupancy measurements, the merge and the CUDA Graphs fix.
        </p>
      </>
    ),
  },
];

const captions = [
  `decode attention for one sequence and one head of ${headDim} dimensions; the bars are illustrative`,
  Number.isFinite(poolBlocks) && poolBlocks > 0
    ? `example block table; the pool held ${poolBlocks.toLocaleString("en-US")} blocks in the recorded run`
    : "example block table",
  `the same block table; inside a block one KV head's 16 keys are 16 × ${headDim} BF16 values in a row, ${(16 * headDim * 2).toLocaleString("en-US")} bytes, as paged_attention_v2.cu lays them out`,
  `${stepV1} → ${stepV2} µs per decode step (eager profile, contexts of at most 64 tokens), from docs/profiling.md; v1 made three passes and held at most 2,048 tokens, v2 shares each key between eight lanes`,
  `${v2Loop} µs → ${split8} µs, one row at 2,048 tokens, from docs/profiling.md; one block per query head is ${shape.heads} blocks at batch 1; the busy SMs are an illustration`,
];

export const attentionSteps = steps;
export const attentionCaptions = captions;
export const attentionNumbers = { stepV1, stepV2, v2Loop, split8, poolBlocks };

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
          case 2:
            return <BlocksFigure />;
          case 3:
            return <V2LinkFigure v1={stepV1} v2={stepV2} />;
          default:
            return <SplitFigure v2={v2Loop} split={split8} />;
        }
      }}
    />
  );
}
