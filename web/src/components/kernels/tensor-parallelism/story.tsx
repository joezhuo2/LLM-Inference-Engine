"use client";

import Link from "next/link";
import { firstSplit, loadHistory, mib, mostRanks, perCopySpan, rankCopies, rankShare, shardedForwardCaption, shardedForwardSentences, split, tpFmt, tpShape, whole } from "@/content/kernels/tensor-parallelism";
import { about, msText } from "@/lib/round";
import { expectAnswer, type Prediction, Predict } from "../predict";
import { StickyStory, type StoryStep } from "../sticky-story";
import { ForwardFigure, LoadFigure, MatrixFigure, RankBytesFigure, SumsFigure, WireFigure } from "./figures";

const fmt = tpFmt;
const chapter = (slug: string, text: string) => (
  <Link href={`/kernels/${slug}`} className="link-underline text-accent">
    {text}
  </Link>
);
const s = tpShape;
const two = firstSplit;
const last = split[split.length - 1];
const twoLoad = loadHistory.rows.find((r) => r.n === two.n);
const lastLoad = loadHistory.rows.find((r) => r.n === last.n);
const wholeLoad = loadHistory.rows.find((r) => r.n === 0);
if (!twoLoad || !lastLoad || !wholeLoad) throw new Error("results.md Tensor parallelism: the second table has no row for the whole model or a split one");
const forward = shardedForwardSentences();
const half = 100 / two.n;

const prediction: Prediction = expectAnswer({
  question: `TinyLlama is cut across ${two.n} ranks. What share of the model's bytes does one rank hold?`,
  choices: ["less than half", "half", "more than half"],
  answer: two.calc.pct < half - 0.5 ? "less than half" : two.calc.pct > half + 0.5 ? "more than half" : "half",
  why: `${about(two.mib)} of ${about(whole.mib)} MiB, ${fmt(two.calc.pct)}% (measured). The layer matrices halve, but the embedding and the LM head, ${mib(rankShare(1).embed)} each, are whole in both ranks.`,
}, "more than half");

const steps: StoryStep[] = [
  {
    id: "heads",
    title: "attention is cut by heads",
    body: (
      <>
        <p>
          Attention already works in heads. TinyLlama has {s.heads} query heads, and every {s.group} of them share one of {s.kv_heads} K/V heads. A head never sees another head&apos;s numbers until the output projection mixes them. So the cut follows the heads: with N ranks, a rank gets {s.heads}/N query heads and {s.kv_heads}/N K/V heads, and each group stays together.
        </p>
        <p>
          In the weights a head is a band of rows. The engine keeps Q, K and V in one fused QKV weight, so a rank takes its rows of Q, of K and of V and stores them one after the other. The output projection wo reads the heads along its columns, so the rank takes the same heads&apos; columns of wo, and what wo gives is a <span className="text-accent">partial result</span> of the full width, {fmt(s.hidden)} numbers.
        </p>
        <p>The heads also set the limit. {s.kv_heads} K/V heads cannot be shared out to more than {s.kv_heads} ranks, so this model can be cut at most {mostRanks} ways, and plan_shard_layout() refuses any other count.</p>
      </>
    ),
  },
  {
    id: "channels",
    title: "the MLP is cut by channels",
    body: (
      <>
        <p>
          The MLP widens a token&apos;s {fmt(s.hidden)} numbers to {fmt(s.intermediate)} channels, applies SwiGLU and narrows them back. {chapter("swiglu", "SwiGLU")} works one channel at a time: channel c needs only the gate and up values of channel c. So a rank can take {fmt(s.intermediate / mostRanks)} channels of {fmt(s.intermediate)} and never need anyone else&apos;s.
        </p>
        <p>A rank takes its channels&apos; rows of the gate weight and of the up weight, stored together in the fused gate and up weight, and the same channels&apos; columns of wdown. Like wo, wdown then gives a partial result of the full width.</p>
        <p>
          The pattern is the same in both halves of the layer: the first matrix is <span className="text-accent">cut by rows</span>, the second <span className="text-accent">by columns</span>, and everything between them happens inside one rank.
        </p>
      </>
    ),
  },
  {
    id: "sums",
    title: "two sums in every layer",
    body: (
      <>
        <p>
          Why do the partial results just add? Each number a matrix product writes is a sum: one term for every column of the second matrix. Cut those columns into N groups and each rank adds up one group. The N partial sums together are the whole sum, added in a different order.
        </p>
        <p>
          That works only if each rank has the whole input to its columns, and the row cut gives it exactly that: a rank made the very heads, or channels, whose columns it holds. So the cut costs nothing between the two matrices and <span className="text-accent">one sum at the end of each half</span>, {s.sums} per token.
        </p>
        <p>The rest is not cut. RMSNorm needs the whole vector, so its weights are whole in every rank and it runs after the sum. The embedding and the LM head stay whole as well. A unit test adds up the partial outputs of 2 and 4 ranks on a small model and checks they equal the full layer&apos;s.</p>
      </>
    ),
  },
  {
    id: "forward",
    title: "a forward pass on the shards",
    body: (
      <>
        <p>
          With the weights cut, a forward pass on the shards goes like this. Every rank keeps its own K/V cache, for its own K/V heads, and its own scratch buffers. In each layer the shared RMSNorm runs first. Then every rank runs its QKV GEMM, RoPE, its K/V write, attention over its heads and its slice of wo, into a partial result.
        </p>
        <p>
          The all-reduce adds the partial results into the residual stream. The MLP repeats the pattern: the shared RMSNorm, each rank&apos;s gate and up GEMM, SwiGLU and wdown, and the second sum. After the last layer the whole LM head turns the vector into logits.
        </p>
        <p>
          On one GPU the ranks run one after another. The pass checks the cut and cannot be faster than the whole model: it does the same work in N smaller pieces.{forward.length ? ` ${forward.join(" ")}` : ""}
        </p>
      </>
    ),
  },
  {
    id: "bytes",
    title: "a rank holds more than its share",
    body: (
      <>
        <p>
          Cutting the layers divides most of the bytes, not all of them. A rank of {two.n} holds {about(two.mib)} MiB, <span className="text-accent">{fmt(two.calc.pct)}% of the model and not {fmt(half)}%</span>, and a rank of {last.n} holds {about(last.mib)} MiB, {fmt(last.calc.pct)}% and not {fmt(100 / last.n)}% (measured).
        </p>
        <p>
          The layer matrices do divide by N. The embedding and the LM head are each {fmt(s.vocab)} rows of {fmt(s.hidden)} numbers, {mib(rankShare(1).embed)}, and every rank holds both whole, so {mib(2 * rankShare(1).embed)} never shrinks. {two.n} ranks together hold {about(two.calc.together)} times the model and {last.n} ranks {about(last.calc.together)} times.
        </p>
        <p>Counting the bytes of every tensor from the shapes gives the same MiB the engine allocated. In the forward pass here only rank 0 embeds and makes the logits, so the other ranks could drop both tables. They still load them.</p>
        <Predict {...prediction} />
      </>
    ),
  },
  {
    id: "load",
    title: "a column slice is one copy per row",
    body: (
      <>
        <p>
          Loading a rank means copying its slices out of the file. A slice of rows is one piece of the file, one copy. A slice of columns is not: a matrix is stored row after row, so a rank&apos;s columns are a short piece of every row. wo and wdown have {fmt(s.hidden)} rows each, and a rank needs <span className="text-accent">{fmt(rankCopies(two.n))} copies</span> where the whole model needs {fmt(rankCopies(0))}.
        </p>
        <p>
          The first upload_weights gave every copy its own copy into pinned memory, its own transfer and its own event. That cost per copy made a rank load in {fmt(perCopySpan.min / 1000, 1)} to {fmt(perCopySpan.max / 1000, 1)} s, about {fmt(twoLoad.perCopy / wholeLoad.perCopy)} times what the whole model took then, {msText(wholeLoad.perCopy)}, for a fraction of the bytes (measured at {loadHistory.before}).
        </p>
        <p>
          The fix used the order of the copies: the row pieces of one tensor come one after another, and so do their destinations. upload_weights now fills one chunk with all of them and sends it in one call. A rank of {two.n} loads in {msText(twoLoad.packed)} and a rank of {last.n} in {msText(lastLoad.packed)}, against {msText(wholeLoad.packed)} for the whole model (measured at {loadHistory.after}).
        </p>
      </>
    ),
  },
  {
    id: "wire",
    title: "one GPU, and no wire",
    body: (
      <>
        <p>
          Everything in this chapter ran on one GPU. This laptop has no second device, and the engine has no NCCL, the library that moves data between NVIDIA GPUs. When the ranks share one GPU, a sum is an add in that GPU&apos;s memory.
        </p>
        <p>
          On real GPUs each of the {s.sums} sums of a token is a transfer: every rank sends its partial vector, gets the total back, and its next kernel waits for it. What that costs depends on the link between the GPUs, and <span className="text-accent">none of it is measured here</span>.
        </p>
        <p>What one GPU can show is the cut: which bytes each rank holds, how long a rank takes to load, and whether the partial results add up to the whole model&apos;s. {chapter("beyond", "Beyond this engine")} says what several GPUs would buy.</p>
      </>
    ),
  },
];

const source = "results.md, Tensor parallelism: the weights of one rank";
const captions = [
  `rows and columns as plan_shard_layout() in src/loader/tensor_parallel.cpp picks them; TinyLlama's shapes from config.json; boxes not to scale`,
  `rows and columns as plan_shard_layout() picks them; ${fmt(s.intermediate)} channels, ${fmt(s.hidden)} wide; boxes not to scale`,
  "the order of one layer; PartialOutputsAddUpToTheFullLayer in tests/unit/test_tensor_parallel.cpp checks the sums in double precision",
  `the structure of the sharded forward pass: what each rank keeps and where the two sums are; ${shardedForwardCaption()}`,
  `${source}: MiB on the device, every rank loaded onto the one GPU (measured); the parts calculated from the shapes; ${two.n} ranks together hold ${fmt(two.calc.together, 2)} times the model, ${last.n} ranks ${fmt(last.calc.together, 2)} times`,
  `${source}: upload ms, median of 21 uploads, one process per layout (measured at ${loadHistory.before} and ${loadHistory.after})`,
  `${s.layers} layers × 2 sums, calculated; the one GPU of hardware.md`,
];

export const tensorSteps = steps;
export const tensorCaptions = captions;

const figures = [() => <MatrixFigure part="attention" />, () => <MatrixFigure part="mlp" />, SumsFigure, ForwardFigure, RankBytesFigure, LoadFigure, WireFigure];

export function TensorStory() {
  return (
    <StickyStory
      steps={steps}
      caption={(i) => captions[i]}
      figure={(i) => {
        const Figure = figures[i] ?? WireFigure;
        return <Figure />;
      }}
    />
  );
}
