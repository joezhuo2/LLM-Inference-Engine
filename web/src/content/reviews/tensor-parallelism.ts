import { firstSplit, loadHistory, mib, mostRanks, rankCopies, rankShare, tpFmt, tpShape, whole } from "@/content/kernels/tensor-parallelism";
import type { ChapterReview } from "./types";

const fmt = tpFmt;
const s = tpShape;
const two = firstSplit;
const twoLoad = loadHistory.rows.find((r) => r.n === two.n);
if (!twoLoad) throw new Error(`results.md Tensor parallelism: no load of a rank of ${two.n}`);

export const review: ChapterReview = {
  lesson: "Cut the first matrix of each half of a layer by rows and the second by columns, and the ranks' partial results add up to the whole layer's: two sums per layer.",
  why: "It is how a model larger than one GPU is served, and it shows which bytes divide across GPUs and which do not.",
  takeaways: [
    `Attention is cut by heads and the MLP by channels: rows of the fused QKV and gate and up weights, columns of wo and wdown. TinyLlama's ${s.kv_heads} K/V heads allow at most ${mostRanks} ranks.`,
    `A rank of ${two.n} holds ${fmt(two.mib, 1)} MiB, ${fmt(two.calc.pct)}% of the model and not half (measured): the embedding and the LM head, ${mib(rankShare(1).embed)} each, stay whole in every rank.`,
    `A column slice is one copy per row, ${fmt(rankCopies(two.n))} copies against ${fmt(rankCopies(0))}; packing the copies of one tensor into one chunk took a rank of ${two.n} from ${fmt(twoLoad.perCopy)} to ${fmt(twoLoad.packed)} ms (measured).`,
  ],
  questions: [
    {
      question: `With ${mostRanks} ranks, how many query heads and how many K/V heads does one rank hold?`,
      answer: `${s.heads / mostRanks} query heads and ${s.kv_heads / mostRanks} K/V ${s.kv_heads / mostRanks === 1 ? "head" : "heads"}: ${s.heads} and ${s.kv_heads}, each divided by ${mostRanks}.`,
      why: `The ${s.group} query heads that share one K/V head stay in the same rank, so a rank never needs another rank's keys or values.`,
    },
    {
      recall: "swiglu",
      question: "Why can the MLP be cut between the gate and up GEMM and wdown without any sum in between?",
      answer: "SwiGLU works one channel at a time, so a rank can finish its own channels alone.",
      why: "Channel c of the activation needs only the gate and up values of channel c. The rows a rank holds of gate and up make exactly the channels whose columns of wdown it holds.",
    },
    {
      question: `Two ranks of ${fmt(two.mib, 1)} MiB each hold more than the ${fmt(whole.mib, 1)} MiB model. Where do the extra bytes come from?`,
      answer: `Both ranks hold the embedding and the LM head whole: ${fmt(two.calc.together, 2)} times the model in all.`,
      why: "Only the layer matrices are cut. The two vocabulary-sized tables are copied into every rank.",
    },
  ],
};
