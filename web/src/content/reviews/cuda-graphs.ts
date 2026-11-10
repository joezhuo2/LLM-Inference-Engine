import { kvReplay, launchGaps } from "@/lib/data";
import trace from "@/content/kernels/cuda-graphs-trace.json";
import { padding, sweepKnee, sweepOps, sweepPoints } from "@/content/kernels/cuda-graphs";
import { chapterNumber } from "../book-order";
import type { ChapterReview } from "./types";

const eager = launchGaps("eager");
const replay = launchGaps("graphs");
const blockSize = kvReplay().meta.block_size;
const gap = trace.eager.gap_us.p50;
const lengths = trace.eager.kernels_in_layer.map((k) => k.end - k.start);
const shortest = Math.min(...lengths);
const longest = Math.max(...lengths);
const idleShare = (us: number) => (100 * gap) / (us + gap);
const pct = (v: number) => (v < 10 ? v.toFixed(1) : Math.round(v).toString());
const kneePoint = sweepPoints.find((p) => p.batch === sweepKnee) ?? sweepPoints[0];
const lastPoint = sweepPoints[sweepPoints.length - 1];
const afterKnee = Math.round(100 * (lastPoint.graphs / kneePoint.graphs - 1));
const halfStep = sweepOps.find((o) => o.share >= 50) ?? sweepOps[sweepOps.length - 1];

export const review: ChapterReview = {
  lesson: "When kernels are short, the gaps between launches cost time: record the step once and replay it.",
  why: "It shows that a batch-1 step can leave the GPU waiting even when every kernel is fast, so the speed of a step depends on how many launches it needs, not only on how long each kernel runs.",
  takeaways: [
    `Launched one by one, every kernel is preceded by a gap of a few microseconds while the GPU takes the next launch and sets up its grid. Those gaps leave the GPU idle for ${eager.idle_pct}% of a batch-1 decode step, and faster kernels would not remove them.`,
    `A CUDA Graph records the chain of launches once and replays it with a single launch, which cuts the idle share to ${replay.idle_pct}% while the kernels take the same time. The saving is about a fixed time per step, because every step has the same kernel boundaries.`,
    `A graph freezes each kernel's pointers and grid, so a replay can only change values. Inputs go through fixed buffers, and anything that changes a grid, like the split count of attention, needs its own graph. Batches are padded to a bucket, and padded rows are still computed, so the buckets come ${padding.steps} per power of two: padding ${padding.rows} rows to ${padding.coarseBucket} instead of ${padding.fineBucket} costs about ${padding.costPct}% more time per token.`,
  ],
  questions: [
    {
      question: `Launched one by one, each kernel is preceded by a gap of about ${gap} µs. Which loses a bigger share of its time to that gap: the longest kernel in a layer (${longest.toFixed(0)} µs) or the shortest (${shortest.toFixed(1)} µs)?`,
      answer: "The shortest.",
      why: `The gap is about the same before every kernel, so its share is the gap divided by the kernel plus the gap: about ${pct(idleShare(shortest))}% for the shortest kernel and ${pct(idleShare(longest))}% for the longest. Short kernels are where launches hurt, and a graph removes the gap without touching the kernels.`,
    },
    {
      question: `With graphs, the max batch goes from ${sweepKnee} to ${lastPoint.batch} rows, ${lastPoint.batch / sweepKnee} times as many. Does the throughput rise by about as much?`,
      answer: "No, by very little.",
      why: `It rises by about ${afterKnee}% (calculated from the measured medians of an offline run). Up to ${sweepKnee} rows a bigger batch shares each read of the weights among more rows. Past that the GEMMs are compute bound and attention, which reads each row's own keys and values, is ${halfStep.share}% of the step at ${halfStep.batch} rows, so a step with twice the rows takes about twice as long.`,
    },
    {
      question: `Chapter ${chapterNumber("paged-kv")} finds a request's cached tokens through a block table. A CUDA Graph freezes every kernel's pointers. Why can one captured graph still serve a request whose context grows every step?`,
      answer: "Each step changes values in the staged table and slots, never an address.",
      why: `The kernels read the cache pool and the staged block table at fixed addresses. Token pos goes to slot table[pos / ${blockSize}] · ${blockSize} + pos mod ${blockSize}, and stage writes the new table, slots and context length into the same buffer before each replay, so the same pointers reach different blocks. A request gets a new block only every ${blockSize} tokens.`,
      recall: "paged-kv",
    },
  ],
};
