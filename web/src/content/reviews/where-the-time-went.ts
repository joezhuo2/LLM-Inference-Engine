import { eagerStage, firstStage, load, loadVerdict, next, nowStage, prefill, stageById, timeFmt, token } from "@/content/kernels/where-the-time-went";
import type { ChapterReview } from "./types";

const fmt = timeFmt;
const before = stageById("fast");
const after = stageById("v2");

export const review: ChapterReview = {
  lesson: "Measure where the time is before choosing what to change: most of a token was never in the kernels this book rewrote.",
  why: "It puts every chapter's gain on one scale, says what share of a token no kernel can remove, and ranks what is left by evidence.",
  takeaways: [
    `A batch-1 token takes ${fmt(token.ms, 2)} ms and ${fmt(token.floorPct)}% of it is the read of ${fmt(token.weightGb, 2)} GB of weights at the measured peak (calculated from measured values). The GEMMs took ${fmt(firstStage.groups.gemm)} µs of the first profile and ${fmt(nowStage.groups.gemm)} µs of today's: no chapter changed them.`,
    `The large wins were outside batch-1 decode: sharing the read across a batch, and prefill, where one ${prefill.long}-token prompt went from ${fmt(prefill.paged, 1)} to ${fmt(prefill.tiled, 1)} ms to its first token (measured).`,
    `Under load on ${load.date} (commit ${load.commit}): ${loadVerdict}, with both systems held to ${next.loadBatch} sequences (measured).`,
  ],
  questions: [
    {
      question: `In one run the kernels are busy for ${fmt(eagerStage.busy)} µs of a step, the step spans ${fmt(eagerStage.span)} µs, and a token takes ${fmt(eagerStage.tpot ?? 0, 2)} ms. Where are the two differences spent?`,
      answer: "Span minus busy is launch gaps on the GPU. Token minus span is host work between two steps.",
      why: "The span only runs from the first kernel to the last, so anything the host does between steps (scheduling, building the batch, sampling, reading the token back) is outside it. Inside the span, the GPU leaves a few microseconds before each kernel that is launched on its own.",
    },
    {
      recall: "foundations",
      question: `Attention v2 took attention from ${fmt(before?.groups.attention ?? 0)} to ${fmt(after?.groups.attention ?? 0)} µs per step. Why did a batch-1 token gain only a few percent?`,
      answer: `Because attention was a small slice: the GEMMs hold ${fmt(token.gemmShare)}% of the kernel time, and they are bound by the bytes of the weights.`,
      why: `Foundations gave the floor: ${fmt(token.weightGb, 2)} GB at ${token.peakGbs} GB/s is ${fmt(token.floorMs, 2)} ms whatever the arithmetic costs. A kernel that is not on that path can only win back its own slice.`,
    },
    {
      question: `Which of the three next steps could change a batch-1 token, and by how much at most?`,
      answer: `Preparing the next step while this one runs: ${fmt(next.hostMs, 2)} ms, ${fmt(next.hostPct)}% of the token (calculated).`,
      why: `The other two act on steps with many rows or on prompts. At batch 1 attention is a few percent of the step and the load limit does not apply, so what is left above the floor is mostly the time between steps.`,
    },
  ],
};
