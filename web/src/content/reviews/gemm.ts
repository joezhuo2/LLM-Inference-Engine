import { gemmOps, prefillRoof, roofline } from "@/content/kernels/gemm";
import { ceilings } from "@/lib/data";
import type { ChapterReview } from "./types";

const peak = ceilings();
const ridge = (peak.tflops * 1e12) / (peak.gbs * 1e9);
const batchRows = 16;
const projectionUs = (rows: number) => Math.round(gemmOps.reduce((s, op) => s + roofline(op, String(rows), "us/step"), 0)).toLocaleString("en-US");

export const review: ChapterReview = {
  lesson: "At the memory roof, the only way to go faster is to move fewer bytes.",
  why: "It shows that the projections already use nearly all the available memory bandwidth, so improving decode means reading fewer weight bytes per token or sharing each read across requests.",
  takeaways: [
    "The wrapper asks cuBLAS for the transposed product so row-major weights and activations can stay where they are. No copy or transpose in memory is needed.",
    `Below the ridge at ${Math.round(ridge)} FLOP per byte, memory traffic limits speed. A single decode row uses each weight once, so faster arithmetic does little.`,
    "Batching lets each weight read serve several token rows. The measured batch still waits on memory, so it produces more tokens in about the same projection time.",
    `Prefill is on the other side of the ridge. At ${prefillRoof.rows.toLocaleString("en-US")} rows each weight read feeds every row, ${Math.round(prefillRoof.intensity)} FLOP per byte (calculated), and the measured prefill projections reach ${prefillRoof.tflops.toFixed(1)} TFLOP/s, ${Math.round(prefillRoof.pct)}% of the ${peak.tflops} TFLOP/s compute roof. There the tensor cores limit the time, not memory.`,
  ],
  questions: [
    {
      question: "If the GPU's arithmetic gets faster but its memory bandwidth stays the same, should a projection for a single decode row get much faster?",
      answer: "No, it still waits on memory.",
      why: "Each weight is read for just a multiply-add, so the arithmetic costs little compared with fetching the weights. Faster arithmetic does not make those bytes arrive sooner.",
    },
    {
      question: "If more requests decode together, should attention gain the same reuse of reads as the projections?",
      answer: "No, attention reads each request's own keys and values.",
      why: "The projections share weights across requests, so each weight read can serve more token rows. Attention still reads separate keys and values for every row.",
    },
    {
      question: `Foundations placed every op on a roofline. Why do the five projections take about as long at ${batchRows} rows, ${projectionUs(batchRows)} µs per step, as at one row, ${projectionUs(1)} µs?`,
      answer: `${batchRows} rows are still far under the ridge, so memory sets the time.`,
      why: `Each extra row adds FLOPs but no weight bytes, so at ${batchRows} rows gate/up does ${roofline("gate_up", String(batchRows), "FLOP/byte").toFixed(1)} FLOP per byte (calculated) against a ridge of ${Math.round(ridge)}. On the foundations roofline that is still the bandwidth slope, where the time is the time to read the weights.`,
      recall: "foundations",
    },
  ],
};
