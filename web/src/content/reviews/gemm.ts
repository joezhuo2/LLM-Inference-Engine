import { gemmOps, prefillRoof, roofline } from "@/content/kernels/gemm";
import { ceilings } from "@/lib/data";
import { about, msText } from "@/lib/round";
import { published } from "../published";
import { bigSide } from "../scale";
import type { ChapterReview } from "./types";

const peak = ceilings();
const ridge = (peak.tflops * 1e12) / (peak.gbs * 1e9);
const batchRows = 16;
const h100 = published.h100;
const projectionMs = (rows: number) => gemmOps.reduce((s, op) => s + roofline(op, String(rows), "us/step"), 0) / 1e3;
const figures = msText(projectionMs(batchRows)) === msText(projectionMs(1)) ? 3 : 2;

export const review: ChapterReview = {
  lesson: "At the memory roof, the only way to go faster is to move fewer bytes.",
  why: "It shows that the projections already use nearly all the available memory bandwidth, so improving decode means reading fewer weight bytes per token or sharing each read across requests.",
  takeaways: [
    "The wrapper asks cuBLAS for the transposed product so row-major weights and activations can stay where they are. No copy or transpose in memory is needed.",
    `Below the ridge at ${about(ridge)} FLOP per byte, memory traffic limits speed. A single decode row uses each weight once, so faster arithmetic does little.`,
    "Batching lets each weight read serve several token rows. The measured batch still waits on memory, so it produces more tokens in about the same projection time.",
    `Prefill is on the other side of the ridge. At ${prefillRoof.rows.toLocaleString("en-US")} rows each weight read feeds every row, ${about(prefillRoof.intensity)} FLOP per byte (calculated), and the measured prefill projections reach ${about(prefillRoof.pct)}% of the compute roof. There the tensor cores limit the time, not memory.`,
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
      question: `Foundations placed every op on a roofline. Why do the five projections take about as long at ${batchRows} rows, ${msText(projectionMs(batchRows), figures)} per step, as at one row, ${msText(projectionMs(1), figures)}?`,
      answer: `${batchRows} rows are still far under the ridge, so memory sets the time.`,
      why: `Each extra row adds FLOPs but no weight bytes, so at ${batchRows} rows gate/up does ${roofline("gate_up", String(batchRows), "FLOP/byte").toFixed(1)} FLOP per byte (calculated) against a ridge of ${about(ridge)}. On the foundations roofline that is still the bandwidth slope, where the time is the time to read the weights.`,
      recall: "foundations",
    },
    {
      question: `${bigSide.model} runs on an ${bigSide.gpu}, published at ${h100.bf16TFlops} TFLOP/s of dense BF16 and ${h100.bandwidthTBs} TB/s. A batch of 64 requests decodes together. Are its projections memory or compute bound?`,
      answer: `Memory bound: about 64 FLOP per byte against a ridge of ${Math.round(bigSide.ridge)}.`,
      why: `Each BF16 weight read feeds one multiply-add per row, 2 FLOPs per 2 bytes, so 64 rows do about 64 FLOP per byte. The ridge is ${h100.bf16TFlops} ÷ ${h100.bandwidthTBs} = ${bigSide.ridge.toFixed(1)} FLOP per byte, so it takes about ${Math.round(bigSide.ridge)} rows to turn compute bound, against about ${Math.round(ridge)} on this GPU. Calculated from published specs; nothing was measured on that GPU.`,
    },
  ],
};
