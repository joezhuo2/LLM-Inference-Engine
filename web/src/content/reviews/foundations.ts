import { ceilings, modelShape } from "@/lib/data";
import type { ChapterReview } from "./types";

const m = modelShape();
const peak = ceilings();
const headDim = m.hidden / m.heads;
const layerParams = m.hidden * (m.hidden + 2 * m.kv_heads * headDim) + m.hidden * m.hidden + m.hidden * 2 * m.intermediate + m.intermediate * m.hidden;
const readBytes = (m.layers * layerParams + m.vocab * m.hidden) * 2;
const floorMs = (readBytes / (peak.gbs * 1e9)) * 1e3;
const ridge = (peak.tflops * 1e12) / (peak.gbs * 1e9);

export const review: ChapterReview = {
  lesson: "Decode is limited by the bytes it moves, not by the arithmetic it does.",
  why: "It says the engine already moves bytes almost as fast as this GPU's memory allows, so what is left to win comes from moving fewer bytes or sharing them across a batch, not from faster arithmetic.",
  takeaways: [
    `Decoding one token reads every weight once, so its floor is bytes ÷ bandwidth: ${(readBytes / 1e9).toFixed(2)} GB ÷ ${peak.gbs} GB/s, about ${floorMs.toFixed(1)} ms, whatever the arithmetic costs.`,
    `An op's FLOPs per byte decides what limits it. Below the ridge, ${Math.round(ridge)} FLOP per byte on this GPU, it waits on memory; above it, on compute. Batch-1 decode sits far below.`,
    "The levers are fewer trips to DRAM: fuse kernels so data stays on chip, and batch requests so each weight byte serves many rows.",
  ],
  questions: [
    {
      question: "This GPU's compute doubles but its memory bandwidth stays the same. What happens to the time of a batch-1 decode step?",
      answer: "Almost nothing.",
      why: "The step is waiting for the weights to arrive; the arithmetic was already about 1% of it. Only more bandwidth or fewer bytes would make it faster.",
    },
    {
      question: "At batch 1 the gate/up projection does about 1 FLOP for every byte it reads. Roughly how many rows must share one read of its weights before it stops being memory bound?",
      answer: `About ${Math.round(ridge)}.`,
      why: `Each extra row adds FLOPs but almost no bytes, so FLOPs per byte grows with the rows until it reaches the ridge, ${Math.round(ridge)} FLOP per byte. That is why batch 16 still runs at the bandwidth roof.`,
    },
  ],
};
