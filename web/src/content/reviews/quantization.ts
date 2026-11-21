import { budget } from "@/content/kernels/memory-budget";
import { bf16Bits, bf16Floor, e4m3, fp8Floor, kvPool, kvToken, normalReach, q4Floor, qFmt, qGb, qMs, ridge } from "@/content/kernels/quantization";
import type { ChapterReview } from "./types";

const fmt = qFmt;

export const review: ChapterReview = {
  lesson: "The batch-1 floor is the weights' bytes over the bandwidth, so storing each weight in fewer bits moves the floor itself. This is the plan and the arithmetic: the engine's FP8 kernels are stubs, and nothing here is measured yet.",
  why: "It is the one lever left after the kernels: no kernel can beat the floor, but fewer bytes per weight lower it.",
  takeaways: [
    `A token reads ${qGb(bf16Floor.gb * 1e9)} of BF16 weights, a floor of ${qMs(bf16Floor.ms)}; in FP8 ${qGb(fp8Floor.gb * 1e9)} and ${qMs(fp8Floor.ms)}; at 4 bits ${qMs(q4Floor.ms)} (calculated). The parts that are not weights do not shrink.`,
    `E4M3 is ${e4m3.sign} sign, ${e4m3.exponent} exponent and ${e4m3.mantissa} mantissa bits, largest value ${e4m3.max}. One scale per row, the row's largest |w| ÷ ${e4m3.max}, fits each row into it, and rounding moves a weight by at most 1/${2 * e4m3.perPower} of its power of two, against 1/${2 * bf16Bits.perPower} in BF16.`,
    `An FP8 cache halves a token's ${fmt(kvToken.bf16)} bytes, so the same pool holds ${fmt(kvPool.fp8)} tokens instead of ${fmt(kvPool.bf16)} (calculated).`,
  ],
  questions: [
    {
      question: `A token's weights take ${qMs(bf16Floor.ms)} to read in BF16. Will FP8 weights make the whole token twice as fast?`,
      answer: "No. The floor halves, but attention reading the cache, the norms and the gaps between kernels do not shrink with the weights. How much is gained is not measured yet.",
      why: "Only the weights' bytes get narrower. The rest of the step stays as it is.",
    },
    {
      recall: "foundations",
      question: `At ${budget.maxBatch} rows, 4-bit weights do ${fmt((2 * budget.maxBatch) / 0.5)} FLOP per byte. Does the step become compute bound?`,
      answer: `No. The ridge is ${fmt(ridge)} FLOP per byte, so the read still sets the time.`,
      why: "Intensity is the work divided by the bytes. Fewer bytes per weight raise it, but decode starts so far left of the ridge that it stays on the memory side.",
    },
    {
      question: "Why does each row of a weight matrix get its own scale instead of storing the weights as they are?",
      answer: `So the row's largest weight lands on ${e4m3.max}, nothing saturates, and every weight within ${fmt(normalReach)} times of the largest keeps all ${e4m3.mantissa} mantissa bits. Without a scale, weights smaller than 2^${e4m3.minNormalExp} fall among the coarse values near 0.`,
      why: "E4M3 has full precision only from 2^-6 up. Dividing by a scale slides the whole row up the ladder; one FP32 number per row is all it costs.",
    },
  ],
};
