import type { ChapterReview } from "./types";

export const review: ChapterReview = {
  lesson: "Fusing two kernels saves bytes; saving time takes a kernel that stops waiting on itself.",
  why: "It shows that combining a residual add with its norm speeds up inference only when the fused kernel also cuts the waits inside the work it replaces.",
  takeaways: [
    "The residual add writes the row that the next norm reads, so fusion can save memory traffic and a launch. Fewer bytes alone do not guarantee less time.",
    "Matching the naive pair means preserving where values round to BF16. The norm must use the rounded sum, and its result must round before the learned weight is applied.",
    "Wide loads fetch several values per instruction, registers keep the rounded sums for reuse, and xor shuffles let every thread compute the scale. These changes address waits left inside the first fused kernel.",
  ],
  questions: [
    {
      question: "A fused kernel moves fewer bytes but takes the same time as the separate pair. Would removing more memory traffic necessarily make it faster, and why?",
      answer: "No, waits inside the kernel can still dominate.",
      why: "The first fused kernel saved bytes but still waited on dependent loads and barriers. Its memory traffic alone did not explain its runtime.",
    },
    {
      question: "A fused kernel keeps the sum in float for normalization and rounds only the sum it writes. Should it match the naive pair's norm, and why?",
      answer: "No, the norm must use the rounded sum too.",
      why: "The naive norm reads the BF16 sum written by the add. Squaring an unrounded float sum changes the values being reduced and breaks that contract.",
    },
  ],
};
