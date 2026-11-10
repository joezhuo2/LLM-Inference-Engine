import { chapterNumber } from "../book-order";
import type { ChapterReview } from "./types";

export const review: ChapterReview = {
  lesson: "Fusing two kernels keeps the value between them on the chip and drops a launch; it saves time only as far as the fused kernel stops waiting on itself.",
  why: "It shows what fusing neighbouring kernels buys at decode: the bytes of a DRAM round trip and one launch, not arithmetic, and only a kernel that stops waiting on itself turns that into time.",
  takeaways: [
    `Every separate elementwise kernel reads its inputs from DRAM, writes its output back and pays a launch. Fusing two neighbours keeps the value between them in registers, so that round trip and one launch disappear. SwiGLU (chapter ${chapterNumber("swiglu")}) uses the pattern again, and CUDA Graphs (chapter ${chapterNumber("cuda-graphs")}) remove the launch cost for every kernel.`,
    "Matching the naive pair means preserving where values round to BF16. The norm must use the rounded sum, and its result must round before the learned weight is applied.",
    "Fewer bytes alone do not guarantee less time. The first fused kernel waited on dependent loads and barriers and ran as long as the pair. Wide loads fetch several values per instruction, registers keep the rounded sums for reuse, and xor shuffles let every thread compute the scale; these removed those waits.",
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
