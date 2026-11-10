import { measured } from "@/content/kernels/types";
import { ceilings } from "@/lib/data";
import type { ChapterReview } from "./types";

const peak = ceilings();
const oneToken = { tokens: "1" };
const fastUs = measured("fast-rope-branch-31", oneToken, "fast (us)");
const fastKb = measured("fast-rope-branch-31", oneToken, "DRAM, fast (KB)");
const readUs = (fastKb * 1e3) / (peak.gbs * 1e9) * 1e6;

export const review: ChapterReview = {
  lesson: "A faster kernel first has to match the old one bit for bit.",
  why: "It shows that exposing independent rotations and sharing trigonometry across heads can speed up a kernel while staying bit-identical to the naive kernel, which matches Hugging Face.",
  takeaways: [
    "RoPE turns query and key pairs by position. For fixed query and key vectors, their attention score depends on the gap between positions and stays the same when both move together.",
    "Independent pairs can run in parallel, and heads can share the same sine and cosine. This removes the draft's long chain of pair rotations and avoids repeating trigonometry for every head.",
    "Matching the rotation formula is not enough to match the model's bits. The source of the frequencies and the rounding of each product matter, especially at large positions.",
  ],
  questions: [
    {
      question: "If fixed query and key vectors both move forward by the same distance, what happens to their attention score after RoPE?",
      answer: "It stays the same.",
      why: "Both rotations advance together, so the angle between them stays the same. The score depends on their position gap, which has not changed.",
    },
    {
      question: "A device frequency calculation differs from the host's only in the last bit. Why might the rotated outputs disagree more at large positions?",
      answer: "Position multiplies the frequency error.",
      why: "The angle is position times frequency, so the error grows with position and can cross a rounding boundary. The chapter shows how the later subtraction can magnify that difference.",
    },
    {
      question: `The GEMM chapter timed a decode GEMV by its bytes over the bandwidth. The fast RoPE kernel moves ${fastKb.toFixed(1)} KB for one token and takes ${fastUs.toFixed(1)} µs (measured). Is it waiting on memory?`,
      answer: `No: at ${peak.gbs} GB/s those bytes take about ${readUs.toFixed(2)} µs (calculated).`,
      why: "The GEMVs run near the bandwidth roof because they move megabytes. RoPE moves so little that the launch and one chain of load, sine and store set its time, far from that roof.",
      recall: "gemm",
    },
  ],
};
