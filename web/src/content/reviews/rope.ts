import type { ChapterReview } from "./types";

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
  ],
};
