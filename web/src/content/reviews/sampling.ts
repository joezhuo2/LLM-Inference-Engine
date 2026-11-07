import { sampling } from "@/content/kernels/sampling";
import type { ChapterReview } from "./types";

export const review: ChapterReview = {
  why: "It shows that choosing the next token does not require sorting the vocabulary, and that avoiding the sort can save time even before a sort-based sampler does its draw.",
  takeaways: [
    "Adding independent Gumbel noise to the scaled logits and taking the argmax draws from the softmax distribution. The sampler does not need to build that distribution explicitly.",
    "Top-k searches for a cut by counting tokens, while top-p searches by adding probability mass. When both are set, top-k goes first, and ties at the cut stay in.",
    `A threshold search can cost less than ordering the whole vocabulary. In this benchmark, top-k sampling is ${sampling.headline.value} faster than sorting the row alone.`,
  ],
  questions: [
    {
      question: "Which can take longer here: temperature sampling alone, or temperature sampling with top-k, and why?",
      answer: "Temperature sampling alone.",
      why: "Every token needs Gumbel noise with costly double-precision logs. Top-k adds a threshold search but only the kept tokens need noise, so filtering can save more work than it adds.",
    },
    {
      question: "A request moves from running alone to running in a batch, with its logits, seed, step and sampling settings unchanged. Should its next token change?",
      answer: "No.",
      why: "Philox gets each token's noise from the request's seed and its offset at that decode step. That noise does not depend on which other requests share the launch, and the tests check that the draw stays the same.",
    },
  ],
};
