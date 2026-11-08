import { chunkBudget } from "@/content/kernels/chunked-prefill";
import type { ChapterReview } from "./types";

export const review: ChapterReview = {
  lesson: "Cap the tokens in each step so one long prompt cannot stall everyone else.",
  why: "It shows how limiting the work in a step lets running requests keep producing tokens while a long prompt fills its cache.",
  takeaways: [
    `A token budget of ${chunkBudget} caps all rows in a step. Decode rows go first and the prompt fills what is left.`,
    "A smaller budget spreads a long prompt over more steps, delaying its first token while shortening the waits for running requests.",
    "When cache space runs out, the partly prefilled sequence is preempted first. Its progress is lost and it starts again when admitted.",
  ],
  questions: [
    {
      question: "The budget stays fixed but more requests are decoding. What happens to the space left for a prompt chunk?",
      answer: "Less space remains for the chunk.",
      why: "Decode rows get their share first, so more decoders leave fewer rows for prefill within the same budget.",
    },
    {
      question: "You lower the budget while keeping the prompt and decoding requests the same. Why can the prompt wait longer for its first token even though the steps are shorter?",
      answer: "The prompt needs more chunks before it can produce a token.",
      why: "Smaller chunks spread prefill over more steps. Only the chunk that finishes the prompt adds it to the sampled list.",
    },
  ],
};
