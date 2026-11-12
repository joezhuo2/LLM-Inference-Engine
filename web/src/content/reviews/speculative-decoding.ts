import { spec, specFmt } from "@/content/kernels/speculative-decoding";
import type { ChapterReview } from "./types";

const fmt = specFmt;
const chat = spec.chatAt(spec.best.k) ?? spec.chatWorst;
const miss = spec.missStep;

export const review: ChapterReview = {
  lesson: "A step reads the weights once however many rows it carries, so the rows no request needs can check guesses of the next tokens.",
  why: "It is the one technique in this book that gets more than one token out of a step, and every token it keeps is one the model sampled.",
  takeaways: [
    `${spec.maxBatch} rows cost ${fmt(spec.rowCost, 2)} times one row: ${fmt(spec.step16, 2)} ms against ${fmt(spec.step1, 2)} ms per step (measured). A wrong guess wastes a row, not a step.`,
    `The guess is the tokens that followed the last ${spec.maxNgram} tokens the last time they appeared in the context. No draft model, no extra pass on the GPU.`,
    `On answers that repeat their prompt the model agreed with ${fmt(spec.best.acceptedPct)}% of the guesses and the time per token fell from ${fmt(spec.plain.ms, 2)} to ${fmt(spec.best.ms, 2)} ms; on chat answers it agreed with ${fmt(chat.acceptedPct)}% and the time went from ${fmt(spec.chatPlain.ms, 2)} to ${fmt(chat.ms, 2)} ms (measured).`,
  ],
  questions: [
    {
      question: `A step carries the last real token and ${miss.draft.length} guesses. The model agrees with the first ${miss.kept} and not with the next. How many tokens does the step write?`,
      answer: `${miss.kept + 1}: the token after the real input, and one more for each of the ${miss.kept} guesses it agreed with.`,
      why: "The row of the last right guess still samples a right token. Only the rows after the first wrong guess were computed from a wrong input.",
    },
    {
      question: "Why is the output the same as plain decoding, even with a temperature?",
      answer: "A token is kept only if every token before it is the one the model sampled, and the sampler's random number depends on the seed and the token's place in the answer, not on which step it ran in.",
      why: "A guessed row whose inputs were all right has the same context and the same random number as the plain step would have had, so it samples the same token. Only a near tie in BF16, one rounding step between the two best tokens, can go the other way, as it can when the batch size changes.",
    },
    {
      recall: "paged-kv",
      question: "A rejected row has already written its key and value into the cache. What happens to them?",
      answer: "The sequence's length is cut back to the last kept row. Emptied blocks return to the free list, and a stale slot in a kept block is overwritten by the next step before anything reads it.",
      why: "Attention only reads the slots below a row's context length, and write_kv runs before attention in every layer.",
    },
    {
      recall: "serving",
      question: `${spec.maxBatch} requests are decoding at max batch ${spec.maxBatch}. How many guessed rows does a step carry?`,
      answer: "None. Guesses only take rows that no request needs.",
      why: "A row given to another request yields a token for certain; a row given to a guess yields one only if the guess is right.",
    },
  ],
};
