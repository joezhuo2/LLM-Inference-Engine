import { chapterNumber } from "../book-order";
import type { ChapterReview } from "./types";

export const review: ChapterReview = {
  lesson: "SwiGLU keeps PyTorch's two roundings, then wins with wide loads on big inputs and scalar threads on small ones.",
  why: "The speedup shows how wider memory accesses can bring an activation kernel to the GPU's bandwidth limit when enough tokens keep it busy.",
  takeaways: [
    "SwiGLU multiplies each up value by a transformed gate. Matching the reference also requires rounding the gate's activation before the product, even though the value now stays in a register.",
    "Vector loads reduce memory instructions for the same bytes. Small batches can still run faster with scalar threads because they spread the work across more of the GPU.",
    `A faster kernel barely changes decode speed when it takes only a small share of the step. Folding SwiGLU into the gate/up projection would be the fusion of chapter ${chapterNumber("add-rmsnorm")} again; at decode it would remove one launch and a row of activations that is tiny next to the weights.`,
  ],
  questions: [
    {
      question: "For a decode launch with very little work, which path can be faster, scalar or vector, and why?",
      answer: "The scalar path, because it spreads the work across more of the GPU.",
      why: "The vector path launches too few blocks and makes each thread run its activation chains in sequence. Scalar threads spread those chains wider.",
    },
    {
      question: "If the kernel keeps the same formula but skips rounding the activation before the product, should the output still match the reference bit for bit?",
      answer: "No, the output can change.",
      why: "The reference rounds the activation before multiplying by up, then rounds the product. Keeping extra precision until the end changes that chain.",
    },
  ],
};
