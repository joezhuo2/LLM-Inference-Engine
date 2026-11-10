import { chapterNumber } from "../book-order";
import { modelShape } from "@/lib/data";
import type { ChapterReview } from "./types";

const shape = modelShape();
const rowBytes = 2 * shape.intermediate * 2;
const weightBytes = shape.hidden * rowBytes;
const roundTripKb = ((2 * rowBytes) / 1024).toFixed(1);
const roundTripShare = (((2 * rowBytes) / weightBytes) * 100).toFixed(1);
const weightsMb = (weightBytes / 1024 ** 2).toFixed(1);

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
    {
      question: `Chapter ${chapterNumber("add-rmsnorm")} fused two kernels so the value between them stayed in registers. Fusing SwiGLU into the gate/up GEMM would keep its row out of memory the same way. Why would that save little at decode?`,
      answer: "The row it keeps out of memory is tiny next to the weights the GEMM reads.",
      why: `Fusion saves the round trip of the value through DRAM and one launch. At one token that round trip is ${roundTripKb} KB per layer, ${roundTripShare}% of the ${weightsMb} MB of gate and up weights (calculated from config.json), so it saves a launch more than bytes.`,
      recall: "add-rmsnorm",
    },
  ],
};
