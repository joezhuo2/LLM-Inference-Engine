import { pagedAttention } from "@/content/kernels/paged-attention";
import { measured } from "@/content/kernels/types";
import type { ChapterReview } from "./types";

const split = "split-context-attention-branch-40";
const sixteenRows = { rows: "16", context: "2,048" };
const v2SixteenRows = measured(split, sixteenRows, "v2");
const splitSixteenRows = measured(split, sixteenRows, "8");

export const review: ChapterReview = {
  lesson: "A block table lets attention read a context scattered anywhere in the pool.",
  why: "It shows that with one sequence the kernel left most of the GPU idle, so cutting the same work into pieces that run side by side, not doing less of it, is what made decoding faster.",
  takeaways: [
    "The KV cache is one pool of 16-token blocks, and a block table says where each of a sequence's blocks sits. The kernel looks every key up there, so the arithmetic is unchanged and only where each key is read from changes.",
    "An online softmax lets the kernel read keys in chunks. It keeps a running max, a running sum and an accumulator, and when a chunk raises the max it scales the sum and accumulator down instead of rereading earlier chunks. That removes v1's need to hold every score at once, and with it the context cap.",
    `When a kernel launches too few blocks to fill the GPU, cutting one long context into ranges and merging the partial results can speed it up: ${pagedAttention.headline.value} here. With enough rows the blocks already exist, so splitting mostly ties or loses.`,
  ],
  questions: [
    {
      question: "Two runs hold the same sequence in the same blocks. In one the block table lists them in order, in the other they sit scattered across the pool. Do the two attention outputs differ?",
      answer: "No, they are the same.",
      why: "The block table only says where each key is read from. The scores, the softmax and the weighted sum are the same arithmetic on the same keys, which is why the tests can check scattered tables against a CPU reference.",
    },
    {
      question: "Splitting the context into 8 ranges speeds up one row at 2,048 tokens. Would 16 rows at the same context gain more, about the same, or less?",
      answer: `Less. In the measurements it gets slower: ${v2SixteenRows.toFixed(1)} µs becomes ${splitSixteenRows.toFixed(1)} µs.`,
      why: "Splitting helps by giving idle parts of the GPU something to do. With 16 rows there are already enough blocks to keep it busy, so the extra splits have nothing left to fill and the merge is added cost.",
    },
  ],
};
