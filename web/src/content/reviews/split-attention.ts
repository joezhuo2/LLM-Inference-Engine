import { measured } from "@/content/kernels/types";
import type { ChapterReview } from "./types";

const occupancy = "split-attention-occupancy-branch-66";
const graphs = "split-attention-in-cuda-graphs-branch-43";
const longPrompt = { "max batch": "1", prompt: "1,984" };
const perSm = 6;
const sms = 26;
const slots = perSm * sms;
const at = (splits: number, column: string) => measured(occupancy, { kernel: "split partial", splits: String(splits) }, column);
const v2Blocks = measured(occupancy, { kernel: "v2", splits: "1" }, "grid blocks");
const blocks4 = at(4, "grid blocks");
const blocks8 = at(8, "grid blocks");
const time4 = at(4, "time (us)");
const time8 = at(8, "time (us)");
const before = measured(graphs, longPrompt, "main graphs");
const after = measured(graphs, longPrompt, "this branch graphs");

export const review: ChapterReview = {
  lesson: "A long job on too few blocks leaves most of the GPU idle: cut it up.",
  why: "It shows that a kernel can be slow because its launch is too small for the GPU, not because memory is slow, and that a fix only pays off when the path that really runs, here the replayed CUDA graph, launches it.",
  takeaways: [
    `At batch 1 attention launches one block per query head, ${v2Blocks} blocks, which fills only a fifth of what this GPU can hold. DRAM sits nearly idle, so the kernel is limited by too few warps to hide load latency, not by bandwidth.`,
    "Cutting each context into ranges multiplies the blocks, and the time roughly halves with every doubling while the grid still fits in one wave. Past that the extra blocks wait for a second, mostly empty wave, and with enough rows the grid is already full, so splitting only adds work.",
    "A block that sees only part of the context cannot write the output, so it writes a partial result, and a small merge kernel adds the pieces in split order. That fixed order is why the same input gives the same bits on every run.",
  ],
  questions: [
    {
      question: "One row, 2,048 tokens. Going from 2 to 4 splits cuts the time sharply, and 8 splits doubles the blocks again. Does 8 cut the time by a similar amount?",
      answer: "No, it barely moves.",
      why: `4 splits make ${blocks4} blocks, which fit in one wave of ${slots} slots. 8 splits make ${blocks8}, so the last ${blocks8 - slots} wait for a second, mostly empty wave, and the time only goes from ${time4} to ${time8} µs.`,
    },
    {
      question: "Splitting is wired into the decode step, but each CUDA graph was captured on a dummy batch with no context, where the rule picks 1 split. At a 1,984-token prompt, what do the replayed steps launch?",
      answer: "v2's grid, so nothing gets faster.",
      why: `A graph replays the launch it was captured with, and with no context the rule chose 1 split. Capturing one graph per batch size and split count, and replaying the matching one, took the step from ${before} to ${after} ms.`,
    },
  ],
};
