import { flashFmt, flashKeyRows, flashLengths, flashLong, flashNumbers, flashPath, flashPaths, flashShape, flashSystems, flashTtft } from "@/content/kernels/flash-prefill";
import type { ChapterReview } from "./types";

const n = flashNumbers;
const s = flashShape;
const half = flashLengths[flashLengths.length - 2];
const growth = (system: string) => flashTtft(system, flashLong) / flashTtft(system, half);
const pagedPath = flashPath(flashPaths.paged, flashLong);

export const review: ChapterReview = {
  lesson: "A shorter path to each byte saves nothing if every row still fetches it alone; sharing the fetch does.",
  why: "It shows a rewrite that passed every test and was slower, then one that changed which threads share the work, and it says plainly which parts of the result were measured and which were not.",
  takeaways: [
    `With one walk per row, a ${flashLong}-token prompt reads ${flashFmt(flashKeyRows.perRow)} key rows per head; with one pass per ${s.tile} rows it reads ${flashFmt(flashKeyRows.tiled)} (calculated). The score and value products are the same number, and they move to the tensor cores.`,
    `Tests prove a kernel right, not fast. The first kernel passed all ${s.tests} tests and took ${flashFmt(n.first)} ms where the path it replaced took ${flashFmt(pagedPath)} ms (measured), so it stayed behind a switch.`,
    `The tiled kernel takes the same prompt from ${flashFmt(n.paged)} to ${flashFmt(n.tiled)} ms (measured) and keeps the cost per prompt token between ${n.perTokenLow} and ${n.perTokenHigh} ms. It was not profiled with Nsight Compute, so why it is fast is argued from its design, not measured.`,
  ],
  questions: [
    {
      question: "The first kernel read the chunk's keys and values straight from the QKV buffer instead of through the block table. Why was prefill not faster?",
      answer: `Because the walk did not change: ${flashFmt(n.first)} against ${flashFmt(pagedPath)} ms at ${flashLong} tokens, ${n.firstLoss.toFixed(0)}% slower.`,
      why: "It still ran one thread block per row and query head, took the keys one at a time and waited at two barriers per key. Every row fetched every key of its context by itself, so the number of fetches was the same and only their address changed.",
    },
    {
      question: `Double the prompt from ${half} to ${flashLong} tokens. By what factor does the time to first token grow with the tiled kernel, and with one walk per row?`,
      answer: `About ${growth(flashSystems.flash).toFixed(1)} times with the tiled kernel and ${growth(flashSystems.paged).toFixed(1)} times with one walk per row (calculated from the measured medians).`,
      why: "A cost that is flat per token doubles when the prompt doubles. One walk per row adds the square: twice the rows, each walking twice as far. The tiled kernel still computes the square of products, yet its measured time grows almost in step with the prompt.",
    },
    {
      recall: "attention-v2",
      question: `The attention v2 chapter's online softmax keeps a running maximum and sum and rescales them when a larger score arrives. Here the scores of one query row against a key tile sit in ${s.lanesPerRow} different lanes. What has to happen before the row can update its maximum and sum?`,
      answer: `The ${s.lanesPerRow} lanes have to combine their parts: two shuffles give every one of them the tile's maximum, and two more its sum.`,
      why: "The update needs the largest score of the whole tile for that row, and each lane only holds four of the sixteen. Two rounds of exchanging values with a neighbour, one lane away and then two lanes away, leave all four lanes with the same result. The rule after that is the one from attention v2.",
    },
  ],
};
