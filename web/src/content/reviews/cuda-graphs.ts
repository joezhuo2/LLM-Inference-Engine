import { kvReplay, launchGaps } from "@/lib/data";
import trace from "@/content/kernels/cuda-graphs-trace.json";
import { measured } from "@/content/kernels/types";
import { chapterNumber } from "../book-order";
import type { ChapterReview } from "./types";

const eager = launchGaps("eager");
const replay = launchGaps("graphs");
const blockSize = kvReplay().meta.block_size;
const gap = trace.eager.gap_us.p50;
const lengths = trace.eager.kernels_in_layer.map((k) => k.end - k.start);
const shortest = Math.min(...lengths);
const longest = Math.max(...lengths);
const idleShare = (us: number) => (100 * gap) / (us + gap);
const pct = (v: number) => (v < 10 ? v.toFixed(1) : Math.round(v).toString());
const tpot = (system: string, batch: string) => measured("throughput-and-latency", { system, "max batch": batch }, "TPOT p50 ms", 0, "results");
const savedShare = (batch: string) => (100 * (tpot("engine", batch) - tpot("engine + graphs", batch))) / tpot("engine", batch);

export const review: ChapterReview = {
  lesson: "When kernels are short, the gaps between launches cost time: record the step once and replay it.",
  why: "It shows that a batch-1 step can leave the GPU waiting even when every kernel is fast, so the speed of a step depends on how many launches it needs, not only on how long each kernel runs.",
  takeaways: [
    `Launched one by one, every kernel is preceded by a gap of a few microseconds while the GPU takes the next launch and sets up its grid. Those gaps leave the GPU idle for ${eager.idle_pct}% of a batch-1 decode step, and faster kernels would not remove them.`,
    `A CUDA Graph records the chain of launches once and replays it with a single launch, which cuts the idle share to ${replay.idle_pct}% while the kernels take the same time. The saving is about a fixed time per step, because every step has the same kernel boundaries.`,
    "A graph freezes each kernel's pointers and grid, so a replay can only change values. Inputs go through fixed buffers, batches are padded to a bucket size, and anything that changes a grid, like the split count of attention, needs its own graph.",
  ],
  questions: [
    {
      question: `Launched one by one, each kernel is preceded by a gap of about ${gap} µs. Which loses a bigger share of its time to that gap: the longest kernel in a layer (${longest.toFixed(0)} µs) or the shortest (${shortest.toFixed(1)} µs)?`,
      answer: "The shortest.",
      why: `The gap is about the same before every kernel, so its share is the gap divided by the kernel plus the gap: about ${pct(idleShare(shortest))}% for the shortest kernel and ${pct(idleShare(longest))}% for the longest. Short kernels are where launches hurt, and a graph removes the gap without touching the kernels.`,
    },
    {
      question: "A graph replay saves about the same time per step at batch 1 and at batch 16, because both launch the same kernels. Does it help a batch-16 token more or less, as a share of its time?",
      answer: "Less.",
      why: `The saved time is nearly fixed per step, but a batch-16 step takes longer, so the same saving is a smaller share: about ${pct(savedShare("1"))}% of a batch-1 token and ${pct(savedShare("16"))}% of a batch-16 token.`,
    },
    {
      question: `Chapter ${chapterNumber("paged-kv")} finds a request's cached tokens through a block table. A CUDA Graph freezes every kernel's pointers. Why can one captured graph still serve a request whose context grows every step?`,
      answer: "Each step changes values in the staged table and slots, never an address.",
      why: `The kernels read the cache pool and the staged block table at fixed addresses. Token pos goes to slot table[pos / ${blockSize}] · ${blockSize} + pos mod ${blockSize}, and stage writes the new table, slots and context length into the same buffer before each replay, so the same pointers reach different blocks. A request gets a new block only every ${blockSize} tokens.`,
      recall: "paged-kv",
    },
  ],
};
