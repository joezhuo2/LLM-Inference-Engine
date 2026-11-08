import { measured } from "@/content/kernels/types";
import { profilingTable } from "@/lib/data";
import type { ChapterReview } from "./types";

const v2Section = "paged-attention-v2-branch-37";
const wired = "paged-attention-v2-in-the-forward-pass-branch-38";
const shared = "attention-shared-memory-branch-67";

const v2Bytes = measured(shared, { context: "2,048" }, "v2 bytes");
const dram = Math.max(...["v1", "v2"].flatMap((k) => ["1 x 2,048", "16 x 512"].map((s) => measured(v2Section, { "rows x context": s, kernel: k }, "DRAM throughput", 1))));
const tokensRow = "bench tokens/s at max batch 1 / 4 / 16 with graphs";
const tokensPerSecond = (column: string) => {
  const { header, rows } = profilingTable(wired, 0);
  const cell = rows.find((r) => r[0] === tokensRow)?.[header.indexOf(column)] ?? "";
  const values = cell.split("/").map((v) => Number(v.trim()));
  if (values.length !== 3 || values.some((v) => !Number.isFinite(v))) throw new Error(`${wired}: no tokens/s triple in "${column}"`);
  return values;
};
const before = tokensPerSecond("before (e07c022)");
const after = tokensPerSecond("after (79c1f91)");
const gain = (i: number) => ((after[i] / before[i] - 1) * 100).toFixed(1);

export const review: ChapterReview = {
  lesson: "Change how a kernel keeps its state, and the same math can run several times faster.",
  why: "It shows that restructuring how a kernel reads memory and keeps its state can cut its time several times over with the same softmax and the same result, and that the saving survives inside a real decode step.",
  takeaways: [
    `An online softmax keeps a running maximum and sum and rescales them whenever a chunk brings a larger score. That makes it one pass, and shared memory holds one chunk, ${v2Bytes.toLocaleString("en-US")} bytes at any context length, instead of growing with the context.`,
    `Neither kernel came near the memory limit, with DRAM throughput at most ${dram.toFixed(1)}%. The win came from keeping more warps busy and letting a group of lanes share a key so its loads sit side by side.`,
    `A much faster kernel does not make the whole step much faster. Attention was a small slice of a batch-1 step, which gained only ${gain(0)}% in tokens per second.`,
  ],
  questions: [
    {
      question: "A new chunk's largest score is smaller than the running maximum so far. By what factor does the online softmax scale the old sum and values?",
      answer: "By 1: they are not scaled at all.",
      why: "The old sum is scaled by exp(m_old − m_new), and when the maximum does not rise, m_new equals m_old, so the factor is exp(0), which is 1. The chunk's own exponentials are then taken against the existing maximum and added.",
    },
    {
      question: "With v2 in the decode step and contexts of at most 64 tokens, does batch 1 or batch 16 gain more tokens per second, and why?",
      answer: `Batch 16: ${gain(2)}% against ${gain(0)}% at batch 1, with CUDA Graphs.`,
      why: "Attention work grows with the number of rows while the weight reads of the GEMVs do not. So attention is a bigger share of a batch-16 step, and a faster attention kernel saves more of it.",
    },
  ],
};
