import { review as foundations } from "./foundations";
import { review as gemm } from "./gemm";
import { review as rope } from "./rope";
import { review as pagedKv } from "./paged-kv";
import { review as pagedAttention } from "./paged-attention";
import { review as attentionV2 } from "./attention-v2";
import { review as splitAttention } from "./split-attention";
import { review as addRmsnorm } from "./add-rmsnorm";
import { review as swiglu } from "./swiglu";
import { review as sampling } from "./sampling";
import { review as cudaGraphs } from "./cuda-graphs";
import { review as chunkedPrefill } from "./chunked-prefill";
import { review as flashPrefill } from "./flash-prefill";
import { review as serving } from "./serving";
import { review as whereTheTimeWent } from "./where-the-time-went";
import { review as oneToken } from "./one-token";
import { review as memoryBudget } from "./memory-budget";
import { chapterNumber } from "../book-order";
import { kernelStory } from "../kernels";
import type { ChapterReview, ReviewQuestion } from "./types";

const reviews: Record<string, ChapterReview> = { foundations, gemm, rope, "paged-kv": pagedKv, "paged-attention": pagedAttention, "attention-v2": attentionV2, "split-attention": splitAttention, "add-rmsnorm": addRmsnorm, swiglu, sampling, "cuda-graphs": cudaGraphs, "chunked-prefill": chunkedPrefill, "flash-prefill": flashPrefill, serving, "where-the-time-went": whereTheTimeWent, "one-token": oneToken, "memory-budget": memoryBudget };

export const chapterReview = (slug: string): ChapterReview | undefined => (reviews[slug]?.takeaways.length ? reviews[slug] : undefined);

export function recallOf(q: ReviewQuestion): { slug: string; href: string; line: string } | undefined {
  if (!q.recall) return undefined;
  if (q.recall === "start") return { slug: "start", href: "/every-byte/start", line: "recalls start here, the life of a token" };
  const title = kernelStory(q.recall)?.title;
  if (!title || chapterNumber(q.recall) === 0) throw new Error(`a review question recalls "${q.recall}", which is not a chapter of the book`);
  return { slug: q.recall, href: `/kernels/${q.recall}`, line: `recalls chapter ${chapterNumber(q.recall)}, ${title}` };
}
