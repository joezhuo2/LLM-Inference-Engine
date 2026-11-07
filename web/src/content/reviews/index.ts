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
import type { ChapterReview } from "./types";

const reviews: Record<string, ChapterReview> = { foundations, gemm, rope, "paged-kv": pagedKv, "paged-attention": pagedAttention, "attention-v2": attentionV2, "split-attention": splitAttention, "add-rmsnorm": addRmsnorm, swiglu, sampling, "cuda-graphs": cudaGraphs, "chunked-prefill": chunkedPrefill };

export const chapterReview = (slug: string): ChapterReview | undefined => (reviews[slug]?.takeaways.length ? reviews[slug] : undefined);
