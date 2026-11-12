import { addRmsnorm } from "./add-rmsnorm";
import { attentionV2 } from "./attention-v2";
import { chunkedPrefill } from "./chunked-prefill";
import { cudaGraphs } from "./cuda-graphs";
import { flashPrefill } from "./flash-prefill";
import { foundations } from "./foundations";
import { gemm } from "./gemm";
import { memoryBudget } from "./memory-budget";
import { oneToken } from "./one-token";
import { pagedAttention } from "./paged-attention";
import { pagedKv } from "./paged-kv";
import { rope } from "./rope";
import { sampling } from "./sampling";
import { serving } from "./serving";
import { speculativeDecoding } from "./speculative-decoding";
import { splitAttention } from "./split-attention";
import { swiglu } from "./swiglu";
import { whereTheTimeWent } from "./where-the-time-went";
import { deadEnds } from "../dead-ends";
import { beyondTitle } from "../book-order";
import type { KernelStory } from "./types";

export const kernelStories: KernelStory[] = [oneToken, foundations, pagedAttention, addRmsnorm, rope, swiglu, gemm, cudaGraphs, pagedKv, splitAttention, attentionV2, sampling, chunkedPrefill, flashPrefill, memoryBudget, serving, speculativeDecoding, whereTheTimeWent].map((k, index) => ({ ...k, index }));

export function kernelStory(slug: string): KernelStory | undefined {
  return kernelStories.find((k) => k.slug === slug);
}

export type KernelCard = { slug: string; index: number; title: string; short: string; dates: string; ready: boolean };

export const kernelCards: KernelCard[] = kernelStories.map((k) => {
  const first = k.timeline[0]?.date;
  const last = k.timeline[k.timeline.length - 1]?.date;
  return { slug: k.slug, index: k.index, title: k.title, short: k.short, dates: first === last ? (first ?? "") : `${first} – ${last}`, ready: true };
}).concat({ slug: "beyond", index: kernelStories.length, title: beyondTitle, short: "What the engine leaves out, in bytes per token: quantization, a draft model for speculative decoding, prefix caching and multiple GPUs.", dates: "", ready: true }, {
  slug: "dead-ends",
  index: kernelStories.length + 1,
  title: "what didn't work",
  short: "Slower drafts, a broken bit, rules with a cliff and a profile read wrong, each with what was kept.",
  dates: `${deadEnds[0].date} – ${deadEnds[deadEnds.length - 1].date}`,
  ready: true,
});

export type LayerOp = { op: string; label: string; slug: string; note?: string };

export const layerOps: LayerOp[] = [
  { op: "add2_norm1", label: "add + RMSNorm", slug: "add-rmsnorm", note: "from the previous MLP" },
  { op: "qkv", label: "QKV projection", slug: "gemm" },
  { op: "rope", label: "RoPE", slug: "rope" },
  { op: "write_kv", label: "write_kv", slug: "paged-kv" },
  { op: "attention", label: "paged attention", slug: "paged-attention" },
  { op: "o_proj", label: "output projection", slug: "gemm" },
  { op: "add1_norm2", label: "add + RMSNorm", slug: "add-rmsnorm", note: "after attention" },
  { op: "gate_up", label: "gate/up projection", slug: "gemm" },
  { op: "swiglu", label: "SwiGLU", slug: "swiglu" },
  { op: "down", label: "down projection", slug: "gemm" },
];
