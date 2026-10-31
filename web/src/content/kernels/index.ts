import { addRmsnorm } from "./add-rmsnorm";
import { attentionV2 } from "./attention-v2";
import { cudaGraphs } from "./cuda-graphs";
import { gemm } from "./gemm";
import { pagedAttention } from "./paged-attention";
import { pagedKv } from "./paged-kv";
import { rope } from "./rope";
import { splitAttention } from "./split-attention";
import { swiglu } from "./swiglu";
import type { KernelStory } from "./types";

export const kernelStories: KernelStory[] = [pagedAttention, addRmsnorm, rope, swiglu, gemm, cudaGraphs, pagedKv, splitAttention, attentionV2];

export function kernelStory(slug: string): KernelStory | undefined {
  return kernelStories.find((k) => k.slug === slug);
}

export type KernelCard = { slug: string; title: string; short: string; dates: string; ready: boolean };

export const kernelCards: KernelCard[] = kernelStories.map((k) => {
  const first = k.timeline[0]?.date;
  const last = k.timeline[k.timeline.length - 1]?.date;
  return { slug: k.slug, title: k.title, short: k.short, dates: first === last ? (first ?? "") : `${first} – ${last}`, ready: true };
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
