import { addRmsnorm } from "./add-rmsnorm";
import { cudaGraphs } from "./cuda-graphs";
import { gemm } from "./gemm";
import { pagedAttention } from "./paged-attention";
import { pagedKv } from "./paged-kv";
import { rope } from "./rope";
import { swiglu } from "./swiglu";
import type { KernelStory } from "./types";

export const kernelStories: KernelStory[] = [pagedAttention, addRmsnorm, rope, swiglu, gemm, cudaGraphs, pagedKv];

export function kernelStory(slug: string): KernelStory | undefined {
  return kernelStories.find((k) => k.slug === slug);
}

export type KernelCard = { slug: string; title: string; short: string; ready: boolean };

export const kernelCards: KernelCard[] = kernelStories.map((k) => ({ slug: k.slug, title: k.title, short: k.short, ready: true }));

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
