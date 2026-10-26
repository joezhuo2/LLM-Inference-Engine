import { addRmsnorm } from "./add-rmsnorm";
import { pagedAttention } from "./paged-attention";
import { rope } from "./rope";
import { swiglu } from "./swiglu";
import type { KernelStory } from "./types";

export const kernelStories: KernelStory[] = [pagedAttention, addRmsnorm, rope, swiglu];

export function kernelStory(slug: string): KernelStory | undefined {
  return kernelStories.find((k) => k.slug === slug);
}

export type KernelCard = { slug: string; title: string; short: string; ready: boolean };

export const kernelCards: KernelCard[] = [
  ...kernelStories.map((k) => ({ slug: k.slug, title: k.title, short: k.short, ready: true })),
  { slug: "gemm", title: "GEMM and GEMV", short: "cuBLAS with the row-major trick, and how close batch 1 runs to the memory bandwidth roof.", ready: false },
  { slug: "cuda-graphs", title: "CUDA Graphs", short: "The whole decode step captured once and replayed with one launch.", ready: false },
  { slug: "paged-kv", title: "paged KV cache", short: "write_kv and the block manager: memory handed out 16 tokens at a time.", ready: false },
];

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
