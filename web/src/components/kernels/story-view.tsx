"use client";

import { AddRmsnormStory } from "./add-rmsnorm/story";
import { PagedAttentionStory } from "./attention/story";
import { CudaGraphsStory } from "./cuda-graphs/story";
import { GemmStory } from "./gemm/story";
import { PagedKvStory } from "./paged-kv/story";
import { RopeStory } from "./rope/story";
import { SplitAttentionStory } from "./split-attention/story";
import { SwigluStory } from "./swiglu/story";

export function KernelStoryView({ slug }: { slug: string }) {
  switch (slug) {
    case "paged-attention":
      return <PagedAttentionStory />;
    case "add-rmsnorm":
      return <AddRmsnormStory />;
    case "rope":
      return <RopeStory />;
    case "swiglu":
      return <SwigluStory />;
    case "gemm":
      return <GemmStory />;
    case "cuda-graphs":
      return <CudaGraphsStory />;
    case "paged-kv":
      return <PagedKvStory />;
    case "split-attention":
      return <SplitAttentionStory />;
    default:
      return null;
  }
}
