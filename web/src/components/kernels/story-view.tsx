"use client";

import { AddRmsnormStory } from "./add-rmsnorm/story";
import { PagedAttentionStory } from "./attention/story";
import { CudaGraphsStory } from "./cuda-graphs/story";
import { GemmStory } from "./gemm/story";
import { RopeStory } from "./rope/story";
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
    default:
      return null;
  }
}
