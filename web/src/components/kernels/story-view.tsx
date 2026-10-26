"use client";

import { AddRmsnormStory } from "./add-rmsnorm/story";
import { PagedAttentionStory } from "./attention/story";
import { RopeStory } from "./rope/story";

export function KernelStoryView({ slug }: { slug: string }) {
  switch (slug) {
    case "paged-attention":
      return <PagedAttentionStory />;
    case "add-rmsnorm":
      return <AddRmsnormStory />;
    case "rope":
      return <RopeStory />;
    default:
      return null;
  }
}
