"use client";

import { PagedAttentionStory } from "./attention/story";

export function KernelStoryView({ slug }: { slug: string }) {
  switch (slug) {
    case "paged-attention":
      return <PagedAttentionStory />;
    default:
      return null;
  }
}
