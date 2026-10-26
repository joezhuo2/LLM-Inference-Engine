import { pagedAttention } from "./paged-attention";
import type { KernelStory } from "./types";

export const kernelStories: KernelStory[] = [pagedAttention];

export function kernelStory(slug: string): KernelStory | undefined {
  return kernelStories.find((k) => k.slug === slug);
}
