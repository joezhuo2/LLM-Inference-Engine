import type { ChapterDef } from "../chapter";

export const bookChapters: Record<string, () => Promise<ChapterDef>> = {
  rope: () => import("./rope").then((m) => m.ropeChapter),
  "paged-attention": () => import("./paged-attention").then((m) => m.pagedAttentionChapter),
};
