import type { ChapterDef } from "../chapter";

export const bookChapters: Record<string, () => Promise<ChapterDef>> = {
  "paged-kv": () => import("./paged-kv").then((m) => m.pagedKvChapter),
  rope: () => import("./rope").then((m) => m.ropeChapter),
};
