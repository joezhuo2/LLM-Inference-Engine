import type { ChapterDef } from "../chapter";

export const bookChapters: Record<string, () => Promise<ChapterDef>> = {
  rope: () => import("./rope").then((m) => m.ropeChapter),
  "chunked-prefill": () => import("./chunked-prefill").then((m) => m.chunkedPrefillChapter),
};
