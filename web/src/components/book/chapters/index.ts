import type { ChapterDef } from "../chapter";

export const bookChapters: Record<string, () => Promise<ChapterDef>> = {
  gemm: () => import("./gemm").then((m) => m.gemmChapter),
  rope: () => import("./rope").then((m) => m.ropeChapter),
};
