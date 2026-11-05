import type { ChapterDef } from "../chapter";

export const bookChapters: Record<string, () => Promise<ChapterDef>> = {
  rope: () => import("./rope").then((m) => m.ropeChapter),
  "cuda-graphs": () => import("./cuda-graphs").then((m) => m.cudaGraphsChapter),
};
