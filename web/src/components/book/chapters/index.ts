import type { ChapterDef } from "../chapter";

export const bookChapters: Record<string, () => Promise<ChapterDef>> = {
  foundations: () => import("./foundations").then((m) => m.foundationsChapter),
  rope: () => import("./rope").then((m) => m.ropeChapter),
};
