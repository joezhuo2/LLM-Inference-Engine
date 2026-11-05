import type { ChapterDef } from "../chapter";

export const bookChapters: Record<string, () => Promise<ChapterDef>> = {
  rope: () => import("./rope").then((m) => m.ropeChapter),
  sampling: () => import("./sampling").then((m) => m.samplingChapter),
};
