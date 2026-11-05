import type { ChapterDef } from "../chapter";

export const bookChapters: Record<string, () => Promise<ChapterDef>> = {
  rope: () => import("./rope").then((m) => m.ropeChapter),
  "add-rmsnorm": () => import("./add-rmsnorm").then((m) => m.addRmsnormChapter),
};
