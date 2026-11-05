import type { ChapterDef } from "../chapter";

export const bookChapters: Record<string, () => Promise<ChapterDef>> = {
  rope: () => import("./rope").then((m) => m.ropeChapter),
  "attention-v2": () => import("./attention-v2").then((m) => m.attentionV2Chapter),
};
