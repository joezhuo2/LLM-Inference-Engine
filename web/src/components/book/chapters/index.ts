import type { ChapterDef } from "../chapter";

export const bookChapters: Record<string, () => Promise<ChapterDef>> = {
  "paged-kv": () => import("./paged-kv").then((m) => m.pagedKvChapter),
  rope: () => import("./rope").then((m) => m.ropeChapter),
  "attention-v2": () => import("./attention-v2").then((m) => m.attentionV2Chapter),
  "chunked-prefill": () => import("./chunked-prefill").then((m) => m.chunkedPrefillChapter),
  "paged-attention": () => import("./paged-attention").then((m) => m.pagedAttentionChapter),
  "cuda-graphs": () => import("./cuda-graphs").then((m) => m.cudaGraphsChapter),
  "split-attention": () => import("./split-attention").then((m) => m.splitAttentionChapter),
  sampling: () => import("./sampling").then((m) => m.samplingChapter),
  "add-rmsnorm": () => import("./add-rmsnorm").then((m) => m.addRmsnormChapter),
};
