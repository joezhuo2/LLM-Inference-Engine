import type { ChapterDef } from "../chapter";

export const bookChapters: Record<string, () => Promise<ChapterDef>> = {
  "one-token": () => import("./one-token").then((m) => m.oneTokenChapter),
  foundations: () => import("./foundations").then((m) => m.foundationsChapter),
  gemm: () => import("./gemm").then((m) => m.gemmChapter),
  "paged-kv": () => import("./paged-kv").then((m) => m.pagedKvChapter),
  rope: () => import("./rope").then((m) => m.ropeChapter),
  swiglu: () => import("./swiglu").then((m) => m.swigluChapter),
  "attention-v2": () => import("./attention-v2").then((m) => m.attentionV2Chapter),
  "chunked-prefill": () => import("./chunked-prefill").then((m) => m.chunkedPrefillChapter),
  "flash-prefill": () => import("./flash-prefill").then((m) => m.flashPrefillChapter),
  "paged-attention": () => import("./paged-attention").then((m) => m.pagedAttentionChapter),
  "cuda-graphs": () => import("./cuda-graphs").then((m) => m.cudaGraphsChapter),
  "split-attention": () => import("./split-attention").then((m) => m.splitAttentionChapter),
  serving: () => import("./serving").then((m) => m.servingChapter),
  sampling: () => import("./sampling").then((m) => m.samplingChapter),
  "add-rmsnorm": () => import("./add-rmsnorm").then((m) => m.addRmsnormChapter),
  "where-the-time-went": () => import("./where-the-time-went").then((m) => m.whereTheTimeWentChapter),
};
