export const bookParts = [
  { numeral: "I", title: "the machine", slugs: ["one-token", "foundations"] },
  { numeral: "II", title: "one layer", slugs: ["gemm", "rope", "paged-kv", "paged-attention", "attention-v2", "split-attention", "flash-prefill", "add-rmsnorm", "swiglu"] },
  { numeral: "III", title: "one step", slugs: ["sampling", "cuda-graphs"] },
  { numeral: "IV", title: "many requests", slugs: ["serving", "chunked-prefill"] },
  { numeral: "V", title: "the whole picture", slugs: ["where-the-time-went"] },
];

export const bookOrder = bookParts.flatMap((p) => p.slugs);

export const beyondTitle = "beyond this engine";

export const chapterNumber = (slug: string) => bookOrder.indexOf(slug) + 1;
