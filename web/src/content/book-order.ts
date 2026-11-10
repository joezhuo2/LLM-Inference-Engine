export const bookParts = [
  { numeral: "I", title: "the machine", slugs: ["foundations"] },
  { numeral: "II", title: "one layer", slugs: ["gemm", "rope", "paged-kv", "paged-attention", "attention-v2", "split-attention", "add-rmsnorm", "swiglu"] },
  { numeral: "III", title: "one step", slugs: ["sampling", "cuda-graphs"] },
  { numeral: "IV", title: "many requests", slugs: ["chunked-prefill"] },
];

export const bookOrder = bookParts.flatMap((p) => p.slugs);

export const chapterNumber = (slug: string) => bookOrder.indexOf(slug) + 1;
