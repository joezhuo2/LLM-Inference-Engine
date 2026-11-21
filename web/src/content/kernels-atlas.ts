import { kernelCards } from "./kernels";

const lines: Record<string, string> = {
  "one-token": "Trace a question from its first embedding to the moment the engine picks the next token.",
  foundations: "Meet the threads that run a kernel and the memory hierarchy that keeps them fed.",
  gemm: "A batch changes the shape of the math. Watch a matrix multiply become a walk through the weights.",
  rope: "Give each token a position by rotating pairs of values in its queries and keys.",
  "paged-kv": "Keep past keys and values in a shared pool of blocks that grows with each request.",
  "paged-attention": "Follow a query through the block table to find earlier tokens scattered across memory.",
  "attention-v2": "Carry a running maximum and sum through attention so each key needs only one read.",
  "split-attention": "Divide a long context across workers, then combine their partial answers.",
  "flash-prefill": "Reuse tiles of keys across rows of a prompt while the tensor cores do the work.",
  "add-rmsnorm": "Fold the residual addition into normalization and save a round trip to memory.",
  swiglu: "Two streams meet at a gate. A wider load brings eight values into the kernel together.",
  sampling: "Turn a vocabulary of scores into the single token the engine sends back.",
  "cuda-graphs": "Capture one decode step and replay its kernels with a single launch.",
  "speculative-decoding": "Borrow a continuation from the context and let the model check several guesses in one step.",
  "tensor-parallelism": "Slice a layer’s weights into shards and see where devices must exchange their results.",
  "memory-budget": "Fit fixed model weights and a growing KV cache into the same finite device memory.",
  serving: "Follow the scheduler as it admits waiting requests and reclaims blocks under pressure.",
  "chunked-prefill": "Spread a long prompt across several steps so other requests can keep decoding.",
  "where-the-time-went": "Revisit the complete decode step through the profiler and see which work still dominates.",
  quantization: "Store each weight in fewer bytes so every decoded token has less memory to read.",
  beyond: "Look outward to quantization, shared prefixes, draft models, and larger machines.",
  "dead-ends": "Reopen the experiments that fell short and inspect the tradeoffs their measurements exposed.",
};

export const atlasIntro = "How each kernel of the decode step works and how it got faster, version by version, with the measurements behind every change.";

export const atlasStories = kernelCards.map((c) => ({ slug: c.slug, href: `/kernels/${c.slug}`, title: c.title, dates: c.dates, line: lines[c.slug] ?? c.short, abandoned: c.slug === "dead-ends" }));
