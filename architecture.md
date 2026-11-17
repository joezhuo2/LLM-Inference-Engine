# Architecture

How the engine is put together and why. The measurements behind these choices are in [results.md](results.md) and [profiling.md](profiling.md); who wrote which part is in [ownership.md](ownership.md).

## The decode loop

```
CLI / bench harness / HTTP server
      |  requests
      v
  Scheduler  <---------------------------------+
      |  batch for this step                   |
      v                                        |
  KV block manager (free list, block tables)   |
      |                                        |
      v                                        |
  Model runner  -- weights from the loader     |
      |  (graph replay when batch fits)        |
      v                                        |
  Kernels + cuBLAS GEMMs                       |
      |  logits                                |
      v                                        |
  Sampler  ----- one token per sequence -------+
```

Each decode step, the scheduler picks the running sequences, the block manager makes sure each has a free cache slot for its next token, the model runner launches one forward pass for the whole batch (replaying a captured CUDA Graph when the batch size matches a captured bucket), the sampler returns one token per sequence, and the scheduler retires finished sequences and admits waiting ones. `engine serve` runs this loop on one engine thread; HTTP threads queue new requests and cancellations for it to apply between steps.

## Design decisions

- **Numerics:** BF16 weights and activations with FP32 accumulation, matching HuggingFace's details exactly: rotate_half RoPE (pairing dimension i with i + 32), BF16 rounding of the residual stream before each RMSNorm, and BF16 rounding before each weight multiply.
- **Weight loading:** the safetensors file is `mmap`ed, its header is parsed and validated, and every tensor is copied into one device allocation through two pinned staging buffers, so copying out of the mapping overlaps the PCIe transfer of the previous chunk. It is not "zero-copy": `mmap` pages are not pinned, so copying straight from them would go through a hidden driver copy anyway.
- **Fused weights at load time:** Q, K and V are concatenated into one `[2560, 2048]` matrix and gate and up into one `[11264, 2048]` matrix, turning five GEMMs per layer into two at no runtime cost.
- **GEMMs through cuBLAS:** the weights are row-major `[out, in]` and cuBLAS is column-major, so `y = x W^T` is computed as the column-major product `Y^T = W X^T` in one `cublasGemmEx` call. A hand-written GEMM is a learning kernel benchmarked against cuBLAS, not a replacement for it.
- **Paged KV cache:** blocks of 16 tokens laid out `[num_blocks, kv_heads, 16, 64]` per layer, block 0 reserved as a null block, and 90% of the VRAM left after the weights given to the cache. Sequences get blocks from a free list through per-sequence block tables, so memory is never reserved for tokens that do not exist yet.
- **Paged decode attention:** version 1 keeps the scores in shared memory (context capped at 2,048); version 2 uses an online softmax with one thread block per query head and sequence, which removes the cap; the 8 query heads that share a KV head each read its K and V in their own block. For 1 to 4 sequences at 512 tokens or more, a split version cuts each context across several blocks and merges their partial softmax results, which fills the GPU at batch 1 (2.8x faster attention at 2,048 tokens).
- **Continuous batching:** waiting, running and finished queues, admission by free blocks, and recompute-style preemption when the cache runs out. By default each step is all prefill or all decode; with `--token-budget N` (chunked prefill) a step holds at most N tokens, decode rows first and prompts split across steps, which under Poisson arrivals cuts the longest wait between two tokens more than twofold at a budget of 256 (`results.md`, "Under load").
- **Speculative decoding:** with `--speculate N` a decoding sequence carries up to N guessed next tokens as extra rows of its own step and keeps the ones the model agrees with, so one step can write several tokens. The guess is copied from the place where the sequence's last tokens appeared earlier in its context (prompt lookup, no draft model). Guesses only take batch rows that no sequence needs, and every kept token is the one the model sampled. With `--speculate-adaptive` a sequence halves its own limit after a step that rejects a guess and doubles it back, up to N, after a step that keeps them all. With `--draft-model DIR` the guesses come from a smaller model with the same vocabulary instead of from the context: it keeps its own paged cache per sequence, drops what the model rejected and decodes the next N tokens greedily, one small step per token. One request at a time, prompts whose answers repeat them run at 1.92x the tokens per second with up to 8 guesses per step, and chat prompts at 1.28x. With 8 or more chat requests decoding at once the fixed limit costs up to 8% and the adaptive one stays level or ahead, and this draft model (TinyStories-110M) makes decoding slower, 0.80x with 2 guesses (`results.md`, "Speculative decoding").
- **Sampling on the GPU:** greedy, Gumbel-max for temperature sampling, and top-k and top-p through binary-searched thresholds instead of a full sort of the vocabulary.
- **FlashAttention-style prefill:** prompt rows go through `flash_attention_prefill`, a causal kernel that gives each warp 16 query rows of one head and streams the keys past them 16 at a time through `mma.sync` on the tensor cores with an online softmax per row; a 2,048-token prompt reaches its first token in 203 ms instead of 1,508 ms (llama.cpp 231 ms, same session, `results.md`). `engine bench --prefill-attention paged` keeps the old per-row path.
- **CUDA Graphs:** with `--graphs` decode is captured for batch sizes 1, 2, 3, 4, 6, 8, 12, 16 and so on up to the max batch (two per power of two, `--graph-steps 2`, the default; 1 keeps only the powers of two and 4 adds 5, 7, 10, 14 and so on, fewer padded rows for more graphs; `results.md`, "Batch size past 16", measured the choice) and replayed from fixed device buffers, which removes per-kernel launch overhead; each batch size that can split its attention also gets one graph per split count, so long contexts replay the split kernel too; prefill runs eagerly.
- **Serving over HTTP:** one thread owns the scheduler and the GPU, so the kernels and scheduler need no locks. A cancel takes effect between steps because `update` matches tokens to sequences by batch row, and a client that reads slowly loses step events rather than holding up the engine.
- **Portable headers:** public headers in `include/engine/` contain no CUDA includes, so the pure C++ parts (config, parser, layout planner, checksums) build and test on machines without a GPU, including macOS.
