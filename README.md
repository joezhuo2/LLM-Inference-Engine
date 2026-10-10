# llm-inference-engine

A single-GPU LLM inference engine in C++20 and CUDA. It serves [TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) in BF16 with FP32 accumulation, matches HuggingFace output token for token, and serves many concurrent requests with a paged KV cache, a paged decode attention kernel, continuous batching, custom fused kernels, sampling on the GPU, and CUDA Graphs.

The goal is not to beat vLLM. It is a small engine in which every design decision is measured, explained and reproducible: every performance number comes from a script in this repo that anyone can rerun.

## Goals

1. **Correct:** with teacher forcing on 20 fixed prompts, the engine's argmax token matches HuggingFace (BF16, same weights) at 99%+ of positions with a mean absolute logit error under 0.05, and free-running greedy output for the first 64 tokens either matches HF exactly or first differs where HF's own logits for the two tokens are at most 2 BF16 ulps apart, on all 20 prompts (an exact match on most prompts is out of reach in BF16: HF's own full forward pass agrees with its `generate()` on only 14 of the 20).
2. **Fast at batch 1:** decode reaches at least 60% of the GPU's measured DRAM bandwidth.
3. **Fast at batch N:** with 16 concurrent requests, total throughput is at least 5x the batch-1 throughput.
4. **Explained:** every kernel has before and after Nsight Compute numbers, and a results page with plots explains them.
5. **Reproducible:** a build plus one script reproduces every number in this README from a clean clone.

Correct comes first: nothing is optimized until the naive engine produces the same text as HuggingFace, and the naive kernels stay in the repo as the oracles the optimized kernels are tested against.

## Architecture

```
CLI / bench harness
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

Each decode step, the scheduler picks the running sequences, the block manager makes sure each has a free cache slot for its next token, the model runner launches one forward pass for the whole batch (replaying a captured CUDA Graph when the batch size matches a captured bucket), the sampler returns one token per sequence, and the scheduler retires finished sequences and admits waiting ones.

## The model and why decode is memory bound

TinyLlama is a plain Llama: 22 layers, hidden size 2048, 32 query heads and 4 KV heads of width 64 (grouped-query attention with a group size of 8), an MLP of width 5632, a 32,000-token vocabulary and no RoPE scaling. That is about 1.1B parameters, 2.2 GB in BF16.

Decoding one token at batch 1 reads every weight except all but one row of the embedding table, about 2.07 GB, while doing only about two FLOPs per weight. At the measured 323.6 GB/s of this laptop GPU that is a floor of about 6.4 ms per token (about 156 tokens per second), however fast the arithmetic is. Batch-1 decode is therefore a bandwidth problem, and batching is what turns it into a compute problem: the same weight reads serve every sequence in the batch.

## Design decisions

- **Numerics:** BF16 weights and activations with FP32 accumulation, matching HuggingFace's details exactly: rotate_half RoPE (pairing dimension i with i + 32), BF16 rounding of the residual stream before each RMSNorm, and BF16 rounding before each weight multiply.
- **Weight loading:** the safetensors file is `mmap`ed, its header is parsed and validated, and every tensor is copied into one device allocation through two pinned staging buffers, so copying out of the mapping overlaps the PCIe transfer of the previous chunk. It is not "zero-copy": `mmap` pages are not pinned, so copying straight from them would go through a hidden driver copy anyway.
- **Fused weights at load time:** Q, K and V are concatenated into one `[2560, 2048]` matrix and gate and up into one `[11264, 2048]` matrix, turning five GEMMs per layer into two at no runtime cost.
- **GEMMs through cuBLAS:** the weights are row-major `[out, in]` and cuBLAS is column-major, so `y = x W^T` is computed as the column-major product `Y^T = W X^T` in one `cublasGemmEx` call. A hand-written GEMM is a learning kernel benchmarked against cuBLAS, not a replacement for it.
- **Paged KV cache:** blocks of 16 tokens laid out `[num_blocks, kv_heads, 16, 64]` per layer, block 0 reserved as a null block, and 90% of the VRAM left after the weights given to the cache. Sequences get blocks from a free list through per-sequence block tables, so memory is never reserved for tokens that do not exist yet.
- **Paged decode attention:** version 1 keeps the scores in shared memory (context capped at 2,048); version 2 uses an online softmax with one thread block per KV head, so each block reads its K and V once for all 8 query heads that share them.
- **Continuous batching:** waiting, running and finished queues, admission by free blocks, and recompute-style preemption when the cache runs out. Each step is all prefill or all decode; the resulting stall is measured so chunked prefill can fix it later.
- **Sampling on the GPU:** greedy, Gumbel-max for temperature sampling, and top-k and top-p through binary-searched thresholds instead of a full sort of the vocabulary.
- **CUDA Graphs:** decode is captured for batch sizes 1, 2, 4, 8, 16 and 32 and replayed from fixed device buffers, which removes per-kernel launch overhead; prefill runs eagerly.
- **Portable headers:** public headers in `include/engine/` contain no CUDA includes, so the pure C++ parts (config, parser, layout planner, checksums) build and test on machines without a GPU, including macOS.

## Beyond the MVP

Llama-3.2-1B and other models, FP8 and INT8 Tensor Core GEMMs, FlashAttention-style prefill and split-K decode, prefix caching and copy-on-write blocks, chunked prefill, speculative decoding, and an OpenAI-compatible HTTP server with a live metrics dashboard.

## Hardware

Developed on an RTX 5060 Laptop GPU (Blackwell, compute capability 12.0, 8 GB GDDR7, 26 SMs) under WSL2 with Ubuntu 24.04, CUDA 13.3 and GCC 13. Its measured memory bandwidth, 323.6 GB/s (about 84% of the theoretical peak), is the denominator for every "% of peak" figure. Final comparison benchmarks also run on one cloud GPU (an L4 or RTX 4090).

## Building

Requires CUDA 12.8 or newer (for `sm_120`), CMake 3.28 or newer, Ninja and a C++20 compiler; GoogleTest, nlohmann/json and CLI11 are found on the system or fetched. Download the model and link the directory that holds it into the repo as `models` (gitignored), or set `ENGINE_MODEL_DIR` to the model directory instead:

```bash
hf download TinyLlama/TinyLlama-1.1B-Chat-v1.0 --local-dir ~/models/tinyllama
```

```bash
ln -s ~/models models
```

```bash
cmake --preset release && cmake --build --preset release
```

```bash
ctest --preset release -LE todo_joe
```

The `cpu` preset builds and tests only the pure C++ parts, for machines without CUDA.

## Running

```bash
./build/release/bin/engine generate --model models/tinyllama --prompt "Write a haiku about GPUs."
```

The answer streams to stdout as it decodes, followed on stderr by the prompt and generated token counts, the time to the first token and the decode rate. `--system` adds a system message and `--max-new-tokens` (default 256) caps the answer. Decoding is greedy by default; `--temperature 0.7` samples from `softmax(logits / 0.7)` on the GPU with the Gumbel-max trick, and `--seed N` makes the answer repeatable (without it the seed is random and printed on stderr). Generation runs at batch 1 on the naive kernels and a contiguous KV cache (milestone M2): about 110 tokens/s on the RTX 5060, greedy or sampled.

## Repository layout

```
include/engine/   public headers (core, kernels, loader, model, runtime, kv, sched, sampling, tokenizer)
src/              the engine_core (pure C++), engine_kernels (CUDA, cuBLAS) and engine_runtime libraries, and the CLI
tests/            unit tests, GPU tests, test helpers and generated golden data
scripts/          Python reference and golden scripts
docs/             hardware notes, ownership, and learning exercises
```

## Ownership

This is Joe Zhuo's project, built with Claude as tutor, reviewer and writer of the supporting code. The core (kernels, paged attention, the KV block manager, the scheduler, CUDA Graph capture and the safetensors parser) is written by hand; `docs/ownership.md` records who wrote each component.
