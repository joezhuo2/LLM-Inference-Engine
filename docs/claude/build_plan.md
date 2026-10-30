# LLM Inference Engine Build Plan

Section 2 of Joe's handoff doc, reproduced verbatim except that Google Docs export escapes (such as `\~` and `\[`) were removed. Dates are the originals; Joe moved the M1 target to Oct 7 because he is now solo (see `status.md`).

Sep 23, 2026 · @Joe Zhuo

## Overview

We are building a single-GPU LLM inference engine in C++20 and CUDA that runs TinyLlama-1.1B end to end, matches HuggingFace output token for token, and serves multiple concurrent requests with a paged KV cache, custom fused kernels, and CUDA Graphs. The MVP (Part 1) is due **Sunday, November 15, 2026** (about 7.5 weeks from today, 2 people). Part 2 is everything that makes it look and feel like a product, plus the harder kernels, and has no deadline.

The goal is not to beat vLLM. The goal is a system where every design decision is measured, explained, and defensible in an NVIDIA or inference-startup interview. Every number on the resume must come from a benchmark script in the repo that anyone can rerun.

### What is in each part

| Component | Part 1 (MVP) | Part 2 (enhancements) |
| --- | --- | --- |
| Model | TinyLlama-1.1B, BF16 weights and activations, FP32 accumulation | Llama-3.2-1B (GQA + RoPE scaling), Qwen2.5, 3B models on cloud GPU |
| Weight loading | safetensors via mmap, pinned staging buffer | GGUF support, async multi-stream loading |
| Kernels | RMSNorm, RoPE, SwiGLU, residual add, embedding, fused variants | Fused QKV + RoPE epilogue, FlashAttention-style prefill kernel |
| GEMM | cuBLAS (plus one naive/tiled learning kernel benchmarked against it) | INT8 or FP8 Tensor Core GEMM via `mma.sync`, CUTLASS epilogue fusion |
| Attention | Paged decode attention kernel (online softmax), simple prefill | Split-K / FlashDecoding, Tensor Core attention |
| KV cache | Block manager, block tables, free list | Prefix caching, copy-on-write, swap to host |
| Execution | CUDA Graphs for decode at fixed batch sizes | Multi-stream overlap, graph pools per batch bucket |
| Scheduling | Continuous batching with waiting/running/finished queues | Chunked prefill, priorities, preemption |
| Sampling | Greedy, temperature, top-k, top-p on GPU | Repetition penalty, speculative decoding |
| Interface | CLI + benchmark harness | HTTP/SSE server, OpenAI-compatible API, web dashboard |
| Observability | NVTX ranges, Nsight Systems/Compute reports, CSV metrics | Live metrics endpoint, dashboard charts |

### MVP definition of done

1. **Correct:** with teacher forcing on 20 fixed prompts, the engine's argmax token matches HuggingFace (BF16, same weights) at 99%+ of positions, mean absolute logit error stays under 0.05, and free-running greedy output matches HF exactly for the first 64 tokens on at least 18 of 20 prompts.
2. **Fast at batch 1:** decode reaches at least 60% of the GPU's measured DRAM bandwidth (weights read per token divided by time per token).
3. **Fast at batch N:** with 16 concurrent requests, total throughput is at least 5x the batch-1 throughput.
4. **Explained:** each kernel has a before/after Nsight Compute number, and there is a written results page with plots.
5. **Reproducible:** `cmake --build` plus one script reproduces every number on the README from a clean clone.

Part 1 ends when all five are true. Anything not on this list is Part 2, even if it is tempting.

## Hardware and toolchain

Develop everything on the RTX 5060 laptop; it is a Blackwell GPU with compute capability 12.0 (`sm_120`), so you need CUDA Toolkit 12.8 or newer. Rent a cloud GPU only for the final comparison benchmarks.

### Your GPU

| Property | RTX 5060 Laptop | Why it matters |
| --- | --- | --- |
| Architecture | Blackwell, compute capability 12.0 | Compile with `-arch=sm_120`; FP8 Tensor Core `mma.sync` is available, so the Part 2 FP8 GEMM is possible locally |
| CUDA cores | ~3,328 (26 SMs x 128) | Sets how many thread blocks run at once; pick grid sizes that are multiples of 26 |
| VRAM | 8 GB GDDR7, 128-bit bus | TinyLlama in BF16 is 2.2 GB, leaving ~4.5 GB for the KV cache and activations |
| Memory bandwidth | roughly 380 to 450 GB/s depending on memory clock (approximate; measure it) | This is the ceiling for decode. Batch-1 decode is purely bandwidth bound |
| TGP | 45 to 100 W, set by the laptop maker | Your numbers depend on power mode. Always benchmark plugged in, in max performance mode |

In week 1, write a tiny `bandwidth_test.cu` that copies a 1 GB buffer device-to-device 100 times and reports GB/s. That measured number (not the spec sheet) is the denominator for every "% of peak" claim you make.

### Operating system setup

1. **Use WSL2 with Ubuntu 24.04** on Windows 11, or dual-boot Ubuntu if you prefer. WSL2 works well for CUDA, and Nsight Compute has supported WSL2 profiling since version 2022.4.
2. **Install only the Windows NVIDIA driver.** Never install a Linux driver inside WSL; the Windows driver exposes the GPU to WSL. Inside WSL, install the `cuda-toolkit` package from NVIDIA's WSL-Ubuntu repo (toolkit only, no driver).
3. **Enable GPU performance counters** in the NVIDIA app / Control Panel developer settings ("allow access to GPU performance counters to all users"). Without this, `ncu` returns `ERR_NVGPUCTRPERM`.
4. Verify with `nvidia-smi` and `nvcc --version` inside WSL, then build and run the CUDA `deviceQuery` sample.

### Toolchain (pin these versions in the README)

| Tool | Version | Used for |
| --- | --- | --- |
| CUDA Toolkit | 12.8+ (13.x fine) | `nvcc`, cuBLAS, CUB, NVTX, `cuda-gdb`, `compute-sanitizer` |
| GCC | 13 | Host compiler, C++20 |
| CMake + Ninja | CMake 3.28+ | Build with `project(engine LANGUAGES CXX CUDA)`, `CMAKE_CUDA_ARCHITECTURES="120;89;75"` so cloud GPUs work too |
| GoogleTest | latest | C++ and kernel unit tests |
| nlohmann/json | 3.x | Parsing the safetensors JSON header |
| Nsight Systems (`nsys`) | matches toolkit | Timeline: kernel gaps, launch overhead, CPU/GPU overlap |
| Nsight Compute (`ncu`) | matches toolkit | Per-kernel: bandwidth, occupancy, stalls, roofline |
| Python 3.11 venv | torch built for CUDA 12.8 (needed for `sm_120`), `transformers`, `safetensors`, `numpy`, `matplotlib` | Reference outputs, test data generation, plots. Never on the engine's hot path |
| clangd + VS Code (Remote WSL) | latest | Editing with real CUDA autocompletion via `compile_commands.json` |

### Cloud GPUs (only for final benchmarks)

| Option | Compute capability | Use |
| --- | --- | --- |
| Colab / Kaggle T4 | 7.5 | Free sanity check that the code runs on another architecture. No FP8, lower bandwidth |
| L4 (rented) | 8.9 | Datacenter card recruiters recognize; FP8 capable |
| RTX 4090 (rented) | 8.9 | High bandwidth (~1 TB/s); great for a "scales to bigger GPUs" chart |
| A100 / H100 (rented) | 8.0 / 9.0 | Part 2 only, for a vLLM head-to-head |

Budget one or two evenings of rented time near the end of Part 1, run the full benchmark script, and save raw CSVs to the repo.

## Core concepts you must understand

One idea drives every design choice here: generating a token at batch size 1 means reading every weight from VRAM once, so decode speed is set by memory bandwidth, not by FLOPs. Everything below explains why, and what to do about it.

### 1. How a GPU runs code

- **Thread, warp, block, grid.** You launch a *grid* of *thread blocks*; each block holds up to 1,024 threads. The hardware runs threads in groups of 32 called *warps*, in lockstep (SIMT). If threads in a warp take different branches (*divergence*), both paths run one after the other.
- **SM (streaming multiprocessor).** Your GPU has 26. Each SM runs several blocks at once if registers and shared memory allow. *Occupancy* is how many warps are resident versus the maximum. More warps hide memory latency, but occupancy is a means, not a goal.
- **Kernel launch.** Launching a kernel costs a few microseconds of CPU and driver time. At hundreds of kernels per token, this adds up (see section 7).

### 2. The memory hierarchy

| Level | Size (per SM or total) | Latency | Who controls it |
| --- | --- | --- | --- |
| Registers | ~64K 32-bit registers per SM | ~1 cycle | Compiler (your local variables) |
| Shared memory / L1 | ~128 KB per SM, split between them | ~20 to 30 cycles | You (`__shared__`) |
| L2 cache | a few tens of MB, whole GPU | ~200 cycles | Hardware |
| DRAM (VRAM) | 8 GB | ~400 to 800 cycles | You (`cudaMalloc`) |

- **Coalescing.** DRAM is read in 32-byte sectors. When the 32 threads of a warp read 32 consecutive 4-byte values, that is 4 sectors, fully used. When they read scattered addresses, you pay for up to 32 sectors and use a fraction. Rule: thread `i` reads element `i` (or element `i*vec`).
- **Vectorized loads.** Reading `float4` / `uint4` (16 bytes) per thread instead of one `half` (2 bytes) cuts the instruction count 8x and helps reach peak bandwidth. It requires 16-byte aligned pointers and sizes divisible by 8 halves, so assert both.
- **Shared memory banks.** Shared memory has 32 banks; if two threads in a warp hit the same bank at different addresses, the accesses serialize (*bank conflict*). Padding arrays by one element is the usual fix.

### 3. The roofline model

Every kernel is limited either by memory bandwidth or by compute. *Arithmetic intensity* decides which.

**Formula:** AI = FLOPs performed / bytes moved to or from DRAM. Attainable FLOP/s = min(peak FLOP/s, AI x bandwidth).

A matrix-vector product with 16-bit weights does 2 FLOPs (multiply + add) per 2-byte weight, so AI is about 1 FLOP/byte. Tensor Cores need hundreds of FLOPs per byte to be busy. So batch-1 decode is deeply memory bound, and the only way to go faster is to move fewer bytes (fusion, quantization) or to reuse each weight byte for more tokens (batching). This single paragraph is the reason continuous batching and quantization exist.

### 4. The Llama forward pass (what you are implementing)

For each token position, with hidden size `d = 2048`:

1. **Embedding:** look up row `token_id` of a `[32000, 2048]` table.
2. **For each of 22 layers:**
   1. `h = RMSNorm(x)`
   2. `q = h Wq`, `k = h Wk`, `v = h Wv`
   3. Apply RoPE to `q` and `k`
   4. Append `k`, `v` to the KV cache; attention over all cached positions
   5. `x = x + attn_out Wo` (residual)
   6. `h = RMSNorm(x)`
   7. `x = x + (SiLU(h Wgate) * (h Wup)) Wdown` (SwiGLU MLP + residual)
3. **Final RMSNorm, then `lm_head`:** `[2048] x [2048, 32000]` gives logits over the vocabulary.
4. **Sample** one token id from the logits.

The formulas:

- **RMSNorm:** out_i = x_i / sqrt( (1/d) x sum_j(x_j^2) + eps ) x g_i, where g is the learned weight vector.
- **SwiGLU:** out = ( SiLU(h x W_gate) * (h x W_up) ) x W_down, where * is elementwise and SiLU(z) = z / (1 + e^(-z)).
- **RoPE:** for a pair of dimensions (a, b) in a head at position p, with theta_i = 10000^(-2i / 64): a' = a x cos(p x theta_i) - b x sin(p x theta_i), and b' = a x sin(p x theta_i) + b x cos(p x theta_i). In HF's convention, a is dimension i and b is dimension i + 32 (see the trap below).
- **Attention:** out = softmax( q x K^T / sqrt(d_head) ) x V.

**Trap:** HuggingFace Llama does *not* rotate adjacent pairs. It rotates element `i` with element `i + d_head/2` (the "rotate_half" convention), because of how the weights were permuted. Your RoPE kernel must use the same pairing or outputs will be garbage from layer 0. Write the HF-matching version and a test that proves it.

**GQA (grouped-query attention):** TinyLlama has 32 query heads but only 4 KV heads. Each KV head is shared by 8 query heads (`group = 32 / 4`). Query head `h` reads KV head `h / 8`. This shrinks the KV cache 8x.

### 5. Prefill versus decode

|  | Prefill | Decode |
| --- | --- | --- |
| Input | The whole prompt (N tokens at once) | 1 new token per sequence |
| Math | Matrix x matrix (GEMM) | Matrix x vector (GEMV), or skinny GEMM when batched |
| Bound by | Compute | Memory bandwidth |
| Metric | TTFT (time to first token) | TPOT (time per output token) |
| Attention | Causal, N x N (use a simple kernel in the MVP) | 1 query against all cached keys (your paged kernel) |

### 6. The KV cache and why it is paged

Without a cache, each new token would recompute K and V for the whole history. With it, you store them once. Size per token:

**Formula:** bytes per token = 2 (K and V) x n_layers x n_kv_heads x d_head x 2 bytes = 2 x 22 x 4 x 64 x 2 = 22,528 bytes (about 22 KB).

With ~4 GB set aside, that is about 190,000 tokens of cache, far more than one request needs. The problem is allocation: if you reserve `max_len` contiguous slots per request up front, most of it sits empty (internal fragmentation), and you can only fit a few requests. **Paging** borrows the OS idea: split the cache into fixed blocks of 16 tokens, hand out blocks on demand, and keep a per-sequence *block table* mapping logical block `i` to physical block `b`. Waste drops to under one block per sequence, so you fit many more concurrent requests. The attention kernel then reads K/V through the block table instead of from a flat array.

### 7. Launch overhead and CUDA Graphs

Rough budget at batch 1: TinyLlama is 2.2 GB of weights, so at ~400 GB/s one token takes about 5.5 ms of pure memory traffic. A layer launches about 10 to 14 kernels, so ~250 to 300 launches per token. At 3 to 5 microseconds each, that is 1 to 1.5 ms of CPU-side overhead, 20% or more of the token time. A CUDA Graph records the whole sequence of launches once and replays it with a single launch, which recovers most of that time.

### 8. Online softmax (the FlashAttention trick)

Naive softmax needs three passes: find the max, sum the exponentials, divide. Online softmax does it in one pass over chunks by keeping a running max `m`, running sum `l`, and running output `o`, rescaling when a new max appears:

For each new chunk of scores s_j with values v_j:

1. m' = max(m, max of the chunk's scores)
2. l' = e^(m - m') x l + sum_j e^(s_j - m')
3. o' = e^(m - m') x o + sum_j e^(s_j - m') x v_j
4. After the last chunk, the output is o' / l'.

This lets the attention kernel stream through KV blocks once without ever storing the score vector for a long context.

### 9. Continuous batching

Static batching waits for a whole batch to finish before starting new requests, so short requests wait on long ones and the GPU idles. Continuous (iteration-level) batching reschedules on every decode step: finished sequences leave, waiting ones join, and every step reads the weights once for all active sequences. Because decode is memory bound, going from 1 to 16 sequences costs little extra time per step, so throughput scales almost linearly until you become compute bound or run out of KV blocks.

### 10. Numerics

- Store weights and activations in **BF16** (TinyLlama's native dtype), accumulate every dot product and reduction in **FP32**.
- Floating-point addition is not associative. Your sums happen in a different order than PyTorch's, so logits will differ by a small amount (typically under 0.05). That is why correctness is tested with tolerances and teacher forcing (see Testing), not by demanding bit-identical output.

## Model and system architecture

The MVP targets [TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0): open weights (no gated access), plain Llama architecture, GQA, no RoPE scaling, and small enough to fit on 8 GB with lots of room for KV cache.

### Model config ([config.json](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/raw/main/config.json))

| Field | Value | Derived |
| --- | --- | --- |
| `hidden_size` | 2048 | `d_head = 2048 / 32 = 64` |
| `num_hidden_layers` | 22 |  |
| `num_attention_heads` | 32 |  |
| `num_key_value_heads` | 4 | GQA group size 8; KV width = 4 x 64 = 256 |
| `intermediate_size` | 5632 | MLP expansion ~2.75x |
| `vocab_size` | 32000 | Embedding and `lm_head` are separate (`tie_word_embeddings: false`) |
| `rope_theta` | 10000.0 | `rope_scaling: null` |
| `rms_norm_eps` | 1e-5 |  |
| `max_position_embeddings` | 2048 | Cap context at 2048 in the MVP |
| `torch_dtype` | bfloat16 |  |

### Weight tensors (safetensors names and shapes)

PyTorch `nn.Linear` stores weights as `[out_features, in_features]`, row-major, and computes `y = x W^T`.

| Tensor | Shape | Per layer? |
| --- | --- | --- |
| `model.embed_tokens.weight` | [32000, 2048] | no |
| `model.layers.{i}.input_layernorm.weight` | [2048] | yes |
| `model.layers.{i}.self_attn.q_proj.weight` | [2048, 2048] | yes |
| `model.layers.{i}.self_attn.k_proj.weight` | [256, 2048] | yes |
| `model.layers.{i}.self_attn.v_proj.weight` | [256, 2048] | yes |
| `model.layers.{i}.self_attn.o_proj.weight` | [2048, 2048] | yes |
| `model.layers.{i}.post_attention_layernorm.weight` | [2048] | yes |
| `model.layers.{i}.mlp.gate_proj.weight` | [5632, 2048] | yes |
| `model.layers.{i}.mlp.up_proj.weight` | [5632, 2048] | yes |
| `model.layers.{i}.mlp.down_proj.weight` | [2048, 5632] | yes |
| `model.norm.weight` | [2048] | no |
| `lm_head.weight` | [32000, 2048] | no |

Parameter count: about 44M per layer x 22 = 968M, plus 65.5M each for embedding and `lm_head`, total ~1.1B, which is 2.2 GB in BF16. Per decoded token you read everything except the embedding table (only one row), so **~2.07 GB of weight traffic per decode step**. Divide that by your measured step time to get achieved bandwidth.

**Free win at load time:** concatenate `q_proj`, `k_proj`, `v_proj` into one `[2560, 2048]` matrix, and `gate_proj`, `up_proj` into one `[11264, 2048]` matrix. That turns 5 GEMMs per layer into 2 and costs nothing at runtime.

### Calling cuBLAS with row-major weights

cuBLAS is column-major. A row-major `[out, in]` matrix, read as column-major, is its transpose. For `Y[B, out] = X[B, in] W^T`, ask cuBLAS for the column-major product `Y^T = W X^T`:

```cpp
// W: row-major [out, in], X: row-major [B, in], Y: row-major [B, out]
cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
             out, B, in,
             &alpha, W, CUDA_R_16BF, in,
                     X, CUDA_R_16BF, in,
             &beta,  Y, CUDA_R_16BF, out,
             CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
```

Getting this wrong is the most common day-1 bug. Unit test it against a CPU matmul on a 3x5 example before anything else.

### Tokenizer

Do not write a tokenizer. TinyLlama ships `tokenizer.model` (SentencePiece). Link Google's SentencePiece C++ library and call `Encode` / `Decode`. Remember to prepend BOS (id 1) and stop on EOS (id 2). For the chat model, apply the Zephyr-style chat template (`<|user|>`, `<|assistant|>`) in a small helper, and test it against `tokenizer.apply_chat_template` in Python.

### System architecture

The engine has seven parts and data flows through them in this order:

1. **CLI / bench harness** submits requests to the scheduler.
2. **Scheduler** keeps the waiting, running and finished queues and decides each step's batch.
3. **KV block manager** (free list, block tables) gives every sequence cache slots.
4. **Model runner** runs one prefill or decode step, using the **weight loader**'s weights (mmap + pinned copy).
5. **CUDA Graph cache** replays a captured decode step when the batch size matches a bucket.
6. **Kernels** (norm, RoPE, attention, SwiGLU) plus **cuBLAS GEMMs** do the math.
7. **Sampler** returns one token per sequence to the scheduler, and the loop repeats.

Each decode step, the scheduler picks the running sequences, the block manager makes sure each has a free slot for its next token, the model runner launches one forward pass for the whole batch (as a graph replay when the batch size matches a captured graph), the sampler returns one token per sequence, and the scheduler moves finished sequences out.

### Repository layout

```
engine/
  CMakeLists.txt
  README.md                 results table, plots, how to reproduce
  docs/                     design notes, results.md, profiling screenshots
  include/engine/           public headers
  src/
    core/                   tensor.h, device buffer RAII, cuda_check.h, logging
    loader/                 safetensors.cpp (mmap + header parse), weights.cpp
    kernels/                rmsnorm.cu, rope.cu, swiglu.cu, embedding.cu,
                            paged_attention.cu, prefill_attention.cu, sampling.cu
    model/                  llama.cpp (forward pass), config.cpp
    kv/                     block_manager.cpp
    runtime/                model_runner.cpp, cuda_graphs.cpp
    sched/                  scheduler.cpp, request.h
    sampling/               sampler.cpp
    tokenizer/              sp_tokenizer.cpp (SentencePiece wrapper)
    app/                    main.cpp (CLI), bench.cpp
  tests/
    unit/                   gtest per kernel and per C++ component
    golden/                 reference tensors dumped from HF (git-ignored, regenerated by script)
  scripts/
    download_model.sh
    dump_reference.py       HF hidden states / logits per layer
    compare_logits.py
    bench.py                runs the engine, writes CSV
    plot.py
  third_party/              sentencepiece, googletest, nlohmann_json (git submodules or FetchContent)
```

## Part 1 (MVP): schedule and milestones

The MVP is built in five milestones over eight weeks, and the rule is **correct first, then fast**: nothing gets optimized until the naive engine produces the same text as HuggingFace. Person A owns kernels and numerics; Person B owns the C++ runtime (loader, KV manager, scheduler, graphs). Both of you pair on the first two weeks because you are both new to C++ and CUDA.

### Milestones

| Milestone | Due | Exit criteria (all must be true) |
| --- | --- | --- |
| M1: Foundations | Oct 4, 2026 | CMake project builds for `sm_120`; bandwidth test prints a GB/s number; cuBLAS GEMM test passes against CPU; safetensors loader prints every tensor name, shape, dtype and a checksum matching Python |
| M2: Correct naive engine | Oct 18, 2026 | Batch-1 greedy generation of TinyLlama produces readable text; per-layer hidden states match HF within tolerance; teacher-forced argmax match is 99%+ |
| M3: Paged KV + batching | Nov 1, 2026 | Block manager with unit tests; paged decode attention kernel matches the naive kernel; scheduler serves 16 concurrent requests with correct outputs for each |
| M4: Fast | Nov 8, 2026 | Vectorized/fused kernels in; CUDA Graphs for decode; batch-1 decode at 60%+ of measured bandwidth; nsys timeline shows no launch gaps |
| M5: Shipped | Nov 15, 2026 | Benchmark CSVs (laptop + one cloud GPU), plots, README with results and reproduce steps, resume bullets written from real numbers |

### Week-by-week plan

| Week | Dates | Person A (kernels) | Person B (runtime) | Together |
| --- | --- | --- | --- | --- |
| 0 | Sep 23 to Sep 27 | Set up WSL2, CUDA, `deviceQuery`; read PMPP ch. 1 to 5 | Same setup; C++ refresher: RAII, `std::unique_ptr`, move semantics, `std::span` | Create repo, CMake skeleton, CI that compiles; agree on coding style |
| 1 | Sep 28 to Oct 4 | `bandwidth_test.cu`; naive then vectorized vector-add; first warp-shuffle reduction; `CUDA_CHECK` macro | `DeviceBuffer` RAII class, `Tensor` view (ptr, shape, strides, dtype); safetensors mmap loader; checksum tool | cuBLAS GEMM wrapper + CPU test (row-major trick). **M1** |
| 2 | Oct 5 to Oct 11 | Naive kernels: embedding, RMSNorm, RoPE (HF rotate_half), SiLU*mul, residual add, naive attention over a contiguous KV cache | `dump_reference.py` (HF hidden states per layer to `.bin`); C++ golden-test harness; model config + weight struct with fused QKV / gate-up | Wire the forward pass for one token, compare layer 0 then all 22 layers |
| 3 | Oct 12 to Oct 18 | Simple causal prefill attention kernel (one block per head per query row is fine); fix numerics bugs | Tokenizer (SentencePiece) + chat template; CLI `generate` command; greedy + temperature sampling on GPU | Debug until text matches HF. **M2**. (Thanksgiving Oct 12 and likely midterms: this week has buffer on purpose) |
| 4 | Oct 19 to Oct 25 | Paged decode attention kernel v1 (block table lookups, online softmax, GQA) with unit tests vs naive | KV block manager (free list, block tables, allocate/append/free); KV write kernel interface (`slot_mapping`) | Swap contiguous KV for paged KV in the forward pass; outputs unchanged |
| 5 | Oct 26 to Nov 1 | Batched versions of all kernels (`[B, d]` inputs, per-sequence positions and lengths); top-k/top-p sampling | Continuous batching scheduler (waiting/running/finished, admission by free blocks); request abstraction; bench harness with synthetic workloads | 16 concurrent requests produce the same text as 16 sequential runs. **M3** |
| 6 | Nov 2 to Nov 8 | Optimize: vectorized 128-bit loads, fused residual-add + RMSNorm, fused SwiGLU, paged attention tuning; `ncu` before/after for each | CUDA Graph capture per batch-size bucket (1, 2, 4, 8, 16), static input buffers, NVTX ranges; `nsys` timelines | Batch-1 bandwidth target hit. **M4** |
| 7 | Nov 9 to Nov 15 | Cloud benchmark run (L4 or 4090); roofline plot | HF baseline + llama.cpp baseline runs; `plot.py`; README | Write `docs/results.md`, resume bullets, record a 60-second demo GIF. **M5** |

### If you fall behind

Cut in this order, and never cut correctness tests:

1. Top-k/top-p sampling (keep greedy + temperature).
2. CUDA Graphs (move to Part 2; batching alone still gives the big throughput number).
3. Cloud benchmark (report laptop numbers only).
4. Fused kernels beyond RMSNorm (keep the unfused but vectorized versions).

Paged attention and continuous batching are the core of the project. If those slip, move the deadline rather than drop them.

## Part 1 (MVP): component specs

Each component below lists what it does, its interface, how to implement it, and how you know it works. Build them in the order listed.

### C1. Core infrastructure (week 1)

- **`CUDA_CHECK(expr)` and `CUBLAS_CHECK(expr)` macros** that print file, line, and the error string, then abort. Wrap every CUDA call. After each kernel launch in debug builds, call `cudaGetLastError()` then `cudaDeviceSynchronize()` so errors point at the right kernel.
- **`DeviceBuffer`**: a move-only RAII owner of a `cudaMalloc` pointer (`cudaFree` in the destructor, copy constructor deleted). This one class removes most memory leaks.
- **`Tensor`**: a non-owning view: `void* data`, `DType dtype`, `std::array<int64_t,4> shape`, `int ndim`. Keep it dumb: no autograd, no strides beyond contiguous in the MVP.
- **Workspace arena**: at startup, allocate every activation buffer once, sized for the maximum tokens per step (e.g. 2,048 tokens for prefill, 32 sequences for decode). Never `cudaMalloc` during generation. This is also a hard requirement for CUDA Graphs, because a captured graph bakes in pointer addresses.
- **One CUDA stream** for compute in the MVP. Pass it explicitly to every kernel and cuBLAS call (`cublasSetStream`).
- **Debug tooling:** `compute-sanitizer --tool memcheck` on every unit test binary in CI-like local runs catches out-of-bounds reads that would otherwise produce silently wrong text.

### C2. Weight loader (week 1)

**Format.** A `.safetensors` file is: 8 bytes little-endian `uint64` header length `N`, then `N` bytes of JSON, then raw tensor bytes. The JSON maps each name to `{"dtype": "BF16", "shape": [2048, 2048], "data_offsets": [begin, end]}` (offsets relative to the end of the header), plus an optional `__metadata__` key to skip.

**Implementation.**

1. `open` + `mmap(PROT_READ, MAP_PRIVATE)` the file. Nothing is read from disk yet; pages load on first touch.
2. Parse the header with nlohmann/json. Validate that each dtype is `BF16` and each shape matches the config.
3. Allocate **one** `DeviceBuffer` for all weights (plus 256-byte alignment padding per tensor so vectorized loads are always aligned). Record each tensor's device offset.
4. Copy host to device. Pageable `mmap` memory is not pinned, so a plain `cudaMemcpy` silently goes through a driver staging buffer. Better: allocate two 64 MB pinned buffers with `cudaMallocHost`, then loop: `memcpy` a chunk from the mapping into pinned buffer `i % 2`, `cudaMemcpyAsync` it to the device, and record an event so you do not overwrite a buffer still in flight. That double-buffers disk reads against PCIe transfers.
5. At copy time, write `q/k/v_proj` into adjacent slices of one fused `[2560, 2048]` tensor, and `gate/up_proj` into one fused `[11264, 2048]` tensor.
6. `munmap` and free the pinned buffers.

**Verify.** A `checksum` command prints, per tensor, the sum of all elements as FP64. `scripts/checksum.py` prints the same from PyTorch. They must match to 6+ significant digits. Also log load time and effective GB/s; this is a small resume-able number ("loads 2.2 GB of weights in X s at Y GB/s").

### C3. Kernels (weeks 2 to 6)

Write every kernel twice: first a naive version that is obviously correct, then an optimized one that is tested against the naive one. Keep both in the repo; the naive one is your test oracle and your "before" number.

**Shared helper: warp reduction.** Every reduction kernel uses this. `__shfl_xor_sync` leaves the full sum in every lane.

```cpp
__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        x += __shfl_xor_sync(0xffffffff, x, offset);
    return x;
}
```

| Kernel | Inputs, outputs | Launch config | Notes |
| --- | --- | --- | --- |
| `embedding` | `ids[T]` to `x[T, 2048]` | grid `T`, block 256 | Each thread copies one `uint4` (8 BF16). Pure memory copy |
| `add_rmsnorm` (fused) | `residual[T, 2048]` (in/out), `delta[T, 2048]`, `w[2048]` to `out[T, 2048]` | grid `T`, block 256 | 256 threads x 8 elements = 2048, one `uint4` per thread. Fuses the residual add into the norm and saves a full read + write of `x` per layer |
| `rope` (in place) | `qkv[T, 2560]`, `positions[T]`, `cos/sin[max_pos, 32]` table | grid `T`, block 64 per head group | HF rotate_half pairing: element `i` with `i + 32` inside each 64-wide head. Apply to 32 Q heads and 4 K heads, not V. Precompute the cos/sin table in FP32 once at startup |
| `write_kv` | K, V slices of `qkv`, `slot_mapping[T]` to paged cache | grid `T`, block 256 | `slot = block_id * 16 + offset`, computed on the CPU by the block manager |
| `swiglu` | `gu[T, 11264]` to `h[T, 5632]` | grid-stride, vectorized | `h = silu(g) * u` where `g` is the first 5632 columns, `u` the second |
| `paged_attention_decode` | `q[B, 32, 64]`, K/V cache, `block_tables[B, max_blocks]`, `context_lens[B]` to `out[B, 32, 64]` | grid `(32 heads, B)`, block 128 | Detailed below |
| `prefill_attention` | same as decode, but many queries per sequence | grid `(32 heads, T)` | MVP trick: treat each prompt token as its own "decode" query with context length `pos + 1` and reuse the decode kernel logic. O(N^2) reads, fine for prompts under 2,048 |
| `argmax` / `sample` | `logits[B, 32000]` to `ids[B]` | grid `B`, block 1024 | See the sampler in the next part |

**`add_rmsnorm` in detail** (the template for every other elementwise kernel):

```cpp
template <int D, int THREADS>
__global__ void add_rmsnorm(__nv_bfloat16* __restrict__ out,
                            __nv_bfloat16* __restrict__ residual,
                            const __nv_bfloat16* __restrict__ delta,
                            const __nv_bfloat16* __restrict__ weight,
                            float eps) {
    constexpr int VEC = 8;
    static_assert(D == THREADS * VEC);
    const size_t off = size_t(blockIdx.x) * D + threadIdx.x * VEC;

    uint4 r = *reinterpret_cast<const uint4*>(residual + off);
    uint4 d = *reinterpret_cast<const uint4*>(delta + off);
    auto* rh = reinterpret_cast<__nv_bfloat16*>(&r);
    auto* dh = reinterpret_cast<const __nv_bfloat16*>(&d);

    float v[VEC], ss = 0.f;
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
        // Round the sum to BF16 first: HF adds the residual in BF16, then normalizes that.
        rh[i] = __float2bfloat16(__bfloat162float(rh[i]) + __bfloat162float(dh[i]));
        v[i] = __bfloat162float(rh[i]);
        ss += v[i] * v[i];
    }
    *reinterpret_cast<uint4*>(residual + off) = r;

    __shared__ float partial[THREADS / 32];
    __shared__ float inv_rms;
    ss = warp_sum(ss);
    if (threadIdx.x % 32 == 0) partial[threadIdx.x / 32] = ss;
    __syncthreads();
    if (threadIdx.x < 32) {
        float t = threadIdx.x < THREADS / 32 ? partial[threadIdx.x] : 0.f;
        t = warp_sum(t);
        if (threadIdx.x == 0) inv_rms = rsqrtf(t / D + eps);
    }
    __syncthreads();

    uint4 w = *reinterpret_cast<const uint4*>(weight + threadIdx.x * VEC);
    auto* wh = reinterpret_cast<const __nv_bfloat16*>(&w);
    uint4 o;
    auto* oh = reinterpret_cast<__nv_bfloat16*>(&o);
#pragma unroll
    for (int i = 0; i < VEC; ++i)
        oh[i] = __float2bfloat16(__bfloat162float(__float2bfloat16(v[i] * inv_rms)) * __bfloat162float(wh[i]));
    *reinterpret_cast<uint4*>(out + off) = o;
}
```

The odd double rounding on the last line is deliberate: HF casts the normalized value to BF16 *before* multiplying by the weight. Matching these small details is what gets you from "close" to "argmax matches 99%+".

Note for Claude: `add_rmsnorm` is a Part C item that Joe writes himself. Do not hand him this code; the spec block for his stub must describe requirements only (see `delegation.md`).

**`paged_attention_decode` in detail.**

*Cache layout (per layer):* `k_cache[num_blocks, 4 kv_heads, 16 tokens, 64 dims]`, same for `v_cache`. One (block, head) tile is 16 x 64 BF16 = 2 KB, contiguous, so a warp reads it fully coalesced.

*v1 (week 4), simple and correct:*

1. One thread block per (query head `h`, sequence `b`); 128 threads = 4 warps. `kvh = h / 8`.
2. Load the 64-dim query into shared memory, pre-scaled by `1 / sqrt(64)`.
3. **Scores:** loop over logical blocks `j = 0 .. ceil(len / 16)`; physical block `p = block_tables[b][j]`. Each warp takes one token at a time: 32 lanes x 2 dims = 64, dot product, `warp_sum`. Write the score to a shared array `s[2048]` (8 KB of FP32, fine because the MVP caps context at 2,048).
4. **Softmax** over `s[0 .. len)` in shared memory: block-wide max, then `exp(s - max)`, block-wide sum.
5. **Output:** each of the first 64 threads owns one output dimension and accumulates `sum_t s[t] * V[t][dim]` across all tokens (read through the block table), then divides by the sum and writes BF16.

*v2 (week 6), faster:* switch step 3 to 5 to a single pass with **online softmax** (running max, sum, and output accumulator; formula in the concepts section), so scores never need to be stored and the kernel works for any context length. Then launch one thread block per **KV head** instead of per query head, so each K/V tile is loaded from DRAM once and reused by all 8 query heads in the group (GQA-aware). Measure both with `ncu`; the GQA change alone should cut attention DRAM traffic about 8x at long contexts.

*Tests:* random Q/K/V, random block tables (shuffled physical blocks), context lengths including 1, 15, 16, 17, and 2,048 (the off-by-one boundaries). Compare against a naive kernel on contiguous tensors with tolerance 1e-2.

**GEMMs.** Use `cublasGemmEx` (BF16 in, FP32 compute, BF16 out) for all projections and `lm_head`. As a learning exercise, also write a tiled shared-memory GEMM (Person A, week 5 or 6 if time allows), benchmark it against cuBLAS, and put the honest comparison in the results page (for example "my tiled kernel reaches 35% of cuBLAS"). Interviewers respect knowing where your code stands.

### C4. KV block manager (week 4)

A pure C++ class with no CUDA inside it, which makes it easy to unit test exhaustively.

**Sizing at startup.** Load weights and allocate the workspace first, then call `cudaMemGetInfo`. Give 90% of the remaining free memory to the KV cache (leave the rest for the cuBLAS workspace and CUDA Graphs). One block holds 16 tokens for all layers, K and V: `2 x 22 x 4 x 16 x 64 x 2 bytes = 360,448 bytes` (~352 KB). With ~4 GB free that is roughly 11,000 blocks, or ~180,000 tokens.

```cpp
class BlockManager {
public:
    BlockManager(int num_blocks, int block_size);

    bool can_allocate(int num_tokens) const;   // enough free blocks for a prompt?
    void allocate(SeqId seq, int num_tokens);  // reserve blocks for the prompt
    bool can_append(SeqId seq) const;          // room for one more token?
    int  append_slot(SeqId seq);               // returns physical slot for the next token
    void free(SeqId seq);                      // return all blocks to the free list

    int slot(SeqId seq, int pos) const;        // table[pos / bs] * bs + pos % bs
    const std::vector<int>& table(SeqId seq) const;
    int num_free() const;

private:
    int bs_;
    std::vector<int> free_;                     // stack of physical block ids
    std::vector<int> refs_;                     // ref counts, used by Part 2 copy-on-write
    std::unordered_map<SeqId, std::vector<int>> tables_;
    std::unordered_map<SeqId, int> lens_;
};
```

**Rules.** A new block is taken only when a sequence's length crosses a multiple of 16. `free` pushes blocks back onto the stack. Never hand out block 0 as a real block; reserve it as a "null" block so padded batch slots have a safe place to point.

**Tests (gtest).** Allocate and free in random orders for 10,000 iterations and assert that no block is ever owned by two sequences and that `num_free + owned == num_blocks` at every step. Test exact boundaries: lengths 15, 16, 17.

### C5. Model runner (weeks 2 to 5)

The model runner turns one scheduler decision into one forward pass. Every step, the host fills a packed input struct in pinned memory and copies it to fixed device buffers with a single `cudaMemcpyAsync`:

| Field | Shape | Meaning |
| --- | --- | --- |
| `token_ids` | [T] | New tokens this step (all prompt tokens for prefill, 1 per sequence for decode) |
| `positions` | [T] | Absolute position of each token, for RoPE |
| `slot_mapping` | [T] | Physical cache slot for each token's K/V, `-1` for padding |
| `block_tables` | [B, 128] | Physical block ids per sequence (128 blocks x 16 = 2,048 max context) |
| `context_lens` | [B] | Tokens in the cache per sequence after this step, `0` for padding |
| `logit_rows` | [B] | Which of the `T` rows produces each sequence's logits (its last token) |

**Forward pass (per step):**

```
x = embedding(token_ids)
delta = 0
for each layer l:
    h = add_rmsnorm(x, delta, ln1[l])
    qkv = gemm(h, Wqkv[l])                  # [T, 2560]
    rope(qkv, positions)
    write_kv(qkv, slot_mapping, kcache[l], vcache[l])
    attn = attention(qkv, kcache[l], vcache[l], block_tables, context_lens)
    delta = gemm(attn, Wo[l])
    h = add_rmsnorm(x, delta, ln2[l])
    gu = gemm(h, Wgu[l])                    # [T, 11264]
    m = swiglu(gu)                          # [T, 5632]
    delta = gemm(m, Wdown[l])
h = add_rmsnorm(x, delta, final_norm)
logits = gemm(gather(h, logit_rows), Wlm)   # only B rows, not T
ids = sample(logits, sampling_params)
```

That is 10 kernels per layer, ~225 launches per step. Two details matter: compute `lm_head` only for the last token of each sequence (during a 1,000-token prefill this saves 999 x 131 MB of wasted reads), and wrap each line above in an NVTX range so it shows up by name in Nsight Systems.

**MVP simplification:** a step is either all-prefill or all-decode, never mixed. This causes decode stalls whenever a long prompt arrives, which is exactly the problem chunked prefill fixes in Part 2. Measure it so you can show the fix later.

### C6. CUDA Graphs (week 6)

1. **Buckets.** Capture one decode graph per batch size in {1, 2, 4, 8, 16, 32}. At run time, pad the batch up to the next bucket; padded entries have `context_len = 0` and `slot = -1`, and every kernel must early-exit on those.
2. **Static everything.** All kernel arguments must be device pointers to fixed buffers (the input struct above). Values that change each step (`context_lens`, positions) are read from device memory inside kernels, never passed as scalar kernel arguments, because captured arguments are frozen.
3. **Warm up first.** Run each bucket once without capture so cuBLAS selects its algorithms and CUDA lazily loads modules. Call `cublasSetWorkspace` with a preallocated buffer so cuBLAS does not allocate during capture.
4. **Capture:** `cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal)`, run the decode forward pass, `cudaStreamEndCapture`, then `cudaGraphInstantiate`.
5. **Replay:** copy the new inputs into the static buffers (outside the graph), then `cudaGraphLaunch(exec, stream)`.
6. **Verify:** graph and non-graph outputs must be bit-identical. In `nsys`, the per-kernel launch gaps disappear. Report TPOT with and without graphs at batch 1.

Prefill is not captured (its shape changes every time); it runs eagerly.

### C7. Continuous batching scheduler (week 5)

```cpp
struct Request {
    int id;
    std::vector<int> prompt, output;
    SamplingParams params;          // temperature, top_k, top_p, max_new_tokens, seed
    enum class State { Waiting, Running, Finished } state;
    Clock::time_point arrived, first_token, finished;
};
```

**One iteration of the loop:**

1. **Retire:** move sequences that hit EOS or `max_new_tokens` to `finished`; free their blocks.
2. **Admit:** while `waiting` is non-empty, the running batch is under `max_batch`, `can_allocate(prompt_len + 1)` holds, and the prefill token budget (2,048) is not exceeded, move a request to `running` and allocate its blocks.
3. **Run:** if anything was admitted, run one prefill step for those requests (concatenated tokens); otherwise run one decode step for all running sequences.
4. **Guarantee space:** before a decode step, every running sequence needs `can_append`. If the cache is full, **preempt** the most recently admitted sequence: free its blocks, append its generated tokens to its prompt, and push it back to the front of `waiting`. It will be re-prefilled later (recompute-style preemption).
5. **Record timing:** `first_token` on the first sampled token, `finished` at retirement.

**Test:** submit 16 requests with different prompt lengths and `max_new_tokens`, greedy sampling. Each output must equal the output of running that request alone. Then shrink the KV cache to force preemption and check outputs are still identical.

### C8. Sampler (weeks 3 and 5)

One thread block (1,024 threads) per sequence, all on the GPU, no host round-trip of logits.

- **Greedy:** block-wide argmax over 32,000 logits.
- **Temperature:** use the **Gumbel-max trick**: `argmax(logit / T + g)` where `g = -log(-log(u))`, `u` uniform in (0, 1) from a counter-based RNG (`curand_init(seed, seq_id, step)` with Philox). This samples exactly from `softmax(logit / T)` in one pass without normalizing.
- **Top-k (no sorting):** binary-search a threshold value `t` between min and max logit (~24 iterations), counting how many logits are `>= t` with a block reduction each iteration, until the count is about `k`. Mask everything below `t`, then Gumbel-max. The 128 KB logit row stays in L2, so each iteration is cheap.
- **Top-p:** same idea on probabilities: find the threshold where the mass of tokens at or above it first reaches `p`.
- **Seeded determinism:** the same seed gives the same text, which makes sampling testable.

### C9. CLI and benchmark harness (weeks 3 and 7)

Use CLI11 (header-only) for argument parsing.

```
engine generate --model models/tinyllama --prompt "Explain KV caching" --max-tokens 256 --temperature 0.7 --top-p 0.9
engine check    --model models/tinyllama --golden tests/golden           # correctness suite
engine bench    --model models/tinyllama --num-requests 64 --prompt-len 128 --output-len 256 \
                --max-batch 16 --graphs on --out results/laptop_b16.csv
```

`bench` writes one CSV row per request (TTFT, TPOT, total latency, tokens) plus a summary row (throughput in tokens/s, achieved GB/s at batch 1, peak KV blocks used). Every plot and every resume number comes from these CSVs.

## Testing, correctness, and benchmarking

Correctness is proven in four layers (kernel, layer, model, system), and speed is proven by seven benchmark experiments whose CSVs live in the repo. No optimization gets merged without passing the tests and showing a before/after number.

### Test layers

| Layer | What is compared | Tool | Pass condition |
| --- | --- | --- | --- |
| Kernel unit tests | Optimized kernel vs naive kernel vs CPU FP32 reference, random inputs, edge sizes | GoogleTest + `compute-sanitizer` | Max abs error under 1e-2 (BF16 outputs); zero sanitizer errors |
| C++ component tests | Block manager, safetensors parser, scheduler logic | GoogleTest | All invariants hold for 10,000 random operations |
| Layer golden tests | Hidden state after each of the 22 layers vs HF dumps | `engine check` | Relative error (norm of difference / norm of reference) under 1e-2 at layer 1, under 5e-2 at layer 22 |
| Model tests | Teacher-forced logits and free-running greedy text vs HF | `engine check` + `compare_logits.py` | The MVP "Correct" criteria in the Overview |
| System tests | 16 concurrent requests vs 16 solo runs, with and without forced preemption, graphs on and off | `engine check --batching` | Identical token ids |

### Generating reference data

`scripts/dump_reference.py` loads the HF model in BF16 on the GPU, registers a forward hook on every decoder layer, runs the 20 fixed prompts, and saves `input_ids`, each layer's output hidden states, and the final logits as raw little-endian `.bin` files plus a `manifest.json` (name, dtype, shape). The C++ harness reads the manifest, so there is no Python in the test loop. Golden files are git-ignored and regenerated by the script.

```python
hidden = {}
def hook(idx):
    def fn(module, args, output):
        hidden[idx] = output[0].detach().to(torch.float32).cpu().numpy()
    return fn
for i, layer in enumerate(model.model.layers):
    layer.register_forward_hook(hook(i))
```

### Debugging method when outputs are wrong

1. Compare the embedding output first, then layer 0, then find the first layer that diverges.
2. Inside that layer, dump every intermediate (norm, qkv, after RoPE, attention, o_proj, MLP) from both sides and find the first op that diverges.
3. Reproduce that op alone in a unit test with the exact dumped inputs.
4. In debug builds, fill every new buffer with NaN (`cudaMemset` to `0xFF`) so reads of uninitialized memory show up immediately instead of looking "almost right".

**The usual suspects, in order of likelihood:** row-major vs column-major in cuBLAS; RoPE pairing (`i` with `i + 32`, not adjacent); GQA head mapping (`q_head / 8`); positions off by one; forgetting BOS; wrong `eps`; missing `__syncthreads()` between shared-memory write and read; BF16 rounding order differences.

### Benchmark rules

- Laptop plugged in, maximum performance power mode, other GPU apps closed.
- Log clocks, power, and temperature during every run: `nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv -lms 500`. Laptop GPUs throttle; this log explains outliers.
- Warm up for 3 iterations, then run 5 times and report the **median** (and p90 for latency).
- Time kernels with `cudaEvent`s; time end-to-end with a host clock after `cudaStreamSynchronize`.
- For comparisons against other engines: same prompts, same output length (ignore EOS and force a fixed number of tokens), greedy decoding, same dtype, and record every library version.

### Metric definitions

| Metric | Definition | Why it matters |
| --- | --- | --- |
| TTFT | Request arrival to first token | User-perceived responsiveness, dominated by prefill |
| TPOT | (last token time - first token time) / (output tokens - 1) | Streaming speed, dominated by decode |
| Throughput | Total output tokens / wall time, all requests | Serving capacity |
| Achieved bandwidth | 2.07 GB / TPOT at batch 1 | The honest measure of decode efficiency |
| % of peak | Achieved bandwidth / your measured `bandwidth_test` number | The headline number for kernel quality |
| Max concurrency | Largest number of 2,048-token requests that fit in the KV cache | Shows the value of paging |

### The seven experiments

| # | Experiment | Compares | Output |
| --- | --- | --- | --- |
| E1 | Kernel microbenchmarks | Naive vs optimized, each kernel, at T = 1, 16, 512 | Table of GB/s and % of peak |
| E2 | Batch-1 decode speed | Your engine vs HF `generate` (eager, BF16) vs llama.cpp (F16 GGUF, `llama-bench`) | TPOT and tokens/s |
| E3 | Batching scaling | Throughput at batch 1, 2, 4, 8, 16, 32 | Line plot, throughput vs batch |
| E4 | CUDA Graphs | Graphs off vs on, batch 1 and 16 | TPOT before/after, `nsys` screenshots |
| E5 | Attention scaling | Paged attention v1 vs v2 at context 128 to 2,048 | Line plot, time vs context |
| E6 | Paging benefit | Contiguous max-length allocation vs paged | Max concurrent requests that fit |
| E7 | Portability | Repeat E2 and E3 on one rented GPU (L4 or 4090) | Same tables, second hardware column |

### Profiling workflow

```
# Whole-run timeline: launch gaps, CPU overhead, NVTX-named phases
nsys profile --trace=cuda,nvtx,osrt -o reports/decode_b1 ./engine bench --num-requests 1 --output-len 128

# One kernel in depth: bandwidth, occupancy, stall reasons, roofline
ncu --set full -k regex:add_rmsnorm -c 5 -o reports/rmsnorm ./engine bench --num-requests 1 --output-len 8
```

In Nsight Compute, look at four things for every kernel: **DRAM throughput** (% of peak), **Memory Workload Analysis** (are loads coalesced, bytes per request), **Warp State Statistics** (top stall reason), and the **roofline chart** (where the kernel sits relative to the bandwidth line). Save a screenshot of each before and after optimizing; these become figures in `docs/results.md`.

## Team split and workflow

Split by layer (A: kernels and numerics, B: runtime and systems), but make sure each of you writes at least one kernel and one runtime component, because each of you will be interviewed on the whole project, not just your half.

### Ownership

| Area | Owner | Reviewer | Cross-over task for the other person |
| --- | --- | --- | --- |
| Kernels (norm, RoPE, SwiGLU, embedding, KV write) | A | B | B writes `swiglu` and `embedding` |
| Attention (decode v1/v2, prefill) | A | B | B writes the attention unit tests |
| Sampler kernels | A | B |  |
| Loader, tensor/buffer infra | B | A | A writes the checksum tool |
| Block manager, scheduler | B | A | A writes the preemption test |
| CUDA Graphs, NVTX, bench harness | B | A | A runs and reports E4 |
| Reference scripts, golden tests | Shared |  | Pair on the first golden comparison |
| README, results page, plots | Shared |  | Each writes up the parts they own |

### Interfaces to agree on in week 0

Agreeing on these on day one lets you work in parallel without blocking each other:

1. `Tensor` struct and `DType` enum.
2. Kernel launcher signature style: `void launch_add_rmsnorm(Tensor out, Tensor residual, Tensor delta, Tensor weight, float eps, cudaStream_t s);` (launchers are host functions in `.cu` files, declared in headers, so the runtime never includes CUDA kernel code).
3. KV cache layout: `[num_blocks, kv_heads, block_size, head_dim]` per layer, separate K and V.
4. The `StepInput` struct fields (the model runner table above).

Until the real piece exists, each person codes against a stub (for example, B's model runner calls a naive attention function that A later replaces).

### Git workflow

- `main` is always green: builds, and every test passes.
- One branch per task (`kernel/rmsnorm-vec`, `runtime/block-manager`), small PRs (under ~400 lines), the other person reviews within 24 hours.
- PR template: what changed, how it was tested, before/after numbers if it is a performance change, `ncu`/`nsys` screenshot if relevant.
- CI (GitHub Actions, CPU runner): configure, compile all targets for `sm_120`, run the pure C++ tests (block manager, parser, scheduler). GPU tests run locally before merging, and the PR says so.
- `.clang-format` checked in, and formatting enforced in CI.

### Cadence

| When | What | Length |
| --- | --- | --- |
| Sunday evening | Plan the week against the milestone table; move tasks between A and B if one side is blocked | 30 min |
| Wednesday | Async check-in in chat: done, next, blocked | 5 min |
| Every milestone | Demo to each other: run the milestone's exit criteria live, update the milestone table | 30 min |

Track tasks in GitHub Issues with one milestone per M1 to M5, so the repo itself shows the project history.

## Part 2: polish, UI/UX, and enhancements

Part 2 starts only after M5 ships. Do it in priority order: first a demo-able server and dashboard (so people can *see* the engine), then the performance features that make the best interview stories (chunked prefill, quantization, prefix caching), then the hard kernels. Each item is independent, so you can stop at any point with a finished project.

### Priority list

| Priority | Item | Track | Effort (2 people) | Resume / interview value |
| --- | --- | --- | --- | --- |
| 1 | OpenAI-compatible HTTP server with streaming | Serving | 1 week | Medium: makes it usable and demo-able |
| 2 | Web dashboard (playground + live metrics + batch timeline) | UI/UX | 1 to 2 weeks | High for demos and non-specialist recruiters |
| 3 | Chunked prefill + mixed prefill/decode batches | Runtime | 1 week | High: fixes a measured MVP problem |
| 4 | INT8 weight-only quantization (W8A16) | Kernels | 1 to 2 weeks | High: halves decode bytes |
| 5 | Prefix caching + copy-on-write blocks | Runtime | 1 week | High: classic vLLM/SGLang feature |
| 6 | Realistic load generator + latency-vs-load curves | Benchmarking | 3 days | Medium: serving-style results |
| 7 | FlashAttention-style prefill on Tensor Cores | Kernels | 2 to 3 weeks | Very high, but hard |
| 8 | FP8 GEMM with `mma.sync` + `cp.async` double buffering | Kernels | 2 to 3 weeks | Very high for NVIDIA, hardest item |
| 9 | More models (Llama-3.2-1B, Qwen2.5) and GGUF | Runtime | 1 week | Medium: generality |
| 10 | Split-K / FlashDecoding for long contexts | Kernels | 1 week | Medium |
| 11 | Async scheduling and multi-stream overlap | Runtime | 1 week | Medium-high |
| 12 | Swap-based preemption to host memory | Runtime | 3 days | Medium |
| 13 | Speculative decoding (prompt-lookup / n-gram) | Runtime | 1 week | High, very current topic |
| 14 | Engineering polish: Docker, pybind11, docs, perf CI, blog post | Polish | ongoing | High for credibility |

### Track 1: Serving

**P1. OpenAI-compatible HTTP server.**

- **What:** `POST /v1/completions` and `POST /v1/chat/completions` with `stream: true` using Server-Sent Events, plus `GET /health` and `GET /metrics`.
- **How:** use `cpp-httplib` (single header, easy) or Drogon (async, more work). The engine loop runs on its own thread. HTTP handler threads push requests into a mutex-protected queue and wait on a per-request token channel (a small thread-safe queue of token ids). The engine thread drains the queue at the top of each iteration and pushes each sampled token to its request's channel. Detokenize incrementally (SentencePiece pieces can split UTF-8 characters, so only emit text once a complete character is available).
- **Handle:** client disconnect cancels the request and frees its blocks; `max_tokens`, `temperature`, `top_p`, `stop` strings; request ids in responses.
- **Done when:** the official `openai` Python client and `curl` can both stream from it, and 64 concurrent streaming clients work without errors.

**P6. Load generator.**

- Poisson arrivals at a given request rate, prompt and output lengths sampled from a realistic distribution (for example, a public chat dataset's length histogram).
- Sweep the request rate and plot p50/p99 TTFT and TPOT against throughput. The "knee" of this curve is the most standard serving result there is.
- Run the same sweep against vLLM on a rented GPU for a head-to-head (report honestly, even if vLLM wins).

### Track 2: UI/UX (the web dashboard)

Build it with Next.js (or Vite + React), TypeScript, Tailwind, and Recharts, reusing what you already know from CrystalOS. It talks to the engine only through the HTTP API and `/metrics`, so the C++ side stays clean.

**Pages:**

1. **Playground.** Chat UI with streaming tokens, sampling controls (temperature, top-p, top-k, max tokens), and a per-message footer showing TTFT, tokens/s, and tokens generated. A "compare" mode sends the same prompt with two settings side by side.
2. **Live engine view.** Updates every 500 ms from `/metrics`: tokens/s, running and waiting request counts, current batch size, KV cache usage (% of blocks used), GPU memory, and a scrolling chart of throughput over time.
3. **Batch timeline.** A horizontal lane per request showing when it waited, prefilled, decoded, was preempted, and finished. This one picture explains continuous batching better than any paragraph, and it is great in a demo video.
4. **KV block map.** A grid of squares, one per physical block, colored by owning request (and shared blocks highlighted once prefix caching exists). It makes paging visible.
5. **Benchmarks.** Static page that renders the committed benchmark CSVs: kernel table (E1), engine comparison (E2), batching curve (E3), graphs on/off (E4), attention scaling (E5), latency-vs-load (P6).

**UX details that make it feel finished:** dark and light themes; tabular (monospace) numbers so values do not jitter; skeleton loaders; clear empty and error states ("engine offline, start it with ..."); keyboard shortcut to send (Ctrl+Enter); responsive down to phone width; and one shareable URL per benchmark chart.

**Public hosting:** the GPU is on your laptop, so host the Benchmarks page as a static site (GitHub Pages or Vercel) with the committed data, and record a 2-minute demo video of the live pages for the README.

### Track 3: Runtime features

**P3. Chunked prefill.** Split prompts into chunks of at most 256 to 512 tokens, and build every step as a mix: all running decode tokens plus one prefill chunk, within a per-step token budget. Attention must now handle a chunk whose queries start at position `p` with `p` tokens already cached. *Done when:* the E-series shows decode TPOT spikes during long prompt arrivals drop sharply (show both plots).

**P5. Prefix caching and copy-on-write.** Hash each full block by (hash of previous block, its 16 token ids). Keep a hash-to-block map; when a new request's prefix blocks match, reuse them and bump ref counts instead of recomputing. Blocks with ref count 0 go on an LRU list and are only reused when memory is needed. Copy-on-write: when a sequence must write into a shared, partially filled block, copy it first. This also enables `n > 1` parallel sampling. *Done when:* a benchmark with a shared 1,000-token system prompt shows TTFT dropping by the expected amount on cache hits.

**P9. More models.** Make the model config-driven. Llama-3.2-1B adds `llama3` RoPE scaling, a 128,256-token vocabulary, and tied embeddings. Qwen2.5 adds biases on Q/K/V. Each new model gets its own golden tests.

**P11. Async scheduling.** The CPU currently waits for step N before preparing step N+1. Instead, prepare step N+1's inputs while the GPU runs step N (double-buffered pinned input structs, events to order them). Show the removed CPU gap in `nsys`.

**P12. Swap preemption.** Instead of recomputing, copy a preempted sequence's blocks to pinned host memory with `cudaMemcpyAsync` on a separate stream, then back when resumed. Compare swap vs recompute latency by sequence length; the crossover point is a nice result.

**P13. Speculative decoding.** Start with prompt-lookup decoding: propose the next `k` tokens by matching the last n-gram against the prompt, verify all `k` in one forward pass, accept the longest correct prefix. No draft model needed, and it is a big win on summarization and code-editing prompts. Report the acceptance rate and speedup.

### Track 4: Hard kernels

**P4. INT8 weight-only quantization (W8A16).** Quantize each weight row to INT8 with one FP16/BF16 scale per output channel (or per group of 128). The GEMV kernel loads INT8 weights, converts to FP32 in registers, multiplies by BF16 activations, and applies the scale at the end. Decode reads half the bytes, so batch-1 TPOT should approach 2x faster. Measure perplexity on a small text set (for example, a few hundred WikiText passages) before and after to prove quality holds.

**P7. FlashAttention-style prefill.** Tile Q into blocks of 64 rows; loop over K/V tiles; compute `Q K^T` and `P V` with Tensor Core instructions (`wmma` first, then `mma.sync.aligned.m16n8k16` for BF16); online softmax across tiles; causal masking only on diagonal tiles. Compare against your MVP prefill kernel and, on a rented GPU, against FlashAttention or FlashInfer.

**P8. FP8 GEMM.** Your `sm_120` GPU supports FP8 (e4m3/e5m2) Tensor Core `mma.sync`. Quantize weights to e4m3 with per-channel scales; stage tiles into shared memory with `cp.async` and a two- or three-stage pipeline so loads for tile `k+1` overlap math on tile `k`; use `ldmatrix` to feed fragments; apply scales in the epilogue. Benchmark against cuBLASLt FP8 and report % of cuBLASLt. Start by reading and reproducing a known-good tiled GEMM tutorial before adding FP8.

**P10. Split-K decoding.** At batch 1 with long contexts, one block per KV head leaves most of the 26 SMs idle. Split each sequence's context across several blocks, each producing a partial (max, sum, output), then merge them in a second small kernel with the online-softmax rescale.

### Track 5: Engineering polish

- **Docker:** CUDA 12.8 devel base image, one command to build and run the benchmark.
- **Python bindings (pybind11):** `import engine; engine.generate(...)`, so people can try it in a notebook.
- **Perf regression check:** a script that reruns E1 to E3 and fails if anything is more than 5% slower than the committed baseline; run it before every merge to `main`.
- **Static analysis:** `clang-tidy`, sanitizers, and a fuzz test for the safetensors parser (malformed headers must not crash).
- **Documentation:** a design doc explaining every major decision and its alternative ("why block size 16", "why recompute preemption first").
- **Blog post:** "Building an LLM inference engine from scratch on a laptop GPU", with profiling screenshots and the plots. This is often what gets shared and gets you noticed.

## Resume bullets, README, and interview prep

The bullets below are templates in Google XYZ form. Every bracket must be filled from a committed benchmark CSV, and each of you should only claim the parts you built (use "co-designed" for shared work).

### Resume entry (Jake's template)

**Project line:** `[Engine name] | C++20, CUDA, cuBLAS, Nsight` with the GitHub link, dated `Sep 2026 to Nov 2026`.

**Bullet templates (pick 3 or 4):**

1. Achieved [X] tokens/s single-stream decode on TinyLlama-1.1B ([Y]% of measured DRAM bandwidth, [Z]x faster than HuggingFace eager) by building a C++20/CUDA inference engine with fused, 128-bit vectorized RMSNorm, RoPE, and SwiGLU kernels.
2. Scaled serving throughput [N]x, from [A] to [B] tokens/s at 16 concurrent requests, by implementing a paged KV-cache block manager and an iteration-level continuous batching scheduler with recompute-based preemption.
3. Reduced attention memory traffic [N]x at 2K context by writing a GQA-aware paged decode attention kernel with online softmax that reads non-contiguous KV blocks through per-sequence block tables.
4. Cut per-token latency [P]% at batch 1 by capturing decode steps as CUDA Graphs across batch-size buckets, removing ~225 kernel launches of CPU overhead per token (verified in Nsight Systems).
5. Verified [99.x]% token agreement with HuggingFace across 20 prompts by building a layer-by-layer golden-test harness with GoogleTest and compute-sanitizer.

After Part 2, strong replacements are: "halved decode memory traffic with INT8 weight-only quantization at [Q]% perplexity change", and "cut decode latency spikes [N]x with chunked prefill".

### README structure

1. One-line description and a GIF of the dashboard or CLI streaming.
2. **Results table first:** engine vs HF vs llama.cpp (tokens/s, TPOT, TTFT), laptop and cloud GPU columns.
3. The batching plot and the kernel % of peak table.
4. Architecture overview (a short numbered description and one exported PNG diagram).
5. Build and reproduce: prerequisites, three commands, expected output.
6. Design decisions with trade-offs (block size, recompute vs swap, graph buckets).
7. Limitations, stated honestly (single GPU, one model family, simple prefill kernel in the MVP).
8. Roadmap (Part 2 items) and credits (vLLM, FlashAttention, PMPP).

### Questions you must be able to answer cold

**Fundamentals**

- Why is batch-1 decode memory bound? Compute the arithmetic intensity of a GEMV.
- What is the theoretical max tokens/s for this model on your GPU? (Bandwidth divided by 2.07 GB.)
- What does coalescing mean, and how do 128-bit loads help?
- Why does batching increase throughput almost for free until a certain batch size, and what limits it?
- What is a warp shuffle, and why is it faster than a shared-memory reduction?

**Design choices**

- Why a block size of 16? What happens at 1 or 256?
- Why paging instead of contiguous allocation? Quantify the fragmentation you avoided (E6).
- Recompute vs swap preemption: when is each better?
- Why pad batches to graph buckets instead of capturing every size?
- Why is `lm_head` computed only for the last token?
- How does GQA change the attention kernel and the cache size?

**Your numbers**

- Walk through an `ncu` report for your RMSNorm kernel. What was the bottleneck before and after?
- Where does the remaining gap to peak bandwidth come from?
- What would you change first to go 2x faster? (Quantization, and why.)
- How do you know your output is correct? (Explain the teacher-forcing test and tolerances.)

**Stretch (NVIDIA-style)**

- How does FlashAttention avoid materializing the N x N matrix, and why is prefill different from decode?
- What do `cp.async` and multi-stage pipelines hide?
- What changes on Hopper and datacenter Blackwell (TMA, `wgmma`, `tcgen05`) that your `sm_120` code does not use?

## Risks, mitigations, and learning resources

The biggest risk is not a hard kernel. It is losing two weeks to setup and numerical debugging while learning C++ and CUDA at the same time, so the schedule front-loads both and keeps a buffer week in mid-October.

### Risks

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| Output never matches HF (numerics bugs) | High | Blocks everything | Layer-by-layer golden tests from week 2; debugging method above; ask for help after 2 days stuck on one layer |
| Midterms and course load eat October | High | Schedule slip | Week 3 buffer; cut list in the schedule section; milestones are pass/fail, so progress is visible early |
| WSL2 / driver / profiler setup problems | Medium | 1 to 3 days lost | Do setup in week 0 together; fallback to dual-boot Ubuntu; verify `ncu` works on a toy kernel before week 2 |
| Laptop thermal throttling makes numbers noisy | Medium | Unreliable results | Log clocks and power; median of 5 runs; confirm on a rented GPU |
| 8 GB VRAM is tight for later models | Low (MVP), Medium (Part 2) | Limits Part 2 | TinyLlama for MVP; 3B models only on rented GPUs |
| Uneven workload between the two of you | Medium | Friction, one person blocked | Interfaces agreed in week 0, stubs, weekly re-plan, cross-over tasks |
| Scope creep (starting Part 2 items early) | High | MVP misses Nov 15 | Anything not in the MVP definition of done waits; keep a "later" issue label |
| Over-relying on AI-generated code | Medium | Cannot defend it in interviews | Write kernels yourselves; use AI to explain and review, and be able to walk through every line |

### Learning path (both of you, weeks 0 to 2)

| Order | Resource | What to take from it |
| --- | --- | --- |
| 1 | *Programming Massively Parallel Processors* (Hwu, Kirk, El Hajj), 4th ed., ch. 1 to 6 and 10 | Execution model, memory, tiling, reductions. The standard textbook |
| 2 | NVIDIA *CUDA C++ Programming Guide* and *Best Practices Guide* | Reference for warp shuffles, streams, graphs, memory coalescing |
| 3 | [llama2.c](https://github.com/karpathy/llama2.c) (Karpathy) | A complete Llama forward pass in ~700 lines of C. Read `run.c` end to end before writing your model runner |
| 4 | [How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM) (Simon Boehm) | The best step-by-step guide to tiling, vectorization, and profiling with `ncu` |
| 5 | GPU MODE lectures (YouTube) and Discord | Kernel optimization talks and a community to ask questions |
| 6 | [PagedAttention / vLLM paper](https://arxiv.org/abs/2309.06180) | Block tables, fragmentation, copy-on-write. Read before week 4 |
| 7 | [FlashAttention](https://arxiv.org/abs/2205.14135) and [FlashAttention-2](https://arxiv.org/abs/2307.08691) | Online softmax and tiling. Read before week 6 |
| 8 | Orca (OSDI 2022) | Origin of iteration-level (continuous) batching. Read before week 5 |
| 9 | [Sarathi-Serve](https://arxiv.org/abs/2403.02310) | Chunked prefill, for Part 2 |
| 10 | vLLM and nano-vllm source code | How a real engine structures the scheduler and model runner; compare after you write yours, not before |

### Sources

As of Sep 23, 2026, facts in this plan were checked against:

- [NVIDIA CUDA GPUs compute capability list](https://developer.nvidia.com/cuda-gpus) (RTX 5060 is compute capability 12.0)
- [Notebookcheck: GeForce RTX 5060 laptop GPU specifications](https://www.notebookcheck.net/GeForce-RTX-5060-laptops-GPU-unveiled-with-8-GB-GDDR7-VRAM-and-100-Watts-max-power.999887.0.html) (8 GB GDDR7, 128-bit, 45 to 100 W)
- [TinyLlama-1.1B-Chat-v1.0 config.json](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/raw/main/config.json)
- [Nsight Compute 2022.4 release notes](https://docs.nvidia.com/nsight-compute/ReleaseNotes/topics/updates-2022-4.html) (WSL2 profiling support)
- [CUDA 12.8 release notes](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/) (first toolkit with consumer Blackwell support)

The exact memory bandwidth of your laptop's RTX 5060 was not confirmed from an official source and depends on the laptop's memory clock; measure it with `bandwidth_test.cu` in week 1.
