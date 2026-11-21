# Ownership

This project is built by Joe Zhuo with Claude (Anthropic's coding assistant) as tutor, reviewer and writer of the supporting code. The rule is that anything that would appear in a resume bullet or an interview question is written by Joe. Every commit lands under Joe's git identity, so `git log` does not show who wrote what; this page and the tags in `changelog.md` do.

## The three parts

| Part | Who writes it | What Joe does | Covers |
| --- | --- | --- | --- |
| A | Claude | Reviews and tests it | Build system, CI, formatting and editor config, the CLI and benchmark harness plumbing, CSV metrics, plotting, the Python reference and golden scripts, config and JSON parsing, the weight loader beyond the safetensors header parse, the tokenizer wrapper and chat template, test boilerplate, the checksum tool, docs, and the buffer, tensor and error-check infrastructure |
| B | Claude drafts | Reads and explains every line before it merges | cuBLAS GEMM wrappers, the forward pass wiring in the model runner, the sampler, and the naive oracle kernels (embedding, RMSNorm, RoPE, SwiGLU, residual add, attention over a contiguous KV cache) that the optimized kernels are tested against |
| C | Joe, by hand | Writes it; Claude only writes the stub, the spec and the tests, and reviews the result | The safetensors header parser, the fused and vectorized kernels, paged decode and prefill attention, the KV cache layout and `write_kv`, the KV block manager, the scheduler and preemption, and CUDA Graph capture |

For Part C, Claude never writes any of the implementation. It writes a stub that compiles and fails loudly, a `TODO(JOE)` spec above it (interface, shapes, edge cases, what the result must match, which tests prove it) and tests labelled `todo_joe`; when Joe asks for help, it gives explanations, hints and reviews rather than code. Tests labelled `todo_joe` fail until Joe's implementation lands, which is why the README's test command is `ctest --preset release -LE todo_joe`; when they pass, they move from `tests/todo/` into `tests/unit/` or `tests/gpu/`.

## Components from M1

| Component | Files | Part | Written by |
| --- | --- | --- | --- |
| Safetensors header parser | `src/loader/safetensors.cpp` (`parse_safetensors_header`) | C | Joe; Claude wrote the stub, its spec and the 28 tests beforehand, and a follow-up cleanup (reading `data_offsets` as unsigned, validating in one pass) that Joe reviewed |
| cuBLAS handle and BF16 GEMM | `include/engine/kernels/gemm.h`, `src/kernels/gemm.cpp`, `tests/gpu/test_gemm.cpp` | B | Claude drafted, Joe reviewed line by line |
| `DType`, `Tensor`, `Stream` | `include/engine/core/`, `src/core/tensor.cpp` | A | Claude |
| Error checks | `src/kernels/cuda_check.h` (`CUBLAS_CHECK`, `KERNEL_CHECK`) | A | Claude, extending the bootstrap's `CUDA_CHECK` |
| `DeviceBuffer`, `PinnedBuffer` | `include/engine/runtime/`, `src/runtime/device_buffer.cpp`, `src/runtime/pinned_buffer.cpp` | A | Claude |
| Model config | `include/engine/model/config.h`, `src/model/config.cpp` | A | Claude |
| `MappedFile` | `include/engine/loader/mapped_file.h`, `src/loader/mapped_file.cpp` | A | Claude |
| Weight layout, `ModelWeights`, upload and `load_weights` | `src/loader/weight_layout.cpp`, `src/model/weights.cpp`, `src/runtime/weights.cpp` | A | Claude |
| Checksums and `engine checksum` | `src/loader/checksum.cpp`, `src/runtime/checksum.cpp`, `src/app/`, `scripts/checksum.py` | A | Claude |
| Tests and test helpers | `tests/unit/`, `tests/gpu/`, `tests/support/` (except `test_version.cpp` and `test_smoke.cpp`) | A | Claude |
| Build, CI and tooling after the bootstrap | CLI11, the library split, test targets and labels, CI timeouts and retries, `scripts/format.sh`, `scripts/requirements.txt` | A | Claude |
| Bootstrap scaffolding | The first `CMakeLists.txt` files, `CMakePresets.json`, `.clang-format`, `.clangd`, `.github/workflows/ci.yml`, the version library, the smoke kernel and the first two tests | A | Joe created and committed them, following the setup guide written with Claude during planning |
| Learning exercises | `src/practice/` (move-only buffer, safetensors header reader, three reductions, the bandwidth test, the PyTorch TinyLlama forward pass with a KV cache) | none | Joe |
| Measurements and notes | `docs/exercises/` (bandwidth, DRAM throughput and reduction results), `hardware.md` | none | Joe |
| Learning log | `docs/exercises/learning_log.md` | none | Claude lists the concepts each change relies on as blank entries; every definition is Joe's |
| README, changelog, this page | `README.md`, `changelog.md`, `ownership.md` | A | Claude |

## Components after M1

Everything below has landed unless its row says stub.

| Component | Part | Written by |
| --- | --- | --- |
| Tokenizer wrapper and chat template | A | Claude |
| Python reference dumps and logit comparison (`dump_reference.py`, `dump_logits.py`), golden-test harness | A | Claude |
| Naive oracle kernels, forward pass wiring, sampler | B | Claude drafted, Joe reviewed |
| Paged decode attention v1 and v2 (`paged_attention_decode`, `paged_attention_decode_v2`) | C | Joe, from Claude's stubs, specs and tests; Claude fixed the spec violations of the first draft and drafted two v2 optimizations (8 lanes per key, the value sum spread over all threads) |
| Split decode attention (`paged_attention_decode_split`) | C | drafted by Claude at Joe's request, for him to study |
| KV cache layout and `write_kv`, KV block manager | C | Joe, with fixes by Claude where the first drafts broke the spec |
| Scheduler and preemption, chunked prefill (`Scheduler::schedule_chunked`) | C | Joe; the replay fix after preemption was written by Claude at Joe's request |
| Fused `add_rmsnorm` | C | Joe |
| `fast::rope`, `fast::swiglu` | C | Joe's first versions; the parallel RoPE and the vectorized SwiGLU are fixes by Claude |
| CUDA Graph capture (`DecodeGraphs`) and batch buckets (`decode_graph_buckets`) | C | Joe |
| Vectorized embedding lookup (`fast::embedding`) | C | written by Claude at Joe's request |
| FlashAttention-style prefill (`flash_attention_prefill`) | C | written by Claude at Joe's request: the tiled kernel that ships replaced the first kernel (#185), and so were the bounds, 64-bit offset, grid and 32-bit division changes |
| Benchmark harness, CSV metrics, plots, baselines (HuggingFace, llama.cpp, vLLM) | A | Claude |
| HTTP server plumbing, chat API, CORS, SSE, metrics windows (`src/server/`, `engine serve`) | A, with the wiring into the runner (`src/server/gpu_backend.cpp`, `serving_loop.cpp`) B | Claude |
| `Scheduler::cancel` | C | written by Claude at Joe's request |
| Speculative decoding by prompt lookup: `propose_prompt_lookup`, `count_accepted`, `BlockManager::truncate` (`src/sched/speculative.cpp`) | C | Joe, from Claude's stub, spec and tests |
| Speculative decoding in the scheduler (`Scheduler::reserve_drafts`, the draft rows of `schedule`, `schedule_chunked` and `update`), the draft model | B and C | written by Claude at Joe's request |
| Tensor parallelism: the layout of one rank (`plan_shard_layout`) | C | Joe, finished by Claude at Joe's request |
| Tensor parallelism: `ShardedRunner`, `all_reduce_add`, `engine generate --tensor-parallel` | A and B | Claude |
| The site (`web/`) | A | Claude; Joe reviews and edits |
| Prefix caching on `BlockManager` (`match_prefix`, `allocate_tokens`, `publish`) | C | Joe; stub, spec and tests by Claude |
| FP8 KV cache (`write_kv_fp8`, `paged_attention_decode_fp8`), `gemm_fp8`, grouped decode attention (`paged_attention_decode_grouped`) | C | stubs, specs and tests by Claude; not written yet |
