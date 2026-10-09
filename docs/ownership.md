# Ownership

This project is built by Joe Zhuo with Claude (Anthropic's coding assistant) as tutor, reviewer and writer of the supporting code. The rule is that anything that would appear in a resume bullet or an interview question is written by Joe. Every commit lands under Joe's git identity, so `git log` does not show who wrote what; this page and the tags in `changelog.md` do.

## The three parts

| Part | Who writes it | What Joe does | Covers |
| --- | --- | --- | --- |
| A | Claude | Reviews and tests it | Build system, CI, formatting and editor config, the CLI and benchmark harness plumbing, CSV metrics, plotting, the Python reference and golden scripts, config and JSON parsing, the weight loader beyond the safetensors header parse, the tokenizer wrapper and chat template, test boilerplate, the checksum tool, docs, and the buffer, tensor and error-check infrastructure |
| B | Claude drafts | Reads and explains every line before it merges | cuBLAS GEMM wrappers, the forward pass wiring in the model runner, the sampler, and the naive oracle kernels (embedding, RMSNorm, RoPE, SwiGLU, residual add, attention over a contiguous KV cache) that the optimized kernels are tested against |
| C | Joe, by hand | Writes it; Claude only writes the stub, the spec and the tests, and reviews the result | The safetensors header parser, the fused and vectorized kernels, paged decode and prefill attention, the KV cache layout and `write_kv`, the KV block manager, the scheduler and preemption, and CUDA Graph capture |

For Part C, Claude never writes any of the implementation. It writes a stub that compiles and fails loudly, a `TODO(JOE)` spec above it (interface, shapes, edge cases, what the result must match, which tests prove it) and tests labelled `todo_joe`; when Joe asks for help, it gives explanations, hints and reviews rather than code. The README describes the `todo_joe` workflow.

## Components so far (M1)

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
| Measurements and notes | `docs/exercises/` (bandwidth, DRAM throughput and reduction results), `docs/hardware.md` | none | Joe |
| Learning log | `docs/exercises/learning_log.md` | none | Claude lists the concepts each change relies on as blank entries; every definition is Joe's |
| README, changelog, this page | `README.md`, `changelog.md`, `docs/ownership.md` | A | Claude |

## Planned components

| Component | Milestone | Part |
| --- | --- | --- |
| Tokenizer wrapper and chat template | M2 | A |
| Python reference dumps and logit comparison (`dump_reference.py`, `compare_logits.py`), golden-test harness | M2 | A |
| Naive oracle kernels, forward pass wiring, sampler | M2 to M3 | B |
| Paged decode and prefill attention | M3 | C |
| KV cache layout and `write_kv`, KV block manager | M3 | C |
| Scheduler and preemption | M3 | C |
| Fused `add_rmsnorm`, vectorized RoPE, fused SwiGLU | M4 | C |
| CUDA Graph capture and batch buckets | M4 | C |
| Benchmark harness, CSV metrics, plots | M3 to M5 | A |
