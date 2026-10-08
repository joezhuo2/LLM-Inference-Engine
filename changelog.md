# Changelog

Changes are grouped by branch, newest first, in the order the branches merge into `main`. Each entry is tagged with its part of the delegation: Part A and Part B entries are written by Claude and reviewed by Joe, and Part C entries are written by Joe.

## loader/weight-upload

- Fix a debug-only race in `DeviceBuffer`: the NaN fill is a `cudaMemset` that runs asynchronously on the legacy default stream, which a non-blocking stream does not wait for, so a copy issued right after allocation could be overwritten by the fill. The constructor now synchronizes after the fill (debug builds only). (Part A)
- Add `plan_weight_layout(config, header)`, pure C++ that checks every tensor the config implies is present, BF16 and correctly shaped (and that nothing unexpected is in the file), then places all weights in one allocation with 256-byte aligned regions: Q, K and V fused into `wqkv` `[q + 2 kv, hidden]`, gate and up fused into `wgu` `[2 ff, hidden]`, and a copy list sorted by file offset so the file is read front to back. Adds `tests/support/fake_checkpoint.h` (tiny and TinyLlama-sized configs and headers) so the planner is tested without the real file or the parser. (Part A)
- Add `ModelWeights` / `LayerWeights` (BF16 `Tensor` views named after their role: `ln1`, `wqkv`, `wo`, `ln2`, `wgu`, `wdown`, plus `embed`, `final_norm`, `lm_head`) and `bind_weights(layout, base)`, which points every view into the single weight allocation; this is the struct the forward pass will take. (Part A)

## build/ci-timeouts

- Give every CI job a `timeout-minutes` limit (15 minutes, 30 for the CUDA compile job) so a hung step fails quickly instead of running for GitHub's 6-hour default; jobs normally finish in one or two minutes. (Part A)
- Run every `apt-get` in CI with `APT_OPTS` (3 retries, 30-second HTTP and HTTPS timeouts), so a slow or unreachable package mirror is retried and then fails instead of hanging; this is what stalled PRs #9 and #10 for over an hour. (Part A)
- The clang-format job skips `apt-get` entirely when the runner already has clang-format 18 (the version used locally and shipped by Ubuntu 24.04, since other major versions can format differently), and prints the version it used. (Part A)

## loader/safetensors-stub

- Add `engine::MappedFile`, a move-only RAII read-only `mmap` of a whole file (`open`, `fstat`, `mmap(PROT_READ, MAP_PRIVATE)`, then the descriptor is closed because the mapping outlives it; `munmap` on destruction) exposing `std::span<const std::byte>`. Empty files give an empty span, and failures throw `std::system_error` carrying `errno` and the path. Adds `tests/support/temp_file.h` for tests that need real files. (Part A)
- Add the Part C stub `parse_safetensors_header(std::span<const std::byte>)` returning `SafetensorsHeader` (absolute `data_offset` plus a name-sorted map of `TensorEntry` with dtype, shape, absolute offset and byte length); it throws `std::logic_error("unimplemented: parse_safetensors_header")` under a TODO(JOE) spec whose accept and reject rules were checked against Python safetensors 0.8.0. Adds `tests/support/safetensors_bytes.h` and 28 tests in the new `todo_unit_tests` target (label `todo_joe`), all failing until the parser exists. (Part C stub)
- Fix the macOS CI build: `EXPECT_THROW(MappedFile(std::filesystem::temp_directory_path()), ...)` is the most vexing parse under Apple Clang (read as a declaration), so the test now uses brace initialization. (Part A)

## model/config

- Add `engine::ModelConfig` with `parse_config(json)` and `load_config(path)` reading the HF `config.json` through nlohmann/json, plus derived widths (`kv_dim` 256, `qkv_dim` 2560, `group_size` 8 for TinyLlama). It rejects anything the engine does not implement (non-llama models, RoPE scaling, non-SiLU activations, attention or MLP biases, `pretraining_tp` above 1) and inconsistent head counts, with errors that name the offending field. (Part A)
- Add `tests/support/paths.h` with `engine::test::model_dir()` (compile-time default `models/tinyllama`, overridable with the `ENGINE_MODEL_DIR` environment variable) and a test that loads the real TinyLlama `config.json`, skipping when the model is not present (as in CI). (Part A)

## kernel/cublas-gemm

- Add `tests/support/bf16.h`, host-side float to BF16 conversion with round-to-nearest-even, for building CPU reference results; unit tests cover ties and special values, and a GPU-test sweep checks one million random floats against CUDA's `__float2bfloat16`. (Part A)
- Add `engine::Blas`, a non-copyable cuBLAS handle bound to one stream with a caller-owned workspace set through `cublasSetWorkspace` (so cuBLAS never allocates during CUDA Graph capture later); taking the workspace pointer keeps `engine_kernels` independent of `engine_runtime`. (Part B)
- Add `engine::gemm(blas, y, x, w)` computing `y[B, out] = x[B, in] W^T` for row-major BF16 tensors with FP32 accumulation through one `cublasGemmEx` call (the row-major trick: ask cuBLAS for the column-major `Y^T = W X^T` with `CUBLAS_OP_T, CUBLAS_OP_N`); rejects non-2-D, non-BF16 or mismatched shapes. Tests: an exact 3x5 hand example and a CPU-reference sweep over B=1, odd sizes, fused QKV, down-proj and lm_head shapes. (Part B)

## core/tensor-and-buffers

- Add the `DType` enum, `dtype_size`, and a non-owning `Tensor` view (data pointer, dtype, up to 4 dims, `numel`, `bytes`) as the common currency between the loader, kernels and runtime. (Part A)
- Set `ReflowComments: false` in `.clang-format` so long comments stay single-line paragraphs instead of being wrapped at 100 columns; no existing file changes formatting. (Part A)
- Add `engine::Stream`, an alias for `cudaStream_t` declared through a forward-declared `CUstream_st`, so kernel launchers in `include/engine/` can take a stream without including CUDA headers; a static_assert in `gpu_tests` proves the types are identical. (Part A)
- Add `CUBLAS_CHECK` (prints the cuBLAS status name and message, then aborts) and `KERNEL_CHECK(stream)` to `src/kernels/cuda_check.h`; in debug builds `KERNEL_CHECK` also synchronizes the stream so asynchronous faults point at the right kernel, except while the stream is being captured into a CUDA Graph. Death tests cover both abort paths. (Part A)
- Add the `engine_runtime` library (CUDA host code, built only when CUDA is found) and a move-only RAII `DeviceBuffer` over `cudaMalloc`/`cudaFree`; debug builds fill new buffers with 0xFF bytes, which are NaN in BF16 and FP32, so reads of unwritten memory are obvious. (Part A)
- Add a move-only RAII `PinnedBuffer` over `cudaMallocHost`/`cudaFreeHost` for the weight loader's staging buffers, since `cudaMemcpyAsync` from pageable (for example mmap'd) memory goes through a hidden driver copy; a test checks the memory reports as `cudaMemoryTypeHost`. (Part A)
- Remove explanatory comments from this branch's code (only namespace closers and Part C TODO(JOE) spec blocks remain) and list the concepts both M1 branches rely on as blank entries in a new "Part 1 - Engine (M1 Foundations)" section of `docs/exercises/learning_log.md` for Joe to define. `build/m1-scaffolding` (which now includes `main`) is merged in first, so this change only appends to the log. (Part A)

## build/m1-scaffolding

- Add CLI11 (found on the system first, fetched otherwise) and turn `engine` into a subcommand CLI with a `version` subcommand; running it with no arguments still prints the version. (Part A)
- Rename the `json` FetchContent dependency to `nlohmann_json` so CMake stops warning about a package name mismatch on every configure. (Part A)
- CI runs `ctest -LE todo_joe` so tests for unimplemented Part C components do not fail the build, installs CLI11 from apt and Homebrew, and adds a job that byte-compiles every tracked Python file. (Part A)
- `.clangd` reads `build/debug/compile_commands.json` directly, so the root symlink is no longer needed. (Part A)
- Add `scripts/requirements.txt` pinned to the `~/.venvs/llmie` versions, with the cu128 index for an sm_120 torch build. (Part A)
