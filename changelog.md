# Changelog

Changes are grouped by branch, newest first, in the order the branches merge into `main`. Each entry is tagged with its part of the delegation: Part A and Part B entries are written by Claude and reviewed by Joe, and Part C entries are written by Joe.

## tokenizer/sentencepiece

- Add the SentencePiece C++ library as `engine_sentencepiece`: an installed copy found through pkg-config (Ubuntu's `libsentencepiece-dev` and Homebrew ship a `sentencepiece.pc`, not a CMake package) is used first, and otherwise v0.2.1 is fetched (pinned by SHA-256, static, no tcmalloc, `EXCLUDE_FROM_ALL` so its command-line tools are not built). v0.2.1 is the newest release that bundles its own trimmed absl; v0.2.2 git-clones and builds all of abseil at configure time. CI installs `pkg-config` and `libsentencepiece-dev` (0.2.0) on Ubuntu and in the CUDA container; macOS fetches 0.2.1, because Homebrew's 0.2.2 headers need a separate abseil install. The warning flags are now added after the dependencies, so `-Wall -Wextra -Wpedantic` applies to the engine's own targets only and third-party code cannot fail the zero-warning build. (Part A)

## build/repo-housekeeping

- Stop tracking the `models` symlink, which pointed at `/home/crystalflux/models` and so dangled on every other clone; it was already in `.gitignore`. Each machine now creates its own link (`ln -s ~/models models`), as the README's build section says, or sets `ENGINE_MODEL_DIR`. (Part A)
- Stop tracking `src/practice/bandwith_test`, a 1 MB compiled executable of the bandwidth exercise, and ignore it; the source `bandwith_test.cu` stays tracked and rebuilds it with `nvcc`. (Part A)

## docs/m1-readme

- Add `docs/ownership.md`: the Part A, B and C split (who writes, who reviews, what each covers), a component-by-component table of who wrote every piece of M1 including Joe's parser, exercises and notes, and the parts planned for later milestones. Every commit lands under Joe's git identity, so this page and the changelog tags are the record of authorship. (Part A)
- Write the README, which was empty: what the engine is, the milestone table with M1's status, what works today (including the measured 323.6 GB/s bandwidth), the pinned toolchain versions, getting the model (and that the tracked `models` symlink points at `/home/crystalflux/models`, so other machines repoint it or set `ENGINE_MODEL_DIR`), the three presets, test labels and the compute-sanitizer command, what CI checks, the `todo_joe` workflow, `engine checksum` with `scripts/checksum.py` and the golden tests, the repository layout and the commit conventions. (Part A)
- Rewrite the README as a project overview instead of a status and workflow page: the goals (correctness, batch-1 and batch-N speed, explained and reproducible numbers), an architecture diagram and the decode step, the model and why batch-1 decode is memory bound (about 2.07 GB read per token, a floor of about 6.4 ms per token at 323.6 GB/s), the design decisions behind each component, what comes after the MVP, the hardware, a short build section, the layout and ownership. Milestone status, the test labels, the `todo_joe` workflow, the checksum walkthrough and commit conventions are gone. (Part A)

## tools/checksum

- Move the real-checkpoint `load_weights` test from `tests/todo/gpu/` into `gpu_tests` now that the parser exists, and delete the empty `todo_gpu_tests` target; `tests/todo/` is now empty. (Part A)
- Add `dtype_name(DType)`, returning the safetensors spelling (`"BF16"` and so on), for the checksum output and its JSON. (Part A)
- Add `checksum_bf16` (FP64 sum and sum of absolute values over a BF16 byte range, decoding little-endian byte pairs so any alignment works), `checksum_file` (every tensor of a parsed safetensors file, in name order) and `checksums_to_json` / `checksums_from_json` (the `{"tensors": {name: {dtype, shape, sum, abs_sum}}}` golden format shared with the Python script; doubles round-trip exactly). Pure C++, so it runs in CI and on machines without a GPU. (Part A)
- Each `CopyOp` in the weight layout now records the shape of the tensor it copies, so per-tensor results (like checksums) can be reported for tensors that live inside the fused `wqkv` and `wgu` regions. (Part A)
- Add `checksum_device(layout, device)`, which copies each original tensor's slice of the device weight buffer back to the host (in name order, reusing one host buffer sized to the largest tensor) and checksums it, so a match with Python proves the parser, layout, fusion offsets and upload together. Its GPU test uploads a fake checkpoint of valid BF16 values and requires the device checksums to equal the file checksums exactly. (Part A)
- Add `engine checksum --model <dir> [--out file.json] [--from-file]`: in CUDA builds it loads the weights onto the GPU (printing GB loaded, seconds and GB/s) and checksums the device copy; with `--from-file`, and always in CPU-only builds, it checksums the mapped file on the host. Prints name, dtype, shape, sum and abs-sum per tensor and optionally writes the golden JSON with a `source` field. `main` now reports exceptions from any command as `error: <message>` with exit code 1 instead of terminating. The command lives in `src/app/checksum_command.cpp` and passes the default stream to `load_weights`, so no CUDA header enters `src/app/`. (Part A)
- Add `scripts/checksum.py --model <dir> [--out file.json]`, the PyTorch side: it reads each tensor with the safetensors library, sums it in float64 with torch, and prints the same table and writes the same JSON as `engine checksum` (with `source: pytorch`). On the real checkpoint, `engine checksum --from-file` and this script agree on all 201 names, dtypes and shapes, with a worst sum difference of 0 and a worst abs-sum relative difference of 3e-16; the printed tables are byte-identical. (Part A)
- Add the checksum golden tests: `tests/support/checksum_golden.h` requires identical names, dtypes and shapes, `|sum - golden| <= 1e-6 * abs_sum` (6+ significant digits) and the abs-sum within 1e-6 relative; `ChecksumGolden.FileMatchesPyTorch` (unit) checks `checksum_file` and `ChecksumGolden.DeviceWeightsMatchPyTorch` (gpu) checks `checksum_device` after `load_weights`, which is the M1 exit criterion. Both skip when the model or `tests/golden/checksums.json` (generated by `scripts/checksum.py`, gitignored) is missing. Adds `golden_dir()` to `tests/support/paths.h`, overridable with `ENGINE_GOLDEN_DIR`. The file test passes on the real checkpoint and fails, naming the tensor, when one golden sum is nudged by 1e-5 of its abs-sum. (Part A)

## loader/safetensors-parser

- Implement `parse_safetensors_header`: reads the 8-byte little-endian header length (capped at 100,000,000 bytes and bounded by the file size), parses the JSON header with nlohmann, validates `__metadata__` as null or a string-to-string map, and for every tensor checks `dtype` (BF16, F16, F32, I32, I64), integer non-negative `shape`, and integer `data_offsets` with begin <= end. It then checks that each declared byte range equals the shape's byte size (with 64-bit overflow checks) and that the sorted ranges tile the data section exactly, with no gaps, overlaps or uncovered bytes. Whitespace was reformatted with clang-format. (Part C)
- Tighten `parse_safetensors_header`: `data_offsets` must be non-negative JSON integers read as `uint64_t` (before, `-1` or values above INT64_MAX passed through `int64_t` and wrapped, and only the tiling check caught them), the byte-size check and range collection happen in the one pass over the header instead of a second walk of the JSON, and a redundant `uint64_t` cast when storing shape dims is gone. (Part C, cleanup by Claude)
- Move `test_safetensors.cpp` from `tests/todo/unit/` to `tests/unit/` with `git mv`, add it to `unit_tests` and delete the now-empty `todo_unit_tests` target, so the parser's 28 tests run in the normal gate and in CI instead of being excluded by the `todo_joe` label. (Part A)

## loader/weight-upload

- Fix a debug-only race in `DeviceBuffer`: the NaN fill is a `cudaMemset` that runs asynchronously on the legacy default stream, which a non-blocking stream does not wait for, so a copy issued right after allocation could be overwritten by the fill. The constructor now synchronizes after the fill (debug builds only). (Part A)
- Add `plan_weight_layout(config, header)`, pure C++ that checks every tensor the config implies is present, BF16 and correctly shaped (and that nothing unexpected is in the file), then places all weights in one allocation with 256-byte aligned regions: Q, K and V fused into `wqkv` `[q + 2 kv, hidden]`, gate and up fused into `wgu` `[2 ff, hidden]`, and a copy list sorted by file offset so the file is read front to back. Adds `tests/support/fake_checkpoint.h` (tiny and TinyLlama-sized configs and headers) so the planner is tested without the real file or the parser. (Part A)
- Add `ModelWeights` / `LayerWeights` (BF16 `Tensor` views named after their role: `ln1`, `wqkv`, `wo`, `ln2`, `wgu`, `wdown`, plus `embed`, `final_norm`, `lm_head`) and `bind_weights(layout, base)`, which points every view into the single weight allocation; this is the struct the forward pass will take. (Part A)
- Add `upload_weights(file, layout, device, stream, chunk_bytes)`, which walks the copy list in chunks of up to 64 MB through two pinned staging buffers, with one CUDA event per buffer guarding its reuse, so the host's `memcpy` out of the mmap overlaps the PCIe transfer of the previous chunk. Returns bytes copied, seconds and GB/s. GPU tests upload a fake checkpoint with chunk sizes 1, 7, 64 and 1 MiB (so chunk boundaries fall inside tensors at odd offsets) and compare every destination byte with the source. (Part A)
- Add `load_weights(model_dir, stream)` returning `DeviceWeights` (config, layout, the one `DeviceBuffer`, bound `ModelWeights` and upload stats): it loads `config.json`, maps `model.safetensors`, parses the header, plans the layout, uploads, binds the views and unmaps the file on return. Its real-checkpoint test (shapes, the 2,200,096,768-byte total, and four whole tensors compared byte for byte, two of them inside fused regions) needs the parser, so it lives in the new `todo_gpu_tests` target with the `todo_joe` label. (Part A)

## docs/ignore-claude-notes

- Ignore `docs/claude/`, the local handoff notes (project context, build plan, setup guide, working rules, status, review and technical notes) that let a new Claude session continue without the original chat; also add the missing trailing newline to `.gitignore`. (Part A)

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
