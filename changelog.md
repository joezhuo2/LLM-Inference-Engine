# Changelog

Changes are grouped by branch, newest first, in the order the branches merge into `main`. Each entry is tagged with its part of the delegation: Part A and Part B entries are written by Claude and reviewed by Joe, and Part C entries are written by Joe.

## core/tensor-and-buffers

- Add the `DType` enum, `dtype_size`, and a non-owning `Tensor` view (data pointer, dtype, up to 4 dims, `numel`, `bytes`) as the common currency between the loader, kernels and runtime. (Part A)
- Set `ReflowComments: false` in `.clang-format` so long comments stay single-line paragraphs instead of being wrapped at 100 columns; no existing file changes formatting. (Part A)
- Add `engine::Stream`, an alias for `cudaStream_t` declared through a forward-declared `CUstream_st`, so kernel launchers in `include/engine/` can take a stream without including CUDA headers; a static_assert in `gpu_tests` proves the types are identical. (Part A)
- Add `CUBLAS_CHECK` (prints the cuBLAS status name and message, then aborts) and `KERNEL_CHECK(stream)` to `src/kernels/cuda_check.h`; in debug builds `KERNEL_CHECK` also synchronizes the stream so asynchronous faults point at the right kernel, except while the stream is being captured into a CUDA Graph. Death tests cover both abort paths. (Part A)

## build/m1-scaffolding

- Add CLI11 (found on the system first, fetched otherwise) and turn `engine` into a subcommand CLI with a `version` subcommand; running it with no arguments still prints the version. (Part A)
- Rename the `json` FetchContent dependency to `nlohmann_json` so CMake stops warning about a package name mismatch on every configure. (Part A)
- CI runs `ctest -LE todo_joe` so tests for unimplemented Part C components do not fail the build, installs CLI11 from apt and Homebrew, and adds a job that byte-compiles every tracked Python file. (Part A)
- `.clangd` reads `build/debug/compile_commands.json` directly, so the root symlink is no longer needed. (Part A)
- Add `scripts/requirements.txt` pinned to the `~/.venvs/llmie` versions, with the cu128 index for an sm_120 torch build. (Part A)
