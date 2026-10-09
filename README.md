# llm-inference-engine

A single-GPU LLM inference engine in C++20 and CUDA that serves [TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) in BF16 with FP32 accumulation. The goal is an engine whose output matches HuggingFace token for token and that then gets fast: custom fused kernels, a paged KV cache with a paged decode attention kernel, continuous batching, sampling on the GPU, and CUDA Graphs for decode. GEMMs go through cuBLAS.

The rule is correct first, then fast: nothing is optimized until the naive engine produces the same text as HuggingFace.

## Status

| Milestone | Due | Exit criteria | Status |
| --- | --- | --- | --- |
| M1 Foundations | Oct 7, 2026 | Builds for `sm_120`; a bandwidth test prints GB/s; the cuBLAS GEMM test passes against a CPU reference; the loader prints every tensor's name, shape, dtype and a checksum matching PyTorch | Implemented; final GPU verification of the device checksum pending |
| M2 Correct naive engine | Oct 18, 2026 | Batch-1 greedy generation produces readable text; per-layer hidden states match HF within tolerance; teacher-forced argmax match of 99%+ | Not started |
| M3 Paged KV and batching | Nov 1, 2026 | Block manager with unit tests; paged decode attention matches the naive kernel; 16 concurrent requests each produce correct output | Not started |
| M4 Fast | Nov 8, 2026 | Vectorized and fused kernels; CUDA Graphs for decode; batch-1 decode at 60%+ of measured bandwidth; no launch gaps in the nsys timeline | Not started |
| M5 Shipped | Nov 15, 2026 | Benchmark CSVs (laptop and one cloud GPU), plots, results and reproduce steps here, resume bullets from real numbers | Not started |

What works today:

- `engine version` and `engine checksum`, a CLI built on CLI11.
- Loading TinyLlama onto the GPU: `config.json` parsing and validation, a read-only `mmap` of `model.safetensors`, a hand-written safetensors header parser, a layout planner that fuses Q, K and V into `wqkv` and gate and up into `wgu` inside one device allocation, and a double-buffered upload through pinned staging buffers.
- A BF16 cuBLAS GEMM wrapper (`y = x W^T` on row-major tensors) tested against a CPU reference at every TinyLlama projection shape.
- FP64 checksums of every weight tensor, from the file or from the uploaded device copy, compared against PyTorch.
- Measured memory bandwidth on the RTX 5060 Laptop GPU: 323.6 GB/s, about 84% of the ~384 GB/s theoretical peak (`src/practice/bandwith_test.cu`, notes in `docs/exercises/bandwith_test.md`). This is the denominator for every "% of peak" number later.

## Hardware and toolchain

Developed on an RTX 5060 Laptop GPU (Blackwell, compute capability 12.0, 8 GB GDDR7, 26 SMs) under WSL2 with Ubuntu 24.04; `docs/hardware.md` has the device query. These are the versions in use:

| Tool | Version | Notes |
| --- | --- | --- |
| CUDA Toolkit | 13.3 | 12.8 or newer is required for `sm_120`; install the toolkit only from NVIDIA's WSL-Ubuntu repo, never a Linux driver inside WSL |
| GCC | 13.3 | Host compiler, C++20 |
| CMake | 3.28 | Minimum 3.28 |
| Ninja | 1.11 | Generator used by every preset |
| clang-format | 18 | CI checks formatting with major version 18 |
| GoogleTest | 1.15.2 | Found on the system first, fetched otherwise |
| nlohmann/json | 3.11.3 | Found on the system first, fetched otherwise |
| CLI11 | 2.4.2 | Found on the system first, fetched otherwise |
| Python | 3.12 | Reference and golden scripts only, never on the engine's hot path; packages pinned in `scripts/requirements.txt` (torch 2.11.0 from the cu128 index, which has `sm_120` support) |

The default CUDA architecture is 120. To also build for cloud GPUs, configure with `-DCMAKE_CUDA_ARCHITECTURES="120;89;75"`, as CI does.

## Getting the model

```bash
hf download TinyLlama/TinyLlama-1.1B-Chat-v1.0 --local-dir ~/models/tinyllama
```

Tests and examples look for the model in `models/tinyllama` inside the repo. `models` is a symlink to `/home/crystalflux/models`; on another machine, point it at your own model directory or set `ENGINE_MODEL_DIR` to the TinyLlama directory when running tests. Tests that need the model skip themselves when it is missing (as in CI).

## Building

There are three presets, each building into `build/<preset>`:

| Preset | Build type | Use |
| --- | --- | --- |
| `debug` | Debug | Development; debug builds fill new device buffers with NaN and synchronize after every kernel launch |
| `release` | Release | Every benchmark; never benchmark a debug build |
| `cpu` | Debug, `ENGINE_FORCE_CPU=ON` | Machines without CUDA (macOS included); builds only the pure C++ targets |

```bash
cmake --preset debug && cmake --build --preset debug
```

Binaries land in `build/<preset>/bin`: `engine`, `unit_tests` and, in CUDA builds, `gpu_tests`. Without a CUDA compiler, the debug and release presets also fall back to a CPU-only build.

## Testing

```bash
ctest --preset debug -LE todo_joe
```

Tests carry CTest labels:

| Label | Meaning |
| --- | --- |
| `gpu` | Tests in `gpu_tests`, which need a CUDA device; `ctest -LE gpu` runs only the CPU tests |
| `todo_joe` | Tests for a hand-written component that does not exist yet (see below); they fail on purpose until it does |
| none | Pure C++ tests in `unit_tests`; they run everywhere, including CI on Ubuntu and macOS |

Check the GPU tests for memory errors and leaks with compute-sanitizer:

```bash
compute-sanitizer --tool memcheck --leak-check full ./build/debug/bin/gpu_tests
```

CI builds and tests the `cpu` preset on Ubuntu 24.04 and macOS, compile-checks the CUDA code in the `nvidia/cuda:13.3.1-devel-ubuntu24.04` container for architectures 120, 89 and 75 (there is no GPU in CI), checks formatting with clang-format 18, and byte-compiles every Python script. Format the tree before committing:

```bash
scripts/format.sh
```

### The todo_joe workflow

Core components (the safetensors parser, the KV block manager, the scheduler, the paged attention kernels, the fused kernels and CUDA Graph capture) are written by hand; `docs/ownership.md` explains the split. Before such a component exists, the repo holds a stub so everything compiles and links: host stubs throw `std::logic_error("unimplemented: <name>")`, and a `TODO(JOE)` block above the stub specifies the interface, shapes, edge cases and what the result must match. Its tests go in `tests/todo/unit/` (target `todo_unit_tests`) or `tests/todo/gpu/` (target `todo_gpu_tests`), created along with the stub, with the label `todo_joe`, which CI and the normal test command exclude.

1. Implement the component until `ctest --preset debug -L todo_joe` passes.
2. Move the test file into `tests/unit/` or `tests/gpu/` with `git mv`, add it to `unit_tests` or `gpu_tests`, and delete the todo target once it is empty.

No component is in this state right now: the safetensors parser went through it and its 28 tests now run in `unit_tests`.

## Weight checksums (the M1 check)

`engine checksum` prints, for every weight tensor, its name, dtype, shape, the FP64 sum of its values and the FP64 sum of their absolute values. In CUDA builds it loads the weights onto the GPU first (printing GB loaded, seconds and GB/s) and checksums the device copy, which checks the parser, the layout, the fused-weight offsets and the upload together. `--from-file` checksums the mapped file on the host instead, which is also what CPU-only builds do.

```bash
./build/debug/bin/engine checksum --model models/tinyllama
```

```bash
./build/debug/bin/engine checksum --model models/tinyllama --from-file --out file.json
```

`scripts/checksum.py` computes the same table with the safetensors library and PyTorch, and writes the golden file the tests compare against:

```bash
source ~/.venvs/llmie/bin/activate && pip install -r scripts/requirements.txt
```

```bash
python scripts/checksum.py --model models/tinyllama --out tests/golden/checksums.json
```

With that file present, `ChecksumGolden.FileMatchesPyTorch` (in `unit_tests`) and `ChecksumGolden.DeviceWeightsMatchPyTorch` (in `gpu_tests`) require the same 201 tensor names, dtypes and shapes and sums that agree to at least 6 significant digits of each tensor's absolute sum. Both skip when the model or the golden file is missing; `tests/golden/` is gitignored and always regenerated by a script. `ENGINE_GOLDEN_DIR` overrides its location.

## Repository layout

```
include/engine/     public headers, plain C++ with no CUDA includes so the CPU build works
  core/             DType, Tensor view, Stream, version
  kernels/          GEMM and kernel launcher declarations
  loader/           MappedFile, safetensors header, weight layout, file checksums
  model/            ModelConfig, ModelWeights
  runtime/          DeviceBuffer, PinnedBuffer, load_weights, device checksums
  kv/ sched/ sampling/ tokenizer/   empty, for later milestones
src/
  core/ loader/ model/   engine_core: pure C++, builds everywhere
  kernels/               engine_kernels: .cu files, cuBLAS, cuda_check.h (CUDA builds only)
  runtime/               engine_runtime: CUDA host code (CUDA builds only)
  app/                   the engine CLI
  practice/              learning exercises (reductions, bandwidth test, PyTorch TinyLlama), not part of the engine
tests/
  unit/                  unit_tests, pure C++
  gpu/                   gpu_tests, label gpu
  support/               test helpers: BF16 conversion, fake checkpoints, safetensors bytes, paths
  golden/                generated reference data (gitignored)
scripts/                 checksum.py, format.sh, requirements.txt
docs/                    hardware.md, ownership.md, exercises/ (learning log and measurement notes)
changelog.md             every change, grouped by branch and tagged by authorship
```

CUDA headers appear only in `src/kernels/` and `src/runtime/`; `engine::Stream` and forward-declared cuBLAS types keep them out of `include/engine/`.

## Contributing conventions

- `main` stays green: it builds, and every test not labelled `todo_joe` passes. PRs are merged with rebase and merge, so every commit must build and pass on its own.
- Commit messages are one imperative line.
- Every change adds an entry to `changelog.md`, newest branch first, tagged `(Part A)`, `(Part B)`, `(Part C)` or `(Part C stub)`; `docs/ownership.md` explains the tags.
