# M1 plan

The plan Joe approved on Oct 3, 2026, updated with what each branch actually delivered. M1 exit criteria: the CMake project builds for `sm_120`; a bandwidth test prints a GB/s number; the cuBLAS GEMM test passes against a CPU reference; the safetensors loader prints every tensor name, shape, dtype and a checksum matching Python.

## Branches

| # | Branch | Part | State |
| --- | --- | --- | --- |
| 1 | build/m1-scaffolding | A | Merged (#6) |
| 2 | core/tensor-and-buffers | A | Merged (#7) |
| 3 | kernel/cublas-gemm | B | Merged (#8) |
| 4 | model/config | A | Merged (#9) |
| 5 | loader/safetensors-stub | A plus C stub | Merged (#10) |
| extra | build/ci-timeouts | A | Merged (#11) |
| 6 | loader/weight-upload | A | Done, pushed, not merged |
| Joe | (his branch) parse_safetensors_header | C | In progress |
| 7 | tools/checksum | A | Waits for the parser |
| 8 | docs/m1-readme | A | After 7 |

## What each branch contains

1. **build/m1-scaffolding:** CLI11 v2.4.2 via FetchContent with find_package first; `engine` with a `version` subcommand (no arguments still prints the version); `FetchContent_Declare(nlohmann_json ...)` naming fix; CI runs `ctest -LE todo_joe`, installs `libcli11-dev` and `brew cli11`, and byte-compiles tracked Python files; `.clangd` has `CompilationDatabase: build/debug`; `scripts/requirements.txt` pinned to llmie (torch 2.11.0 via the cu128 index, transformers 5.17.0, tokenizers 0.23.2, safetensors 0.8.0, sentencepiece 0.2.2, huggingface_hub 1.33.0, numpy 2.5.3, pandas 3.0.6, matplotlib 3.11.2).
2. **core/tensor-and-buffers:** `include/engine/core/tensor.h` (`DType {BF16, F16, F32, I32, I64}`, `dtype_size`, `Tensor` with `std::array<int64_t, 4>` shape and `ndim`, `numel`, `bytes`; more than 4 dims throws); `stream.h`; `CUBLAS_CHECK` and capture-aware `KERNEL_CHECK(stream)` in `src/kernels/cuda_check.h`; `engine_runtime` library; move-only `DeviceBuffer` (debug NaN fill) and `PinnedBuffer`; changelog; `ReflowComments: false`.
3. **kernel/cublas-gemm:** `tests/support/bf16.h` (round-to-nearest-even, checked against `__float2bfloat16` on a million random floats); `Blas(stream, workspace, bytes)` binding one stream then the workspace; `gemm(blas, y[B,out], x[B,in], w[out,in])`; tests: exact 3x5 hand example, shape mismatches, CPU-reference sweep including B=1, odd sizes, 2560x2048, 5632x2048 and 2048x32000.
4. **model/config:** `ModelConfig` (hidden, layers, heads, kv_heads, head_dim, intermediate, vocab, max_positions, rope_theta, rms_eps as double, bos_id, eos_id, tie_embeddings; `kv_dim`, `qkv_dim`, `group_size`); rejects non-llama, RoPE scaling, non-SiLU, biases, `pretraining_tp` above 1, indivisible heads, mismatched `head_dim`; `tests/support/paths.h` (`model_dir()`, overridable with `ENGINE_MODEL_DIR`).
5. **loader/safetensors-stub:** `MappedFile` (move-only, `mmap(PROT_READ, MAP_PRIVATE)`, fd closed after mapping, empty file gives an empty span, errors are `std::system_error` with errno and path); `include/engine/loader/safetensors.h` (`TensorEntry {dtype, shape, offset (absolute), bytes}`, `SafetensorsHeader {data_offset, std::map tensors}`, `parse_safetensors_header(std::span<const std::byte>)`); the stub with its TODO(JOE) spec; `tests/support/safetensors_bytes.h`; 28 todo_joe tests including `RealTinyLlamaHeader` (201 tensors).
6. **loader/weight-upload:** see `status.md`.
7. **tools/checksum and 8. docs/m1-readme:** merged as #15, #16 and #17; M1 closed Oct 9 after Joe's GPU run (see `status.md`).

## How Joe verifies a branch

```
cmake --preset debug && cmake --build --preset debug && ctest --preset debug -LE todo_joe
ctest --preset debug -L todo_joe
compute-sanitizer --tool memcheck --leak-check full ./build/debug/bin/gpu_tests
cmake --preset cpu && cmake --build --preset cpu && ctest --preset cpu -LE todo_joe
```

After branch 7 and the parser: `source ~/.venvs/llmie/bin/activate && python scripts/checksum.py --model models/tinyllama --out tests/golden/checksums.json`, then `./build/debug/bin/engine checksum --model models/tinyllama`.
