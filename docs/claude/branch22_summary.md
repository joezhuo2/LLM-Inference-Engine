# Branch 22 `kernel/m4-stubs` (M4 Part C stubs for Joe)

Local worktree branch on f552fd2 (origin/main after #34), 6 commits, not pushed, gate green with GPU (debug 368, cpu 188). Rebase-merge safe: every commit passes the gate alone.

| Commit | Content |
| --- | --- |
| 913752b | `fast::add_rmsnorm` stub, TODO(JOE) spec, 25 todo_joe tests, `tests/support/guarded.h` |
| fba285c | `fast::rope` stub, spec, 9 todo_joe tests |
| eddde49 | `fast::swiglu` stub, spec, 23 todo_joe tests |
| d9f9ca9 | `pad_decode_batch` (Part B) + 2 CPU tests |
| 272c78f | `PagedRunner::forward` split into `stage` + `run_staged` (pinned staging, event fence, no host reads for decode) + 2 GPU tests |
| c1cdb9c | `PagedRunner::logits(rows)`, `DecodeGraphs` stub, spec, 8 todo_joe tests |

## What Joe implements (all todo_joe, run `ctest --preset debug -L todo_joe`)

1. `src/kernels/add_rmsnorm.cu`: sum bit-identical to naive add, norm within reduction order, batch invariant, any hidden, unaligned pointers, sum may alias x or residual.
2. `src/kernels/rope_fast.cu`: bit-identical to `naive::rope` (no fast math, no FMA), any even head_dim up to 256.
3. `src/kernels/swiglu_fast.cu`: bit-identical to `naive::swiglu` including inf, NaN, signed zeros.
4. `src/runtime/decode_graphs.cpp`: per-bucket capture in the constructor, replay in `forward`.

Each spec ends with the same move-to-`tests/gpu/` and memcheck/ncu steps as earlier stubs.

## Decisions I made (tell me if you disagree)

- New kernels live in `engine::fast::` next to `engine::naive::`, same signatures, same headers, so the runner switches by changing the namespace.
- Fast kernels are bit-identical to naive (RMSNorm excepted: summation order only), so HF validation carries over and a regression is a one-line diff.
- Graphs are captured eagerly in the constructor from an all-padding batch (slot -1, context 0), which leaves the KV cache untouched.
- A replay is bit-identical to the eager step at the bucket size, not to the unpadded eager step (different GEMM shape, possibly different cuBLAS algorithm); the tests encode exactly that.
- Prefill is not graphed.

## Not done on purpose

Wiring fused/fast kernels into `PagedRunner` and `DecodeGraphs` into `serve()`, `--graphs`, `--max-batch` and peak KV blocks in `engine bench`: branch 23, after Joe's implementations pass.

## Validation

Every todo test file was run against a throwaway implementation (naive delegation, and a real relaxed-mode capture for graphs) and passed, then the stub was restored. The graph throwaway proves capture of the whole runner works under cuBLAS with the fixed workspace.
