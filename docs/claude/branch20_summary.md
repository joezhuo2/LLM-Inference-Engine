# Branch 20 summary: model/paged-runner (Part B)

Three local commits on 9a628be, not pushed. Gate green with the GPU (debug 364, cpu 186).

1. 32aae16 Paged batch builder: `include/engine/sched/paged_batch.h`, `src/sched/paged_batch.cpp`, 6 CPU tests.
2. b729bc4 `PagedRunner`: `include/engine/runtime/paged_runner.h`, `src/runtime/paged_runner.cpp`, 5 GPU tests (HF teacher-forced with 16 concurrent: 1167 of 1171 argmax, mean |dlogit| 0.0249).
3. 62837ba `serve()` and the M3 checks: `src/runtime/serve.cpp`, `tests/gpu/test_serve.cpp`.

Lines worth reading (Part B): paged_batch.cpp:38-48 (prefill rows versus the one decode row, `context_lens = pos + 1`, logit row), paged_runner.cpp:87-94 (packed int upload and slicing), paged_runner.cpp:112-115 (write_kv then paged attention per layer), paged_runner.cpp:126-135 (gather last rows, then norm and LM head on `s` rows only), serve.cpp:24-34 (one step).

Open for Joe: prefill via the decode kernel one row per token (decision recorded in m3_plan.md); `PagedRunner` separate from `ModelRunner`.
