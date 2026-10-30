# Branch 48 `docs/api-contract` (PR summary)

First branch of the web plan (`web_plan.md`). Built in a second worktree, `~/code/llm-inference-engine-api`, while another agent built `docs/m5-readme` in `~/code/llm-inference-engine-claude`. Four commits on 1172a22 (#55), not pushed:

1. 7727068 `docs/api.md`: the step object, the trace file (`llmie-trace` version 1), and the planned server endpoints (`/health`, `/metrics`, `/v1/steps` SSE, OpenAI-compatible `/v1/chat/completions` with an `engine` extension field, CORS with `Access-Control-Allow-Private-Network`). Additive fields don't bump the version.
2. a8e5a7f `StepTrace` and `trace_json` in engine_core (`include/engine/bench/trace.h`), 5 CPU tests. Token text comes from a callback called in step order, so the library stays tokenizer-free.
3. 82a12bf `serve_step`/`run_batched` take an optional `std::vector<StepTrace>*`; `engine bench --trace out.json` (batched only) writes the timed run with one `TextStream` per sequence. GPU test: 16 greedy requests in 56 blocks (forced preemption), the trace's tokens rebuild every output exactly; a mutant that skips prefill steps fails it.
4. 7175e48 `results/traces/batching_b16_rate4.json`: 24 requests at 4/s, max batch 16, graphs, temperature 0.8; 664 steps over 6.5 s, up to 8 sequences together, 177 KB.

Gates: commits 1 and 2 passed debug 575/580 and cpu 192/197, BUT the new worktree lacked the gitignored HF reference dump, so the reference-based GPU tests were skipped in those two runs (commit 1 is docs only and commit 2 is CPU-only code, so nothing they touch depends on them). Fixed by symlinking `tests/golden/reference` to the main checkout's copy; commits 3 and 4 ran the full suite (debug 581, cpu 197).

Notes for Joe:
- The first token of a sequence has no leading space in `text` (SentencePiece decoding drops it); documented.
- The trace text is nonsense because bench prompts are random token ids. Follow-up worth doing before the hero: `engine bench --prompts file.json` with real chat prompts so the rain shows readable text.
- Subtle lines: `src/runtime/serve.cpp` records `end` after `scheduler.update`, so a step's end includes the host-side update (microseconds); `src/app/bench_command.cpp` passes the trace vector only on the timed run (`timed` flag on the `drive` lambda), never the warmup.
