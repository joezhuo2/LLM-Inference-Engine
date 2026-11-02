# Branch 73 `server/http` summary (issue #95, engine half)

Worktree ~/code/llm-inference-engine-server, branch server/http from origin/main 6ba8e13. Not pushed, no PR. Clean tree at 12caf35. `engine serve` works end to end on the real model, including the real dashboard in live mode. One piece is a stub: `Scheduler::cancel` (Part C, Joe's), so a disconnected client's blocks only come back when its sequence finishes.

## Commits

Each was gated alone with `bash ~/.cache/llmie-server/verify.sh` with the GPU up; the last lines are in ~/.cache/llmie-server/layer.log and layer2.log.

| Hash | Message | ctest debug / cpu |
| --- | --- | --- |
| 592d5df | Split the step object out of the trace writer as step_json | 588 / 204 |
| 26a6487 | Add the engine_server library with the OpenAI chat request and chunk format | 600 / 216 |
| d4fb744 | Add the CORS policy with preflight and private network headers | 608 / 224 |
| e7150d4 | Add a bounded channel and the step broadcaster for the live step stream | not kept (session ended; the commit exists, so its gate passed) |
| ffb567d | Add the sliding windows behind the metrics endpoint | 622 / 238 |
| 3e31a96 | Add stop string matching across token boundaries | 629 / 245 |
| 94517f8 | Add the Scheduler::cancel stub with spec and todo_joe tests | 629 / 245 |
| a8bd12d | Add the serving loop that runs the scheduler on one engine thread | 639 / 255 |
| ae25d0c | Add the HTTP server with chat, metrics, step stream and CORS | 650 / 266 |
| e5ab3d4 | Add engine serve on the paged runner with a GPU integration test | 651 / 266 |
| 12caf35 | Document engine serve in the API contract and the README | 651 / 266 |

The first gate of 94517f8 failed on a link error caused by my commit script, not the code: `cp -a` kept old mtimes, so ninja did not rebuild scheduler.cpp. Nothing was committed from that run; it was rebuilt with fresh mtimes and passed.

## Stubs (Part C)

`Scheduler::cancel` in `src/sched/scheduler.cpp` throws `logic_error("unimplemented: Scheduler::cancel")`, with a TODO(JOE) spec above it. The 11 tests in `tests/todo/unit/test_scheduler_cancel.cpp` (target `todo_unit_tests`, label `todo_joe`) fail against the stub, as they should. They passed against a throwaway implementation and caught 5 mutants; the running-list reorder mutant needed one test added.

Until Joe implements it, `ServingLoop::release` catches the `logic_error` and prints one warning. A cancelled or stop-string sequence keeps decoding with its tokens discarded, and holds its blocks and a batch row until `max_tokens` or EOS. `submit` needed no Part C work: the scheduler already accepts requests between steps. No other Part C code changed.

Run when implemented: `ctest --preset cpu -L todo_joe`, then move the file into `tests/unit/` and delete the target.

## Verified end to end (release build, RTX 5060, GPU up)

Transcripts are in ~/.cache/llmie-server/e2e/: `transcript.txt`, `slow_client.txt`, `server*.log`, `dash_during.png`, `dash_after.png`, `chat_capital.png`, `chat_stopped.png`. They ran on the final code except for a missing-include fix in `stop_matcher.h`, made after them.

- /health, /metrics, streamed and non-streamed completions, a stop list, a 400.
- 8 concurrent completions with /v1/steps: peak running 8, 122 steps, no index gaps, every decode step held exactly the running set.
- Disconnect mid-stream: blocks came back 3.11 s later, when the 400-token limit was reached, not at once (stub). `finished_total` did not count it.
- OPTIONS from http://localhost:3000 with the private-network header: 204 with `Access-Control-Allow-Private-Network: true`. From another origin: 403 with no allow headers.
- `--max-batch 2` with 6 requests: peak waiting 4.
- gpu test `ServerTest.StreamsTheGreedyAnswerOfGenerate`: four concurrent streams equal `generate` on the `capital` prompt.

## Check against the dashboard (Branch 75's list)

I ran the real dashboard build (web/out of the dash worktree on http://localhost:8093, headless Windows Chrome) against the real server: connect, tiles, timeline, pool, chat, stop. No console errors, "no steps missed".

1. Flags and port: match. `--model models/tinyllama --cors-origin <origin>` works, default port 8000, and the flag repeats. It replaces the default list rather than adding to it.
2. CORS: `Access-Control-Allow-Origin` is on /health, /metrics, /v1/steps and the chat stream, checked with curl for http://localhost:3000 and in the browser for http://localhost:8093. The Vercel origin is untested because the domain is unknown.
3. Every running sequence in every decode step: holds in the code (`schedule` walks the whole running list) and measured with no preemption. Preemption was not reached live. Once `Scheduler::cancel` exists, a cancelled sequence vanishes from decode steps without being in `finished`; the model already reads that as preempted and frees its blocks, which is the right result.
4. `kv.blocks_total` includes the null block: 12,915 total, 12,914 free when idle. Matches.
5. Framing matches. The server also sends `: keepalive` comments, which `sse.ts` already skips. The role chunk's `engine` has `seq_id` but no `step`; `playground.tsx` types `step` as a number but does not read it. Disconnect: see the stub above.
6. Private Network Access from the HTTPS site: not tested, needs the deployed URL in a real Chrome.

Mismatches: none that break the dashboard. Two small differences to know about: the default origins omit Vercel, and a disallowed origin now gets 403 (not in the mock). The dashboard's "CORS blocked" state should still work, since its no-cors probe cannot see the status, but I did not run that state against the real server.

## Deviations from docs/api.md (all recorded there)

Shapes are unchanged. Additive or narrowed:

- Default origins are only http://localhost:3000. The Vercel domain appears nowhere in the repo or on GitHub, and guessing one would let whoever owns that name call the local server.
- A request with a disallowed `Origin` gets 403 and starts no work (new). Otherwise any site could spend the GPU with a text/plain POST, which skips the preflight.
- `seed` is accepted; the error body is `{"error":{"message","type"}}`; the role chunk's `engine` has only `seq_id`.
- "Frees its blocks" on disconnect is not immediate until the stub is implemented.

## Not verified

- Preemption live: `preemptions_total` stayed 0 (the pool is too large to fill). Covered by a CPU test with a small pool.
- Step dropping live: a client that did not read still got all 361 steps with no gaps, because they fit in its queue and socket buffers. The engine was not slowed. Dropping itself is unit-tested only.
- compute-sanitizer, and GitHub CI (the new tests use loopback sockets).
- HTTPS site to localhost, Safari, Firefox.

## Open questions for Joe

1. The Vercel domain: add it to `kDefaultCorsOrigins` in `include/engine/server/cors.h` once known. Until then the deployed site needs `--cors-origin`, as its how-to already says.
2. Implement `Scheduler::cancel` when convenient; I recommend it before showing the playground, since a stopped answer holds a batch row until its limit.
3. DNS rebinding is not covered: a rebinding page can read /metrics and /v1/steps (GETs from its own origin send no Origin header), though the POST is still refused. A Host header check on loopback would close it in a few lines. I left it out as beyond the brief; I recommend adding it.
4. `engine bench --prompts` encodes chat prompts with BOS while `generate` and the server use none (HF's `apply_chat_template` adds none). This is older than this branch and I did not change it, but the recorded trace's prompts are one token longer than a server's would be.
5. No warmup: the first batch after start is slow (TTFT 0.85 s for 6 requests without graphs, later ones about 40 ms in a run with graphs). Add one at startup if the dashboard's first numbers matter.
6. No limit on queued requests and no rate limit. Fine on loopback.

Greedy output can differ from `engine generate` on longer answers, as a BF16 near-tie between the paged and contiguous paths (a KV cache prompt diverged around token 25 here). This is the behavior `tests/gpu/test_serve.cpp` already tolerates, so the gpu test uses the short `capital` prompt, and docs/api.md says so.

`OwnedStream` and `graph_buckets` are duplicated in `src/server/gpu_backend.cpp` from `bench_command.cpp`, and `gpu_name` too; small, and left rather than refactoring bench.

## Line-by-line review (Part B)

- `src/server/serving_loop.cpp`: `handle` (order: texts, publish, deliver, finished, release; a sequence can stop and finish in the same step), `release` (catches `invalid_argument` first because it derives from `logic_error`).
- `src/server/gpu_backend.cpp`: member order sets destruction order (graphs, then runner, then stream).

## Draft PR description

Serve the engine over HTTP: `engine serve` (engine half of #95)

- `engine serve`: `POST /v1/chat/completions` (OpenAI-compatible, streamed or not, with the `engine` extension), `GET /health`, `GET /metrics` (10 s and 60 s windows), `GET /v1/steps` (SSE, the same step object as the trace, written by the same `step_json`).
- One engine thread runs the scheduler; HTTP workers talk to it through queues. A slow step subscriber drops its oldest steps and never blocks the engine.
- CORS with `--cors-origin`, preflight and `Access-Control-Allow-Private-Network`; requests from other origins get 403. Binds 127.0.0.1, no auth: never expose or tunnel it.
- cpp-httplib v0.61.0 via FetchContent, pinned by tag and hash, no TLS or compression.
- Part C stub: `Scheduler::cancel` with its spec and 11 todo_joe tests. Until it is written, a disconnected client's sequence runs to its limit before its blocks come back.
- Tests: CPU tests for parsing, chunks, CORS, windows, broadcaster, stop strings, the serving loop and the HTTP server over real sockets on a scripted backend; a gpu test comparing four concurrent streams with `generate`.
- Checked by hand on the RTX 5060, including the dashboard's live mode from a localhost origin. Not checked: the deployed HTTPS site to localhost, preemption live.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
