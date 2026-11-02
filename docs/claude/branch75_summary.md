# Branch 75 `web/dashboard` summary (issues #91, #97, web half of #95)

Worktree ~/code/llm-inference-engine-dash, branch web/dashboard from origin/main 6ba8e13. Not pushed, no PR. Every commit was gated alone with `bash ~/.cache/llmie-dash/verify-web.sh` (the working tree held exactly the commit's files each time); each ends in `build: ok (6 html pages)` / `VERIFY WEB OK`. Tests: 22 pass (`npm test`).

## Commits

| Hash | Message |
| --- | --- |
| 1aa4554 | Record the commit that added each trace in the trace index |
| 037dab8 | Add the dashboard step model with BlockManager replay and tile stats |
| 8b83dc4 | Replay the recorded run on the dashboard with metric tiles |
| b62f6b8 | Add the batch timeline to the dashboard |
| 0d310af | Add the live KV block pool to the dashboard |
| 402281f | Add the requests panel to the dashboard |
| 8cd2f58 | Add replay controls to the dashboard |
| aa78c65 | Add live mode to the dashboard with a chat playground |
| 1ea4863 | Link the dashboard from the top bar and the landing page again |
| 93e34b7 | Draw dashboard canvas labels from cached sprites to keep frames cheap |
| 708944e | Fix dashboard fades after idle, lane history after a seek and the phone address field |

Each commit has its changelog.md entry (new `## web/dashboard` section) and blank learning-log terms under `## Dashboard (Branch 75)`.

## What it is

- `web/src/lib/dashboard/model.ts`: one step model for both modes. Reads step objects one at a time, keeps per-request text, color slot (8, reused), block table with the step each block was handed out, and mirrors the BlockManager's lowest-free-block rule. With the trace's prompt lengths it is exact (test: block for block and step for step equal to `scripts/kv-replay.mjs` on the recorded trace). Without them (live), it splits each step's change in `free_blocks` between the likeliest requests; the free count always equals the server's, 91% of per-request block counts are exact on the recorded run, and blocks it cannot place (requests that started before connecting, missed steps) are drawn gray as "request not seen". Index gaps are counted; an index going back means a server restart and starts a fresh model. Tests run the .ts directly through Node 22 type stripping (`scripts/dash-model.test.mjs`, `dash-sse.test.mjs`).
- Replay (default): plays `results/traces/chat_b16_rate4.json` in real time (1x, 2x, 4x; 6.4 s at 1x), labelled "recording" with date, GPU, commit (e1ec9e7, from `git log` of the trace at build time) and workload. Autoplays after the intro; reduced motion shows step 376 still.
- Tiles: tokens/s, running / waiting, KV blocks used of 12,915, TTFT and TPOT p50 / p99 (nearest rank). NumberFlow, 900 ms ease-in-out, updated once a second while playing (and at once on pause or scrub), the same cadence as polling /metrics in live mode. Tile layout never animates.
- Batch timeline (canvas): one row per step, newest at the bottom, bar length = step time on a 0 to 20 ms axis (» past 20 ms), prefill / decode colored, a square per sequence in its request color. The in-flight step slides in and grows with the replay clock, so it scrolls continuously. Hover / tap / arrow keys / Page Up/Down / Home / End read a step (kind, time, each sequence's sampled token, queues and blocks after it); wheel and keys scroll history when paused. Missed steps are amber lines with "N missed".
- KV pool (canvas), the 12,915-block choice: the allocator always hands out the lowest free block, so every held block of the recorded run is below block 128 (113 is the high-water mark). The grid zooms in on blocks 0 to 127 (16 columns), and a bar under it shows the whole pool with the zoomed range outlined, so the scale (0.2% used) stays honest. In live mode the window doubles when blocks above it are used (several blocks per square past 1,024). Blocks fill bottom-up to the slots written, fade in and out (0.22 s time constant), block 0 is outlined as the null block. Hover / arrow keys show request, logical block, slots, the step it was handed out at, and the prompt / generated text in its slots.
- Requests panel (replay): one lane per color with the request holding it (filled square while running, hollow once finished), its token count and the end of its answer, written straight to the DOM. Picking a lane highlights that request in the timeline and pool and shows its whole answer.
- Live mode: replay / live switch, address field (default http://localhost:8000). /health probe, /metrics every 1 s for the tiles, /v1/steps read with fetch + an SSE parser into the same model, chat playground streaming /v1/chat/completions (stop, clear, max tokens, temperature) that outlines its own sequence via the `engine.seq_id` field while it streams. States: not connected, connecting, live, nothing answered, answered but CORS blocked (told apart by a no-cors probe), HTTPS page blocked from the local network, mixed content (HTTPS page to plain http on another machine), not the engine (bad /health or /metrics), lost (retries every 3 s, keeps its state, resumes with missed steps counted), chat stream cut ("the engine went away after N tokens", "ended without [DONE]"). How-to box with `build/release/bin/engine serve --model models/tinyllama --cors-origin <this page's origin>`, Chrome's local network prompt, running the site locally as the fallback, and "keep it on localhost, do not expose the GPU". No tunnelling suggested anywhere.
- /dashboard back in the top bar, not-found page and the landing card. /results untouched.

## Verified, and how

- Gate per commit (above). Lint, typecheck, build with no warnings; 22 tests.
- Headless Windows Chrome over CDP on 9335, static server on 8092, scripts in ~/.cache/llmie-dash (`cdp.mjs`, `cdp-perf.mjs`, `cdp-prof.mjs`, `live-steps.sh`). Screenshots in ~/.cache/llmie-dash/shots: `d_dark_s120/s380/s700.png`, `d_light_*.png`, `d2_light_a.png` (desktop 1280x800, both themes, replay at three steps), `p_dark_full.png`, `p_light_full.png` (390 px phone, no horizontal scroll), `r_a.png` (reduced motion still), `k_tl.png`, `k_pool.png` (keyboard tooltips), `l1_*.png` and `f*_*.png` (live mode), `pl_*.png` (phone live and how-to), `tt_after.png` (theme switch redraw). No console errors in replay or a successful live session; the failure tests log only the browser's own network errors (connection refused, CORS blocked), which are expected.
- Mock engine `~/.cache/llmie-dash/mock-engine.mjs` (node, no deps): /health, /metrics, /v1/steps replaying the trace in real time and looping, streaming and non-streaming /v1/chat/completions that inject the chat sequence into the step stream with the `engine` field, CORS with OPTIONS and Access-Control-Allow-Private-Network; flags `--port`, `--cors-origin`, `--no-cors`, `--gap-every N`, `--chat-every N`, `--die-after S`. Tested against it: connect and live tiles / timeline / pool; chat streaming with its sequence highlighted; server down (8003: "nothing answered"); CORS missing (`--no-cors`: "answered, but the browser blocked the response (CORS)"); missed steps (`--gap-every 200`: "45 steps missed", drawn as gaps); server killed mid-stream ("stopped answering after step N; retrying every 3 s", then restarted: reconnected with a fresh model); server killed mid-chat ("the engine went away after 15 tokens").
- Frame cost over CDP (requestAnimationFrame callbacks timed in page, headless software rendering): playing 1x p50 1.5 ms, max 2.5 ms per frame; 4x p50 1.6 ms; paused and off screen 0 ms (no rAF work, no layouts). Before the sprite-label commit it was 5.3 ms: the first canvas text call of each frame re-resolved the web font (about 4.5 ms).
- Motion measured by sampling: a new block's fill goes 5% to 70% opacity in 0.23 s and settles by 0.8 s, a freed block fades the same way; NumberFlow tile animations run 900 ms cubic-bezier(0.65, 0, 0.35, 1) on its custom properties and are idle 1.2 s later.

## Not verified, needs Joe or the real server

End-to-end against the real `engine serve` (Branch 73) is the main open item. Check:
1. The `engine serve` flags in the how-to (`--model models/tinyllama --cors-origin ...`) and that it listens on 8000 by default; the copy hardcodes both.
2. CORS: OPTIONS preflight for the chat POST (Content-Type: application/json) and Access-Control-Allow-Origin on /health, /metrics, /v1/steps and the chat stream, for http://localhost:3000 and the Vercel origin.
3. That every running sequence appears in every decode step. The model treats a running sequence missing from a decode step as preempted and frees its blocks; if the server ever runs a partial decode batch, the pool attribution goes wrong (totals still match).
4. /metrics `kv.blocks_total` includes the null block (the tile computes used = total - 1 - free, as the trace does: 12,915 total, 12,914 free when idle).
5. `/v1/steps` framing (`event: step`, one JSON per `data:` line) and that a client disconnect from the chat stream frees the sequence.
6. Private Network Access / Local Network Access from the deployed HTTPS site to localhost: only testable from the Vercel URL in a real Chrome (headless here served from http://localhost). Safari and Firefox behaviour untested.
7. Animation smoothness and frame cost in Joe's GPU Chrome (measured only headless).
8. One flaky observation: a single phone run sat on "connecting" for over 2.5 s; three reruns connected in about 0.3 s. Not reproduced.

## Open questions for Joe

- Speeds are 1x, 2x and 4x as asked; the whole run is 6.4 s at 1x, so a slower choice (0.25x or 0.5x) may suit the "slower, smoother" preference better.
- Live attribution is a best guess because the step object has no prompt lengths. An additive field (for example `prompt_tokens` on prefill steps, or block ids per sequence) would make the live pool exact without a version bump.
- The trace's commit comes from `git log` at build time and is null in a shallow clone (CI, possibly Vercel), which drops it from the label. `meta.commit` written by `engine bench --trace` would be more robust.
- The chat highlight clears when the answer finishes; keep it until the next message instead?

## Draft PR description

Dashboard: replay a recorded run or connect to engine serve (#91, #97, web half of #95)

- `/dashboard` replays `results/traces/chat_b16_rate4.json` in real time, labelled as a recording (date, GPU, commit, workload), with play, pause, step, 1x/2x/4x and a scrubber.
- Metric tiles (tokens/s, running/waiting, KV blocks used, TTFT and TPOT p50/p99) with NumberFlow, a canvas batch timeline (one row per step, width = step time, prefill/decode colored, a square per sequence), a 2D KV block pool (zoomed on blocks 0 to 127 where lowest-free-first allocation keeps every held block, plus a whole-pool bar), and a requests panel with streaming answers. Hover, tap and keyboard read exact values.
- Live mode connects to `engine serve` (docs/api.md): /health, /metrics every second, /v1/steps stream, and a chat playground that outlines its own sequence. Clear states for nothing listening, CORS, local network and mixed-content blocks, a server going away, and missed steps, with a short how-to. Never suggests exposing the server.
- One step model for both modes, exact against the build-time KV replay with prompt lengths, reconciled with the server's free count without them; tests run with Node type stripping.
- /dashboard is linked again from the top bar and landing page.
- Checked in headless Chrome: both themes, desktop and 390 px, reduced motion, keyboard, a mock server for every live failure state; replay frames cost about 1.5 ms of script, nothing runs when paused or off screen.
- Still to check: end-to-end against the real `engine serve` (list in docs/claude/branch75_summary.md).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
