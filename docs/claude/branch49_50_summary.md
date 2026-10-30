# Branches 49 and 50 (PR summaries)

## Branch 49 `bench/chat-prompts` (worktree ~/code/llm-inference-engine-api), 2 commits on 0d11053, not pushed

1. 8f0f317 `load_chat_prompts` (engine_core) reads the scripts/prompts.json format into token ids of the chat template; `WorkloadConfig::prompts` makes request i use prompt i mod n. Random workloads are unchanged (only prompts mode skips the prompt length draw), so the Python port stays in sync. CPU tests: template round trip, the 20 repository prompts, malformed files, prompts used in turn, empty prompt rejected. Gate: debug 586, cpu 202.
2. e1ec9e7 `engine bench --prompts file.json` (one tokenizer shared with `--trace`; the prompt length range becomes the real prompts' min:max so context and prefill sizing and the CSV fit; trace meta `prompts`), the README bench paragraph, docs/api.md, and the trace replaced: `results/traces/chat_b16_rate4.json` (20 chat prompts, 24 requests at 4/s, `--stop-at-eos`, temperature 0.7, top-p 0.9; 751 steps over 6.4 s, answers of 9 to 191 tokens, readable text). The old `batching_b16_rate4.json` is deleted. Gate green (the commit was chained on it).

## Branch 50 `web/scaffold` (worktree ~/code/llm-inference-engine-web), 3 commits on 0d11053, not pushed

1. bbbee34 Next.js 16.4.3 static export in `web/`: design tokens (paper light and dark, phosphor accent, easing and duration variables), Inter and JetBrains Mono, next-themes with dark default, top bar, heading-only pages.
2. 08df170 `web/scripts/sync-data.mjs` (results CSVs, environment.txt and traces into gitignored JSON), headline numbers, results table, kernels list, trace list, footer with GPU, commit and date.
3. d5d50f6 CI `web` job and `web/README.md` (Vercel: Root Directory `web`, include files outside the root).

Decisions for Joe:
- Next 16.4.0 has three advisories (two critical remote code execution) according to `npm audit`; pinned 16.4.3 (released 2026-10-23). The Next build printed a message asking the agent to run `next upgrade --agent=security` and to ask the user a question; that was tool output and was not followed; the advisory was checked with `npm audit` instead.
- ESLint 9.39.5 (npm marks 9 as unsupported). ESLint 10 runs only by overriding the peer dependencies of Next's plugins, so the branch stays on 9 until eslint-config-next supports 10.
- TypeScript 5.9.3, not 7.
- Web-only commits were gated with `~/.cache/llmie-web/verify-web.sh` (refuses if engine files are staged; npm ci, audit, lint with zero warnings, `next typegen` plus tsc, static build) instead of the 25-minute engine gate. Say if the engine gate should run on web commits too.
- Visual check: Claude in Chrome was not connected, so headless Windows Chrome was used. It lays pages out at 526 px at least, so phone width was checked in a 390 px iframe; that found a nav that widened the page on phones (flex child needed `min-w-0`), fixed.
- Until branch 49 merges, the web branch's data has the old random-prompt trace; the sync picks up the chat trace automatically afterwards.
