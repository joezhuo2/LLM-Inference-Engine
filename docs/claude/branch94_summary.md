# Branch 94: web/sandbox-chunked-prefill

Worktree ~/code/llm-inference-engine-sb-chunked-prefill, cut from web/chunked-prefill-story (994ea52). Not pushed.

Commits: 07035f7 Port schedule() and schedule_chunked() to TypeScript with tests; 4cef0b2 Add the budget playground sandbox to the chunked prefill page. Each passed verify-web.sh alone (lint, 25 new node tests plus the existing ones, typecheck, build, no warnings).

Files: web/src/components/kernels/chunked-prefill/scheduler-model.ts (BlockManager counts, Scheduler.schedule and scheduleChunked, Run, Playground), sandbox.tsx, web/scripts/chunked-scheduler.test.mjs, wiring per ~/.cache/llmie-claude/sandbox-wiring.md (types.ts, content/kernels/chunked-prefill.ts, story-view.tsx, page.tsx), changelog.md, docs/exercises/learning_log.md (Branch 94, blank).

Verified over CDP (ports 8494/9494): 1280 and 390, dark and light, reduced motion at 390 light; drop 2,048 at budget 256 gives 9 steps to first token chunked, 1 step and a 1-step stall with a 2,048-token step under schedule(); tight pool at budget 32 gives 1 preemption and 575 steps; keyboard (ArrowLeft on the rule radiogroup and the budget slider); reset; no console errors; no horizontal overflow at 390; story steps render.

Not verified: hover states, motion sampling of the column transitions, screen reader output, the lg layout between 526 and 1024 px.
