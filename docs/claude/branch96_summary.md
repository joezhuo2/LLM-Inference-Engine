# Branch 96: web/sandbox-rope

Branch `web/sandbox-rope` (from origin/main d13ff65), worktree ~/code/llm-inference-engine-sb-rope, not pushed.

Commits:
- 1e7fa3f Add a TypeScript model of the engine's RoPE with a relative-position test
- 19acd62 Add a rotate-it-yourself sandbox to the RoPE page
- ffe75d4 Enlarge the RoPE sandbox vector plot and its labels

Files: web/src/components/kernels/rope/rotation.ts, sandbox.tsx; web/scripts/rope-rotation.test.mjs; wiring in web/src/content/kernels/types.ts (new `sandbox` field), rope.ts, web/src/app/(content)/kernels/[slug]/page.tsx (section after "how it works"), story-view.tsx (KernelSandboxView); changelog.md; docs/exercises/learning_log.md.

Verification: verify-web.sh gate passed alone on each commit (logs ~/.cache/llmie-sb-rope/gate*.log), 5 new node tests. CDP at 1280 and 390, dark and light, plus reduced motion at 1280 dark: sliders, dial buttons, lock toggle, predict buttons, shift buttons, reset, keyboard arrows on sliders, Enter on a dial; no console errors, no horizontal scroll. Screenshots in ~/.cache/llmie-sb-rope/shots/.

Not verified: NumberFlow values read as text (shadow DOM), checked visually only; real touch input; Safari/Firefox; the engine's bf16 rounding is not modelled (float64).

Open questions: sandbox placed after "how it works" via a new `sandbox` field rather than reusing `lead` (which sits above the story); other branches will conflict in page.tsx, types.ts and story-view.tsx.
- 15b5558 Tween the RoPE sandbox reset instead of jumping (reset tweens 600 ms ease-in-out; checked by sampling slider values and vector transform; reduced motion jumps)
