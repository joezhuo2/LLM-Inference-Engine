# Branch 93: web/sandbox-sampling

Worktree ~/code/llm-inference-engine-sb-sampling, from origin/main d13ff65, not pushed.

Commits: Add TinyLlama next-token logits and a TypeScript model of the sampler; Add the sampling sandbox: knobs, stage bars, seeded draws.

Verified: web gate (audit, lint, tests incl. sampler.test.mjs 7/7, typecheck, build) passed on each commit; CDP at 1280 and 390, dark and light, reduced motion: no console errors, readouts change correctly (capital, top-p 0.9: 1 candidate at T 0.7, 167 at T 1.5), keyboard arrows on sliders, greedy at T 0, reset. Shots in ~/.cache/llmie-sb-sampling/shots.

Not verified: the gate copy excludes scripts/*.py and results/*.json from the "engine files staged" check (no C++ touched); hover/active motion measured only by CSS reuse; NumberFlow mid-animation values.

Open questions: the sandbox shows 12 tokens plus an "others" bar; the logits are CPU BF16 HF (not the engine's GPU logits).
