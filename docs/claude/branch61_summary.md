# Branch 61 `web/kernel-paged-kv` (web worktree), from origin/main c3b23fb (#67 merged), not pushed

1. 14f603c Build-time replay of `results/traces/chat_b16_rate4.json` through a model of BlockManager (`web/scripts/kv-replay.mjs`, 4 tests, called by sync-data, `kvReplay()` in lib/data.ts): first fit from block 1, prompt blocks at prefill, a block per decode step that starts one, all back on finish. Every step's free count must equal the trace's `free_blocks` or the build fails (all 751 match); traces with preemptions are refused. Peak 66 of 12,915 blocks at step 607 (8 running), the run never goes past block 67. Gated alone.
2. 1c6e9e7 `/kernels/paged-kv`, the seventh and last story: 6 steps (contiguous reservation vs blocks for the first 4 real requests; request 0's real table [1,2,3,4,21,25,30,...] and slot 484 for position 100; write_kv into a [head,16,64] tile, 32.0 us / 0.4% from results/decode_step.md; append_slot; block 4 freed by request 1 at step 11 and taken by request 0 at step 16, plus preemption with the 56-block test; the whole run replayed at half speed with play/pause and a step slider). Timeline #24, #25, #29 (Joe + fixes), #31 (Joe + fixes), #28/#33, #34, #58/#59. Tables from docs/results.md. Gated.

Checked over CDP on the dev server: every step light and dark, reduced-motion stills, 390 px (no horizontal scroll), player advances 0.5x (0.24 s of trace per 0.48 s), overflow script clean apart from legend "box" notes, console clean.

Open for Joe:
- Copy of the steps and timeline, especially authorship tags ("both" on #29 and #31, "joe" on the scheduler entry that mentions Claude's stub).
- The player uses plain numbers instead of NumberFlow (values change every frame, the rolling digits never settled).
- The "reserving 2,048 each" comparison uses TinyLlama's full context; the bench itself sized max_context to 267 for this workload.
- The preemption sketch in step 5 is schematic (the trace has no preemptions); the numbers come from the test.
- `web/AGENTS.md` still untracked.
