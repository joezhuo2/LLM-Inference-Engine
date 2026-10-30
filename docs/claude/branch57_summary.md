# Branch 57 `web/kernel-swiglu` (web worktree), from origin/main a19dd6e (#63 merged), not pushed

1. 15f32a2 `/kernels/swiglu`: registry entry (headline 1.8x vs naive at 4,096 tokens; timeline #20 naive oracle, #35 spec, #42 Joe's 2D-grid draft + Claude's vectorization + flat grid, #43 small-batch dispatch, #46 wiring), 6 animated steps (gate/up zip with the silu curve, the rounding chain worked for g = 2, u = 2.25, Joe's draft and the 65,535 gridDim.y cap, 16-byte loads with per-kernel GB/s at 4,096 tokens vs the 323.6 peak, 26 SMs at one token 3 vs 22 busy with the dispatch rule, token toggle 1/16/64/512/4096), 2 measurement tables, testing, 3 excerpts. Gated by verify-web.sh.

Checked over CDP on the dev server: overflow script clean on all 5 SVG steps at 1280, every step at two or three moments, dark and light, reduced-motion stills, 390 px page, console clean.

Open for Joe:
- Copy of the 6 steps and the timeline (`web/src/components/kernels/swiglu/story.tsx`, `web/src/content/kernels/swiglu.ts`), in particular the description of his first draft ("already a little faster than naive at every size, and bit-identical") and the author split on #42 (draft "Joe", vectorization and flat grid "drafted with Claude", per the changelog's Part C tags).
- No SwiGLU bench script exists (profiling.md used a throwaway program), so both measurement commands are the ncu line with `<program> <tokens>`.
- Per-kernel GB/s in step 4 divides the vector kernel's DRAM bytes (the only bytes measured) by each kernel's time; at 4,096 tokens the vector kernel reads 332 GB/s, slightly above the 323.6 bandwidth-test peak, shown as measured.
- The g = 2, u = 2.25 example is computed in the browser (Math.exp + fround + BF16 round-to-nearest-even), not on the GPU.
- Hand-written numbers not from a profiling table: 323.6 GB/s peak (as in build-log.ts), 24 vs 3 memory instructions per 8 outputs, 704 vectors / 3 blocks / 22 blocks / 3,328 threshold (from the code and SM count), 28 tests (changelog).
