# Branch 58 `web/kernel-gemm` (web worktree), from origin/main 7e8df6b (#64 merged), not pushed

1. 3ea813d Read docs/results.md at build time: sync-data writes `web/src/data/results_doc.json` (sections parsed like profiling.md, plus `ceilings` from the Roofline "Ceilings:" line); `measured`, `profilingTable` and `KernelMeasurement` take an optional `doc` ("profiling" default, or "results"); `ceilings()` throws if the line is missing; the measurements section names the docs it copied from. Gated alone in a temporary worktree.
2. cb6ddcc `/kernels/gemm`: registry (headline 91–102% of the roof at batch 1; timeline #8 wrapper, #12 fused wqkv/wgu, #22 vs Hugging Face, #37 first profile, #44 cuBLAS in graphs tagged Joe, #55 roofline), 6 animated steps (weights per step beside GPU time per step; row-major trick on a 3 x 4 W; GEMV streaming rows with the 6.39 ms floor vs 7.81 ms measured; GB/s per op vs 323.6 with attention at 2%; 16 rows reusing each weight row; log-log roofline with a 1/16 rows toggle), 2 tables from results.md Roofline, testing, 3 excerpts (cublasGemmEx call, Blas workspace, DecodeGraphs eager run before capture). Gated by verify-web.sh.

Checked over CDP on the dev server (port 3000, this worktree): every step at two moments, dark and light, reduced-motion stills, toggle point eases 0.7 s (sampled every 100 ms), 390 px no horizontal scroll, console clean. Overflow script: one remaining "box" note on step 5's "one row of W from DRAM" label, visually fine.

Open for Joe:
- Branch scope: the plan had GEMM + CUDA Graphs together; he asked for GEMM/GEMV only, so `web/kernel-cuda-graphs` is next.
- Copy of the 6 steps and the timeline (`web/src/components/kernels/gemm/story.tsx`, `web/src/content/kernels/gemm.ts`), in particular the #44 entry tagged "Joe" (DecodeGraphs is his Part C; the eager run before capture is his code) and the #37 entry's wording ("cuBLAS runs each projection as a gemv kernel").
- Measurement commands: the first is the roofline.py line as reproduce.sh runs it with the paths filled in (context 96 = 64 + 64/2), the second is just `scripts/reproduce.sh`.
- Hand-written numbers not from a doc table: weight shapes (2,560 / 2,048 / 11,264 / 5,632 / 32,000, from config.json), 89 cuBLAS calls per step, 4 KB for x, the test shape list (from test_gemm.cpp).
- `web/AGENTS.md` is still untracked and uncommitted.
