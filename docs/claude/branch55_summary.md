# Branch 55 `web/kernel-add-rmsnorm` (web worktree), from origin/main c3e86c3 (#61 merged), not pushed

1. abdcb20 `Frame`/`Label` and `Toggle` shared by every story (moved out of paged attention, no visible change); `useLoop().move()` slides an element by transform inside a loop window (destination shown with reduced motion).
5. 0aef771 hovering or focusing a `/kernels` card paints its bars in the layer map in the text color (store in `web/src/lib/kernel-hover.ts`).
4. 569af6c bars are pills (`EasedBar`, animated clip-path, min 0.5rem) in the layer map and both pickers; layer map bars show share of the step on one scale for both batches (the GEMMs take the same µs at batch 16, so the gray bars never moved before).
3. ea1907f layer map rows 01 and 07 get a muted second line ("from the previous MLP", "after attention"); Joe asked why add + RMSNorm appeared twice.
2. 22b485d `/kernels/add-rmsnorm`: registry entry (headline 3.0x vs the naive pair at 1 token, timeline #20 naive oracles, #35 spec, #39 Joe's fused kernel, #46 wiring, #49 tuning), 7 animated steps (the job, round trip through DRAM with moving boxes, first fused phases and barriers, BF16 rounding order and the two review fixes, 16-byte loads + registers, xor shuffle butterfly, token toggle 1/16/64 over measured times and DRAM bytes plus the current per-step time from results/decode_step.md), 3 measurement tables, testing, 3 excerpts.

Gate: commit 1 built alone in a temporary detached worktree, commit 2 by verify-web.sh. Checked over CDP on the dev server: no text overflow or overlap at 1280 and 390 px, every step at three moments of its loop, light theme, reduced motion stills, toggle eases over about 0.6 s, console clean, index card now links.

Open for Joe:
- Copy of the 7 steps and the timeline (`web/src/components/kernels/add-rmsnorm/story.tsx`, `web/src/content/kernels/add-rmsnorm.ts`), in particular the step 4 reading of his review fixes (3ce6d58 "sum of squares using unrounded sum", baf4a2b "round norm twice") and the #39 author tag "Joe" (only the clang-format fix was Claude's).
- Prose numbers that are not in a profiling table are hand-written: 177 to 73 us per step (timeline #49), 4.8 vs 4.1 us inside the step (#46), "64 rows of 4 KB under 2 us" (step 3). Everything else goes through `measured()`.
- Phase widths in step 3, the example values in steps 1 and 6, and the moving boxes in step 2 are illustrations and say so in the captions.
