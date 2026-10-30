# Branch 56 `web/kernel-rope` (web worktree), from origin/main e379ad1 (#62 merged), not pushed

1. 561c913 timeline entries keyed by branch, PR and title (three RoPE entries share #40; React warned about duplicate keys). Committed before its gate by mistake, then built alone in a temporary detached worktree: VERIFY WEB OK.
2. 94d0441 `/kernels/rope`: registry entry (headline 1.4x vs naive at one token; timeline #20 naive oracle, #35 spec, #40 Joe's draft + Claude's host-frequency fix + Claude's parallel mapping, #46 wiring), 6 animated steps (dials turning at the real rates, QKV layout and rotate_half pairs with the 8 roundings, the last-bit powf chain, the serial draft, 32 pairs x 8 head groups, token toggle 1/16/64/512 with the current decode-step share), 2 measurement tables, testing, 2 excerpts. Gated by verify-web.sh.

Checked over CDP on the dev server: no text past the figure edge at 1280 (remaining overflow-script hits are false positives from translated groups and the row-highlight rects), every step at two or three moments of its loop, dark and light, reduced motion stills (step 5 shows the first head pass), 390 px page, console clean, index card links. Dial rotation measured: pair 0 about 57 deg/s (1 rad/s), pair 1 about 41 deg/s, same at 390 px.

Open for Joe:
- Copy of the 6 steps and the timeline (`web/src/components/kernels/rope/story.tsx`, `web/src/content/kernels/rope.ts`), in particular the description of his first draft and the #40 author split (draft "Joe", the three fixes "drafted with Claude").
- No RoPE bench script exists (profiling.md used a throwaway program), so the measurement's command is the ncu line from profiling.md with `<program>`. A `scripts/rope_bench.cu` like `add_rmsnorm_bench.cu` would be a small Part A engine branch if he wants it.
- docs/profiling.md "Fast RoPE" prose says the first draft had "each thread walked all 36 heads one after another"; the code at 30960dd and the table header both say one thread per head looping over 32 pairs. The page follows the code; the profiling prose should be corrected (engine doc, not touched here).
- Hand-written numbers not from a profiling table: 96.7 to 56.1 us per step (#46, from profiling.md prose), 57 of 1.4 million outputs and 2 of 32 frequencies (changelog), the 1,152 vs 256 sin/cos counts (from the code). The dial rates are computed from theta.
- Same phone wrap as here exists on the add-rmsnorm picker's last line ("in a batch-1 decode step now ... over 44 launches"); left for web/kernels-polish.
