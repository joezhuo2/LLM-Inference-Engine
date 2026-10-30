# Branch 63 `web/polish` (web worktree), from origin/main 1f5d259 (#69 merged), not pushed

Scope from web_plan item 10, applied to what exists (results and dashboard are still the plain shells from web/scaffold; their own branches come later, and polish for them should be rechecked then).

1. 646e083 Social cards: `/og/overview.png` and `/og/<story>.png` (1200 x 630), drawn at build time with `next/og` by a static route handler (`web/src/app/og/[card]/route.tsx`, card in `web/src/lib/og.tsx`). Same numbers as the hero (`headlineStats()` now in `lib/data.ts`), GPU and commit in the corner. JetBrains Mono from `@fontsource/jetbrains-mono` 5.3.0 (dev dependency, audit clean): Satori cannot read woff2. Not the `opengraph-image` file convention, because its static export has no file extension. `metadataBase` from `VERCEL_PROJECT_PRODUCTION_URL`.
2. 15a05e4 Layer map bars gray (`bg-baseline`), green only for the hovered story card (Joe's request mid-turn). Measured over CDP in both themes: rest #6b7280 / #9ca3af, hovering the RoPE card turns only row 03 accent.
3. b4dc610 Favicon (`icon.svg`, green `>_`) and a styled 404 (`not-found.tsx`, noindex, links to the four pages).
4. 54e4dff WCAG AA contrast in the paper theme: accent #15803d to #157438 (accent on the selected toggle's tint was 3.9 to 4.1), code excerpts `github-light-default` (github-light red 4.38, orange 3.34). Dark theme passed on all 12 pages.
5. 9911b74 Keyboard: skip link (first Tab stop), 2 px accent `:focus-visible` ring (inset on toggle choices), ARIA radio pattern on every toggle and the layer map batch switch (`web/src/lib/radio-keys.ts`: arrows wrap, Home/End, one Tab stop per group).
6. 1611c96 Footer at the bottom of the window on short pages (body flex column, main flex-1).
7. 6ef3b3c No benchmark CSVs: `/results` says so instead of an empty table; OG cards drop missing numbers; OG summary clamped to 4 lines.

Each commit gated alone (15a05e4 rebuilt in a detached worktree since the icon/404 files were untracked when it was gated).

## Accessibility pass (CDP, not Lighthouse)

Lighthouse via `npx lighthouse@13.5.0` was blocked by the session's permission classifier (it downloads and runs a package), so it has NOT been run. Instead, over CDP on the static build, all 12 pages in both themes: one h1 each, no skipped heading levels, `lang="en"`, no unnamed buttons/links/radios in Chrome's accessibility tree, no images without alt, and a contrast check of every visible text node (fixed in commit 4). Please run Lighthouse from DevTools on the deployed preview (Performance, Accessibility, Best Practices, SEO) and send me anything under 90.

## Break-ui

- Built with `results/` emptied (in a throwaway worktree): fails on purpose. The CUDA Graphs story needs the launch-gap lines, paged KV needs the chat trace, the stories need their profiling tables. That guard stays (no stale or invented numbers).
- Built with only the benchmark CSVs removed: builds; hero shows dashes; results and OG fixed in commit 7.
- 1920 / 390 / 320 widths, 24 px font, rapid toggling: covered in branch 62, no regressions in the new pages (no sideways scroll at 390 on overview, results, dashboard, 404).

## Animation review

| Where | Before | After | Why |
| --- | --- | --- | --- |
| Layer map bars | near-white (`bg-text`), green on card hover | gray (`bg-baseline`), green on card hover | Joe: the hovered bar should be the only one that stands out |
| Focus | Chrome default ring, clipped inside the toggle group's `overflow-hidden` | 2 px accent ring, inset on toggle choices | visible on both themes, not clipped |
| Toggles from the keyboard | Tab to every choice, arrows did nothing | arrows move the selection (bars still ease 0.7 s, no extra motion) | ARIA radio pattern; motion rules say no motion on keyboard actions beyond the data change |
| Token rain | `fillText` per glyph per frame | unchanged | measured 334 ms/s of script (about 5.5 ms per frame, plan budget 4 ms) in headless Chrome with `--disable-gpu`, no dropped frames, idle (5 ms/s) when scrolled away. A per-glyph canvas cache was tried and was slower there (`drawImage` 328 ms/s, 21 to 30 dropped frames per 4 s), so it was reverted. Software raster is not your Chrome: please record 5 s of the hero in DevTools Performance; if scripting is over 4 ms per frame there, the next try is one shared atlas canvas instead of one canvas per glyph |
| Paged KV block fills | animate SVG `width` over 0.35 s | unchanged | SVG attribute, repaint only, no HTML layout; 242 frames in 4 s during playback, p99 16.8 ms |
| GEMM / CUDA Graphs labels | Motion `x`/`y` on SVG groups | unchanged | a handful of groups, one 0.7 s transition per toggle, no dropped frames; full transform strings would matter for long-running animations, not these |
| Paged KV playhead | `motion.line` with `duration: 0` | unchanged | effectively a plain line; harmless, could be simplified |

Everything else matches the motion rules: ease-out / ease-in-out only, no `transition: all`, hover gated to fine pointers, reduced-motion stills verified in branch 62.

## Open for Joe

- Run Lighthouse (above) and profile the rain in your Chrome (above).
- Domain still open: OG URLs use Vercel's production URL automatically; if you add a custom domain, Vercel's variable follows it.
- Light accent moved slightly darker (#157438); the prototypes still hard-code #15803d (they go when tiles/charts are picked).
- From branch 62, still open: phone label scale for figures, 320 px nav scroll, `web/AGENTS.md` untracked (rewritten by `next dev`).
- Story toggles only exist while their step is on screen, so keyboard users reach them by scrolling to the step first; fine for now, worth knowing.
