# Branch 99: web/nav-speed (page navigation speed)

Worktree `~/code/llm-inference-engine-nav`, branch `web/nav-speed` from origin/main 392c02d. Not pushed, no PR.

## Conclusion

Navigation felt slow for three measured reasons, and each has a small fix (3 commits):

1. Story figures kept animating after you scrolled past them. Every sticky story figure is a set of infinite motion loops on SVG, which run on the main thread. They never paused off screen, so an idle story page used 514 ms/s of main thread at 1x and all of it at 4x. React renders a navigation in time slices, so it got the leftover time: a click on the story nav at the bottom waited about 2.4 s at 4x before anything happened. Fix: `StickyStory` renders its figure only while the panel is on screen.
2. Font subsets were downloaded when a page first appeared. `/kernels` uses `ᵀ` (JetBrains Mono latin-ext) and most stories use Greek letters (JetBrains Mono greek). Those faces were only fetched once the new page committed. React holds a view transition until fonts that start loading during the commit are ready (up to 500 ms, see `startViewTransition` in react-dom-client), which added 50 to 300 ms. Fix: preload those two subsets (about 24 KB).
3. Back and forward animated the scroll. `html { scroll-behavior: smooth }` made Chrome animate scroll restoration: about 1.5 s to fly back to the bottom of a long page. On the overview, the first restored pixel also started GlitchExit, which locked the scroll and left you at "built from scratch" instead of the cards. Fix: the popstate listener in ScrollToTop sets `scroll-behavior: auto` for two frames (not for hash links, so the hero link keeps its smooth scroll and glitch).

## Commits

- 523c34b Render sticky story figures only while their panel is on screen
- 264dce1 Preload the JetBrains Mono latin-ext and greek subsets
- f7e353f Restore the scroll position instantly on back and forward

Each commit has its changelog.md entry (Part A) and blank learning_log.md terms under "## Navigation Speed (Branch 99)". Each was gated alone, in order (`~/.cache/llmie-nav/verify-web.sh`: audit, lint, tests, typecheck, build: `VERIFY WEB OK` each time).

## Before and after (median of 3 runs, local static export, headless Chrome with GPU, 1280x800, cold HTTP cache)

"visible" is when the view transition is ready (old page starts leaving, new snapshot animating), or the first frame after commit when there is no transition. All times in ms from the click.

| route | 1x before | 1x after | 4x before | 4x after |
| --- | --- | --- | --- | --- |
| home -> results (card) | 78 | 51 | 946 | 601 |
| home -> kernels (card) | 100 | 72 | 794 | 671 |
| home -> dashboard (card) | 62 | 43 | 749 | 592 |
| /kernels -> paged-attention | 349 | 170 | 2025 | 1415 |
| /kernels -> foundations | 191 | 128 | 1618 | 1018 |
| story -> next story (story nav) | 132 | 55 | 2404 | 479 |
| results -> kernels (top bar) | 129 | 63 | 863 | 681 |
| back (React commit, all routes) | 24 to 83 | 17 to 45 | 289 to 895 | 99 to 578 |
| back to bottom of a long page (scroll restored) | about 1500 (smooth scroll), overview landed at 817 instead of 8018 after a 0.9 s glitch | 40 (1x), 400 to 550 (4x), right position | | |

Idle main thread at the bottom of /kernels/paged-attention: 514 ms/s (1x), 999 ms/s (4x) before; 0 after. The live site (current main) matched the local baseline. It also showed the Greek font arriving 50 to 100 ms after the click over a real network.

## Root causes ruled out, with evidence

- Prefetch: the cards and story links are prefetched in full before the click: the full `__PAGE__` RSC payload plus the route's JS chunks (CDP network log, no RSC fetch after the click). The loading.tsx skeletons never showed in any run. They only matter when someone clicks before the prefetch finishes, so I kept them.
- The view transition animation itself is 300 ms (unchanged, per the plan), and capturing the snapshots was cheap.
- Home cards use no view transition at all (no transitionTypes, no PageTransition on `/`), and GlitchExit and the intro loader do not delay forward navigation.
- Big JSON in client bundles: `IntroLoader` imports `@/lib/data`, so all data JSON (102 KB chunk) loads on every first page load. It is cached for later navigations, so it is not a navigation cost; noted below.
- Gating the overview's GpuPanel figure off screen made no difference outside noise in a 5-run A/B at 4x, so I dropped it.

## What still costs time (not fixed)

- /kernels -> first story at 1x is still 128 to 178 ms, above the 100 ms target. Breakdown at 1x: about 85 ms to render (about 40 ms of it is evaluating every story's module on the first story visit, because story-view.tsx statically imports all stories and Turbopack merges them into one 350 KB chunk), then about 75 ms inside the view transition's update callback, mostly React's forced layout of the whole page. Second visits (story -> story) are 55 ms.
- The single biggest module-level cost is `rope/sandbox.tsx`: it computes a 4,097-point curve at import, on every story page (about 15 ms at 1x, 60 ms at 4x). That file is off limits on this branch (the sandbox-perf agent owns it): computing the curve lazily inside the component would remove it.
- Removing `"use client"` from story-view.tsx does not split chunks: client references are collected per route, and all slugs share `/kernels/[slug]` (measured, reverted). Real splitting would need `next/dynamic` per story inside the client switch. Those chunks would not be prefetched, so a network wait and an empty-then-pop on the first visit would replace the evaluation cost. Not done.
- Lazy-mounting below-the-fold sections or `content-visibility: auto` would cut the forced layout, but either one changes page height before those sections render, which breaks back-navigation scroll restoration on story pages. Not done.
- 4x home cards (0.6 to 0.7 s) are spent on the new page's own render, Next's forced layout in layout-router scroll handling, NumberFlow and motion setup. No single cause is left.
- dashboard -> anywhere at 4x: about 2 s (97 ms at 1x). The replay loop uses 965 ms/s of main thread even at 1x (style 427, layout 126, script 165 ms/s), so the navigation render starves. Outside the requested routes, and the replay is the feature, so I left it.

## Not verified

- Real browsers: Firefox and Safari (view transitions differ), and a real phone. Measurements are headless Windows Chrome 156 with GPU compositing (AMD 890M) and CDP CPU throttling. Software compositing (`--disable-gpu`) showed the same pattern.
- The live Vercel site after these commits (not deployed). Local numbers are without network latency. On a slow network a click before the prefetch completes would show the skeletons.
- Screen-reader behavior of the story figure now being absent from the DOM while its panel is off screen (the step text is unchanged).
- Rope, gemm and dead-ends still fetch Inter greek or latin-ext (19 KB, 85 KB) on their first visit, so that view transition can still wait for them (up to 500 ms).

## Open questions for Joe

1. Story figures: on re-entry the figure now fades in from empty instead of crossfading from the paused old frame (it was already remounted with the same 300 ms fade). OK?
2. Do you want the first-story-visit cost attacked further? Options: (a) ask the sandbox-perf branch to make the rope curve lazy (smallest, about 15 ms at 1x); (b) per-story `next/dynamic` (saves about 40 ms of evaluation, risks an empty-then-pop on slow networks); (c) defer below-the-fold sections (saves most of the layout, breaks back-scroll restoration unless heights are reserved).
3. Preload Inter latin-ext (85 KB) and greek (19 KB) too, so rope, gemm and dead-ends never wait for fonts, or leave them on demand?
4. GlitchExit still fires if back navigation restores a position inside the first screen of the overview (an instant restore of less than one viewport). Fix with a short settle guard after mount?
5. Dashboard: pause or throttle the replay when the tab is navigating away (it saturates the main thread at 4x)?

## How it was measured (scripts in ~/.cache/llmie-nav/)

- `nav-bench.mjs <base> <cpu> <scenarios> <runs>`: real mouse click over CDP after scrolling the link into view and waiting for prefetch. It records click, URL change, skeleton, new h1 commit, the first frame after commit, startViewTransition / updateCallbackDone / ready / finished (wrapped), long tasks, long animation frames with script attribution, and network with sizes. Probe frames start only after commit, because a rAF loop from the start would force frames and inflate the old page's cost. `PROFILE=` saves click-aligned CPU profiles, `TRACE=` saves Chrome traces.
- `prof.mjs`, `attr.mjs` (source-mapped attribution from a temporary `productionBrowserSourceMaps` build, never committed), `trace-tasks.mjs`, `idle.mjs` (idle main-thread cost and running animations by section), `layout-cost.mjs`, `fonts.mjs`, `scroll-back2.mjs` (scroll trail and glitch overlay after back), `hero-link.mjs`, `check-figure.mjs`, `table.mjs`.
- Raw results: `b-1x.json` / `b-4x.json` (baseline), `f-1x.json` / `f-4x.json` (after), `head-1x.json` (final HEAD), `live-1x.json` / `live-4x.json` (Vercel, main).
- Visual checks (`shots/`): paged-attention dark desktop, foundations light desktop, add-rmsnorm at 390 px, paged-attention with reduced motion. The figure is present on load when in view, removed when scrolled past (0 running animations), back with its 300 ms fade, panel height unchanged, no fade under reduced motion, no console errors. Hero "how it was built" still smooth-scrolls with the glitch and lands at 817, same as on the live site.

## Draft PR description

Title: Make page navigation start faster

Navigation felt slow in three measured ways; each fix is small.

- Sticky story figures now render only while their panel is on screen. Their infinite SVG loops kept running after you scrolled past, so a story page used half the main thread at idle (all of it at 4x CPU), and the click on the story nav at the bottom waited up to 2.4 s at 4x. Story -> story is now 55 ms at 1x (was 132) and 0.48 s at 4x (was 2.4 s).
- Preload the JetBrains Mono latin-ext and greek subsets (24 KB). `/kernels` and most stories use characters from them, and React holds the view transition until fonts that start loading during the commit are ready. Results -> kernels is now 63 ms (was 129).
- Back and forward restore the scroll position instantly. `scroll-behavior: smooth` made Chrome animate restoration for about 1.5 s on long pages, and on the overview it set off the glitch exit and landed at the wrong section. Hash links keep the smooth scroll.

Measured on the static export in headless Chrome at 1x and 4x CPU (before/after table in the branch summary). No visual change: view transitions, skeletons and reduced motion are unchanged.
