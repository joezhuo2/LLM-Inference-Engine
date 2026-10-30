# Branch 52 `web/overview` (worktree ~/code/llm-inference-engine-web), 4 commits STACKED on web/prototypes (1faf8ab), not pushed

Rebase onto main after web/scaffold and web/prototypes merge. Each commit passed the web gate.

1. 093534d Cascade rain promoted to `TokenRain` as a full-screen hero on `/` (100svh minus the top bar, no border); other pages keep their column through the `(content)` route group; canvas/trace helpers moved to `src/lib/`; rain prototype deleted.
2. ae5a9dd Rain loops seamlessly: passes over the trace overlap by 0.4 s on one continuous clock, nothing is cleared (the old 1.5 s pause drained the screen, then reset). Canvas density measured steady across wraps.
3. e2782d0 `usePrefersReducedMotion` (matchMedia + useSyncExternalStore): motion's `useReducedMotion` stayed false after hydration with reduced motion emulated.
4. c24769c Build timeline below the hero: 25 entries, 45 of 59 PRs, M0 to M5, sticky milestone column, scroll-filled line, before/after bars; hover polish Joe asked for (cards and "how it was built" scale 1.03 over 320 ms, GitHub link underlines).

Open for Joe:
- Copy of every timeline entry is Claude's draft (`web/src/content/build-log.ts`); historical numbers are hand-copied from changelog/profiling.md at measurement time (only the llama.cpp bars read `results/`). Entries do not say who wrote what (Part A/B/C); say if they should.
- Tiles, story layout and charts pickers still wait on his picks; `web/AGENTS.md` still uncommitted.
- Verified via headless Windows Chrome over CDP (`~/.cache/llmie-web/cdp-*.mjs`, port 9333): dark, light, 390 px, reduced motion, hovers.
- 5th commit on branch 52: suppressHydrationWarning on body (Grammarly attributes). Reload loop Joe saw came from the gate running `npm ci` under his dev server; gate now skips npm ci when in sync.
- 6th commit 6ddd70f: intro loader (2.4 s, crypto-style hex line + progress line, modeled on waterloo-crypto-v01.vercel.app) then rain starts and hero title/numbers unscramble (1.6 s / 1.4 s). Plays on every full page load, any page; skipped with reduced motion.
