# Branches 53 `web/kernels-data` and 54 `web/kernels-template` (web worktree), stacked on web/overview (merged as #60, so a PR against main shows only these 6 commits), not pushed

## 53 web/kernels-data (2 commits)
1. 75b8959 `docs/profiling.md` parser (`web/scripts/profiling.mjs`, 6 `node --test` tests, `npm test` added to CI and the local gate) -> `src/data/profiling.json`; `profilingTable`/`cellNumber`; pins shiki 4.5.0.
2. fe6e485 `loadExcerpt` (code from `src/` at build time between start/end lines found by text, shiki GitHub light/dark as CSS vars, GitHub line links); kernel registry `web/src/content/kernels/` with `measured(section, row match, column)` lookups; paged attention timeline (#26 to #53) with author tags.

## 54 web/kernels-template (4 commits)
1. 4897392 `PageTitle` (unscramble, 1.6 s) on all page titles, `Scramble trigger="view"`, `ScrollToTop` (fixes Joe's half-hidden title after client navigation; skips back/forward and hashes).
2. f9e8353 `/kernels` index (layer map from results/decode_step*.md, batch 1/16 toggle with 0.7 s eased bars + NumberFlow, cards with `soon`), `/kernels/[slug]` template (header, sticky story, timeline, profiling tables + commands, testing, excerpts, sources), paged attention story: 9 animated SVG steps + interactive split picker using the ported `attention_splits` rule.
3. f724d0d bars ease in-out over 0.7 s (Joe: the toggle still looked instant).
4. 2d01c42 first figure animates without a detour (`AnimatePresence initial={false}` had frozen its loops); the figure restarts each time the panel scrolls into view.

Gate: each commit built alone in a temporary worktree (hard-linked node_modules; a symlink breaks Turbopack). Checked over CDP on the dev server: text overflow/overlap script over all steps, active step matches visible text, bar origins, phone (sticky figure above text) and desktop, no hydration warnings.

Open for Joe:
- Copy of all nine steps and the timeline (`web/src/components/kernels/attention/story.tsx`, `web/src/content/kernels/paged-attention.ts`).
- Illustrations that are not measurements are labelled as such (SM dots, example scores and partials).
- Figure text is small on phones (SVG scales to width); a phone-specific figure layout would be a polish-branch item.
- The measurements section shows tables, not charts, until the charts prototype pick.
