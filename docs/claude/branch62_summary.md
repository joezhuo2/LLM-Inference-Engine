# Branch 62 `web/kernels-polish` (web worktree), from origin/main d058be3 (#68 merged), not pushed

1. b6c26c3 Active story step is the one whose text sits nearest the middle of what can be read: below the sticky figure on phones, the whole window on desktop (rAF-throttled scroll handler instead of an IntersectionObserver band). At 390 px the old band was hidden behind the figure, so the figure lagged one step (7 to 21 out-of-sync scroll positions per page, now 0 on all seven). Phone steps min 55svh instead of 70svh (blank gaps).
2. 2a22521 Top bar fits on phones: "overview" hidden below 640 px (the llmie mark links home), tighter gaps, 13 px links. Links needed 308 px in 214 px at 390 px, "dashboard" was hidden in a sideways scroll on every page; now fits down to 360 px (320 px still scrolls).
3. c53aa8b Story prototype deleted (`/prototypes/story`, `web/src/prototypes/story/`). Tiles and charts prototypes stay until picked.
4. c15fb55 Reduced-motion stills pass: all 46 steps shot twice 2.5 s apart (identical, so nothing moves) and reviewed; only the GEMV "one row at a time" step was wrong (weight rows hidden, empty matrix). Now shows every row read.
5. 0546863 Figure box grows to fit its content (figures stacked in one grid cell instead of absolute, so aspect-ratio is a minimum) and is capped at 28 rem below 768 px. Fixes HTML control steps spilling at 320 px and at a 24 px browser font; panel at 640 px (1280 laptop at 200% zoom) 411 of 800 px instead of 506.
6. 49ece36 Toggle choices wrap under the label when they do not fit (24 px font).

Each commit gated alone in a detached worktree. Checked over CDP on the dev server (port 3100): all 11 pages at 320, 390, 640, 1920 px and at a 24 px root font in light mode (no page-level sideways scroll; data tables scroll inside their own box), step sync, panel spill, console clean; every toggle clicked 30 times at 40 ms and the paged-kv slider/play hammered (one choice checked per group, no NaN, no errors).

Break-ui, what held up: page-level horizontal scroll (none anywhere), wide tables (own scroll box), 1920 px layout, rapid toggling, reduced motion.

Open for Joe:
- Figure labels on phones are small: 11 px SVG text in a 560-wide viewBox renders at about 6.5 px at 390 px. A larger phone label scale would need each figure re-laid out; proposed as its own branch if wanted.
- Toggles are `role="radiogroup"` buttons: Tab reaches each button, but arrow keys do not move the selection as the ARIA radio pattern expects. Recommend adding arrow keys + roving tabindex in `toggle.tsx` (small, one file).
- At 320 px the nav still scrolls sideways (203 px of links in 163 px). Rare width; left as is.
- After pulling, a running `next dev` may fail typecheck on stale `.next/dev/types` for the deleted `/prototypes/story`; delete `web/.next/dev` (or restart dev) to fix.
- `web/AGENTS.md` still untracked.
