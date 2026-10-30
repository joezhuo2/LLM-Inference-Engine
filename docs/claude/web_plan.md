# Web site plan (Part 2)

Written 2026-10-25 from Joe's answers to the design questions. Built on Emil Kowalski's skills (github.com/emilkowalski/skills: emil-design-eng, pick-ui-library, prototype, animation-performance, break-ui). Claude builds everything in `web/`; Joe reviews and changes what he wants. Claude writes all copy; Joe rewrites freely.

## Decisions

| Topic | Decision |
| --- | --- |
| Personality | Restrained terminal: near-black, monospace UI, one green accent, matrix rain only in the hero |
| Color modes | Dark default, light toggle (next-themes, no flash on load) |
| Light mode | Paper printout: off-white paper, near-black ink, darker green accent |
| Typography | Mono for UI, headings, nav, numbers, labels, code (JetBrains Mono); sans for prose (Inter) |
| Accent | Muted phosphor green (around `#4ade80` dark, around `#15803d` light), used sparingly |
| Charts | Engine series in the accent, baselines (HF, llama.cpp) gray and dashed, extra series from a colorblind-safe palette |
| Hero | Token rain from a recorded trace: one column per sequence in the batch, glyphs are the decoded tokens it generated |
| Structure | Top bar: `/`, `/results`, `/kernels` (index plus one page per story), `/dashboard` |
| Motion | Rich: page transitions, scroll-driven diagrams in kernel stories, charts drawing in. Never on high-frequency actions (see Motion rules) |
| Dashboard | One-screen grid: metric tiles on top, streaming batch timeline in the middle, chat playground at the side |
| Stories in v1 | Paged attention v1 to v2 to split; CUDA Graphs; continuous batching and paged KV; fused add+RMSNorm tuning |
| Prototype first | Token rain hero, metric tiles, kernel story layout, results charts (3 variants each behind the picker, Joe chooses) |
| Stack | Next.js (App Router, `output: 'export'`) in `web/` of the engine repo, Vercel root directory `web/` |

## Constraints

- Vercel serves static files only here; no GPU, no backend. Every page must work without an engine running.
- Dashboard has two modes. Replay (default, public): plays a recorded trace JSON, clearly labelled as a recording. Connect: a URL field (default `http://localhost:8000`) switches to the live engine. Needs CORS for the Vercel origin and localhost on the engine; Chrome's Private Network Access may block or prompt for https to localhost, test early. Never expose the GPU publicly through a tunnel.
- Numbers come from `results/` at build time (a script converts the CSVs to JSON in `web/src/data/`), every chart labelled with GPU, commit and date. No hand-copied numbers.

## Libraries (from pick-ui-library)

TypeScript, Tailwind CSS with design tokens as CSS variables, `clsx` and `cva` for classes, `next-themes`, `recharts` (static charts), `Liveline` (streaming tok/s and timeline sparklines), `NumberFlow` (live and headline numbers), `motion` (page transitions, scroll-driven diagrams, layout animations), `base-ui` (popovers, tabs, dialog, select), `shiki` (code blocks in stories, at build time), `Satori` via Next's OG image route (social cards). Fonts through `next/font` (self-hosted, no layout shift).

## Design tokens

- Color: `--bg`, `--surface`, `--border` (semi-transparent, not solid), `--text`, `--text-muted`, `--accent`, `--accent-dim`, `--baseline`, `--warn`, plus the chart palette; dark and paper-light sets.
- Easing: `--ease-out: cubic-bezier(0.23, 1, 0.32, 1)`, `--ease-in-out: cubic-bezier(0.77, 0, 0.175, 1)`; never `ease-in` on UI.
- Duration: press 120ms, tooltip 150ms, popover and tabs 200ms, page transition 250 to 350ms, story diagrams may be longer.
- Radii small (2 to 6px, terminal feel), spacing on a 4px grid, text sizes from a fixed scale.

## Motion rules

- Rich places: page transitions (fade plus 8px rise, ease-out); kernel story diagrams driven by scroll (partials merging, blocks filling, launch gaps closing); charts drawing in once when scrolled into view; short stagger (40ms) on page-load lists; NumberFlow on changing numbers.
- No motion: nav and theme toggle clicks beyond press feedback, the variant picker, keyboard actions, dashboard tile layout, anything seen 100+ times a session.
- Every pressable element gets `scale(0.97)` on `:active`; hover effects only under `@media (hover: hover) and (pointer: fine)`; popovers scale from their trigger (`--transform-origin`), from 0.95 with opacity, never from 0.
- Animate only `transform` and `opacity` (and `clip-path` for reveals); list exact properties, never `transition: all`; full `transform` strings in motion, not `x`/`y`, so they stay off the main thread.
- `prefers-reduced-motion`: keep opacity and color fades, drop movement; token rain shows a still frame with a play button; scroll diagrams become stepped figures.

## Token rain (hero)

- Data: a recorded trace (per step: prefill or decode, sequence ids, sampled token text per sequence). Each column is one sequence; it appears when the request is admitted, its head glyph is the newest token (bright accent), the trail fades; a finished sequence stops and fades out; a preempted one flickers and restarts. A one-line legend explains it, with a link to the batching story.
- Rendering: one `<canvas>` with 2D context, `requestAnimationFrame`, device pixel ratio capped at 2, glyph atlas cached; paused when off screen (IntersectionObserver) or the tab is hidden; frame time budget under 4ms on the laptop; the playback speed tied to the trace's real step times, scaled.
- Text over it (title, one sentence, 3 to 4 headline numbers) sits on a scrim so it stays readable.

## Pages

1. `/` Overview: hero; headline numbers (batch-1 bandwidth share of peak, batch-16 throughput against batch 1, against HF and llama.cpp); architecture diagram (scheduler, block manager, paged runner, sampler); cards linking the four stories; reproduce steps and GitHub link.
2. `/results`: throughput against batch size (engine, engine with graphs, HF, llama.cpp); TPOT against context length (split attention); decode-step profile breakdown; roofline (branch 45); scoreboard of the MVP criteria; each chart with GPU, commit, date and the command to reproduce it.
3. `/kernels` and `/kernels/<story>`: one page per story with a scroll-driven diagram beside the text, before and after Nsight Compute numbers, the key code excerpt (shiki), and links to the source files.
4. `/dashboard`: metric tiles (tokens/s, running and waiting, KV blocks used of total, TTFT and TPOT p50/p99); batch timeline (one row per step, prefill and decode colored, width = step time); chat playground streaming tokens; replay controls (play, pause, speed, scrub) or the connect field.

## Branches (in order)

Engine side first, because the rain and replay need a trace format:

1. `docs/api-contract`: `docs/api.md` with the trace JSON format, the `/metrics` JSON, the SSE chunk format of `/v1/chat/completions`; `engine bench --trace out.json` writing the trace from `serve_step` (Part A/B). Records the first real trace into `results/traces/`.
2. `web/scaffold`: Next.js app in `web/`, tokens, fonts, theme toggle, top bar, page shells, results CSV to JSON script, Vercel config, CI job (lint, typecheck, build). Plain look, real routes.
3. `web/prototypes`: the four pickers under `/prototypes/<slug>` (rain, tiles, story layout, charts), 3 named variants each with a stated axis. Stop for Joe's choices; the prototype surface is deleted once promoted.
4. `web/overview`: the hero with the chosen rain, headline numbers, architecture diagram, story cards.
5. `web/results`: all charts from the data pipeline with the chosen chart style.
6. `web/kernel-stories`: the four story pages with their diagrams.
7. `web/dashboard-replay`: the grid in replay mode on the recorded trace.
8. `server/http` (engine, Part 2 item 1): OpenAI-compatible server with SSE, `/health`, `/metrics`, the trace stream, CORS.
9. `web/dashboard-live`: connect mode against the real server.
10. `web/polish`: break-ui pass (long text, empty and huge values, no engine), mobile pass, animation review (Before/After/Why table), OG images, Lighthouse and accessibility checks.

Each branch follows the usual rules (from origin/main, small one-line commits, changelog entries, blank learning-log terms, no code comments, never pushed).

## Verification per web branch

`npm run lint`, `npm run typecheck`, `npm run build` (static export) with no warnings; every page loads without console errors in a browser (screenshots of dark and light at desktop and phone width); reduced-motion checked; the rain and timeline profiled in Chrome DevTools (no dropped frames, CPU idle when off screen). Numbers on the site must match `docs/results.md`.

## Still open (ask when it comes up)

Site name and domain; whether to add privacy-friendly analytics; the exact copy of the hero sentence (Claude drafts, Joe edits).

## Decisions since the plan (Joe, 2026-10-25 and 26)

| Topic | Decision |
| --- | --- |
| Rain | Cascade variant, full screen under the top bar (no panel, no border), loops with no pause (passes overlap by 0.4 s). Prototype deleted. |
| Landing page | Below the hero, a scroll timeline of the build: milestones M0 practice to M5 shipped, sticky milestone column, entries with PR links and before/after bars. Cards to results, kernels and dashboard at the end. |
| Intro | A loading screen on every full page load, modeled on waterloo-crypto-v01.vercel.app: a status line with an accent caret stepping through engine start-up, a random hex line, a 1 px progress line, a 3-digit percent; 2.4 s plus a 0.6 s fade. Then the rain starts and the hero title and numbers unscramble (1.6 s and 1.4 s). Joe asked twice for these to be slower, not faster. |
| Titles | Page titles and headline numbers unscramble on load; kernel card names when scrolled into view; nothing else (rules in `kernels_plan.md`). |
| Hover | Cards and the hero's "how it was built" link grow to 1.03x over 320 ms (press stays 120 ms); text links underline on hover. |
| Toggles | Bars that change with a toggle ease with an ease-in-out curve over 0.7 s (the site ease-out read as a jump), numbers change with NumberFlow. |
| Story layout | Sticky, with each step's figure animating the kernel at work (see `kernels_plan.md`). |
| Navigation | Every newly opened page starts at the top (except back/forward and hash links). |
