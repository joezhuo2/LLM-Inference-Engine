# Branch 74 `web/results` (issue #92), summary

Worktree ~/code/llm-inference-engine-claude, from origin/main 6ba8e13. 4 local commits, unpushed, each gated alone (`~/.cache/llmie-results/verify-web.sh`; commits 2 and 3 gated in a temporary detached worktree with `cp -al` node_modules, then committed on the branch with `git diff <temp commit> HEAD` empty).

## Commits

| hash | message |
| --- | --- |
| 5eae8ce | Read the batch-16 launch gaps and the CUDA Graphs op table into the site data |
| f345d0a | Add the results chart components and build-time results data |
| 2f716a8 | Build the results page as a sticky story of measured charts |
| ad9c426 | Show the results page in the top bar, home cards and 404 page again |

Each has its changelog.md entry (new `## web/results` section, Part A) and blank learning-log terms under `## Results Page (Branch 74)`, except ad9c426, which adds no new concept (changelog only).

## What was built

- `/results` is a sticky story (same `StickyStory` as the kernel pages, phone layout unchanged) with 7 steps: the MVP scoreboard; tokens/s against max batch (engine + CUDA Graphs, engine, llama.cpp, HF); TPOT against max batch (same chart, the dots move to their new values and the lines redraw); TPOT against prompt length with split attention (64 to 1,984 tokens, 4 series, mean of the two recorded runs, both runs in the readout); the batch-1 decode step eager against CUDA Graphs; batch 1 against batch 16; the whole-step roofline. Below the story: the existing table of every run (now 1 decimal, matching docs/results.md) and the reproduce command.
- `StickyStory` got a `morph` prop: one figure stays mounted across steps instead of the crossfade per step. Inside, views of different type crossfade (0.35 s); within a view things move: dots translate with a 0.7 s ease-in-out, lines redraw left to right with an animated `clip-path: inset()` (1.1 s ease-in-out, 120 ms stagger), profile segments grow and slide with `translate + scaleX` (0.7 s), scoreboard gauges grow with `scaleX` (0.9 s), numbers in the stat strip change with NumberFlow. Only transform, opacity and clip-path are animated.
- Hand-built SVG charts (no chart library): crosshair that snaps to the nearest x on hover or tap, arrow keys / Home / End when the chart is focused (focus shows the first point), one readout box listing every series at that x with the exact value, and an `aria-live` copy for screen readers. Profile segments and scoreboard rows have the same per-mark readout. Charts measure their container (ResizeObserver) and draw at their real pixel width, so phone text stays at 11 px instead of shrinking with a 560-wide viewBox.
- Every step's caption: GPU, commit, date, source file(s), and the reproduce command (`scripts/reproduce.sh`; `scripts/profile.sh nsys decode_step --max-batch 1` / `decode_step_b16 --max-batch 16`; the `engine bench ... --prompt-len 1984 ... --graphs` command for the split attention runs). Links to kernel stories: paged-kv, cuda-graphs, split-attention, gemm (twice), attention-v2.
- Data (`web/src/content/results.ts`): every number at build time from results/ (CSV medians, decode_step*.md op tables and launch gap lines), docs/results.md (scoreboard, bench table, roofline tables) and docs/profiling.md (Branch 42/43 tables, occupancy table). The build fails if any CSV median differs from the docs/results.md table, or if the scoreboard sentences change shape. sync-data now also reads the batch-16 launch gap line and the graphs op table.
- OG card `/og/results.png`; `/results` back in the top bar, home card and 404 page. `/dashboard` untouched and still hidden.

## Verified

- Gate per commit: `VERIFY WEB OK` (lint 0 warnings, tests, typecheck, static build, 6 html pages) in ~/.cache/llmie-results/gate.log, gate-c2.log, gate-c3.log, gate-c4.log.
- Screenshots of every step (headless Chrome over CDP, port 9334, static server 8091): ~/.cache/llmie-results/shots/final_{dark,light}_{desktop,phone}_{top,0..6,end}.png, final_reduced_desktop_*.png, hover_1280_{dark,light}_{3,5,6}.png. 0 console errors or warnings on every run. (Phone shots scroll each step to the window center, so with the sticky panel on top some show the neighbouring step; step sync itself is checked by cdp-phone below.)
- Phone (390 px, mobile emulation): no page horizontal scroll, nav fits (233/233), step sync ok in dark and light (`cdp-phone.mjs` copy). Panel height 419 px for the chart steps, 465 to 478 for the profile and scoreboard steps (kernel pages: about 350).
- Crosshair values (`~/.cache/llmie-results/hover.mjs`, 65 checks, 0 failed): keyboard-stepped readouts at every batch for tokens/s (16 values) and TPOT (16) against the docs/results.md table, every roofline readout (TFLOP/s, GB/s, % of roof, 16 points) against the docs/results.md roofline table, every split attention readout (16) against the "run 1 / run 2" cells of docs/profiling.md Branches 42 and 43 and their mean, and a mouse hover near the HF batch-4 dot snapping to batch 4. Profile readouts print the raw cells of results/decode_step.md and decode_step_graphs.md (checked by eye, identical strings).
- Animation sampling over CDP (`sample.mjs`, `sample-load.mjs`, log in sample.log): line clip goes inset 100% to 0% between about 0.25 s and 1.4 s after the step activates, the dot fades in once the line reaches it; TPOT morph moves the batch-16 dot from y 35.9 to 182.6 over about 0.7 s ease-in-out while the new line redraws; profile morph grows row 2's attention segment from scaleX 0 to 74.3 and slides the idle segment; scoreboard gauges go 0 to 1 in sequence over about 1.1 s. With reduced motion: the same samples jump to the final values in the first frame (stepped stills).
- Palette check (dataviz validator) on accent, series-2, series-3, baseline: the gray baseline fails the chroma floor by design (plan: baselines gray and dashed), pink against gray is 4.1 protan ΔE in dark mode (context chart and profile segments; dash pattern, legend, inline labels and readouts carry identity), light-mode gray is 2.47:1 against the surface (the every-run table is the text alternative). The site's dark accent is outside the validator's lightness band; left as the site token.

## Not verified

- Joe's GPU Chrome (frame cost of the morphs, real touch behaviour of the tap readout). Headless Chrome only.
- Lighthouse.
- `/dashboard` and the other agents' branches were not touched or run.

## Open questions for Joe

- The split attention chart uses docs/profiling.md (Branches 42/43, older commits e298faf and 7770e78, 2026-10-23/24), not reproduce.sh; its commit and date line is written by hand in results.ts because those sections only say it in prose. OK, or should profiling.md sections get a machine-readable commit/date line?
- The graphs row of the profile is one "graph replay" segment (7,275.1 us), since Nsight Systems cannot split a replayed graph by op. Segment widths are mean GPU us per step; the idle segment is median span minus median busy, so a bar's total is close to but not exactly the median span (the label states the span).
- On phones the reproduce command scrolls sideways inside the caption instead of wrapping (wrapping made the panel 160 px taller on the long `engine bench` command).
- If results/ is empty the page now fails the build (like the kernel pages) rather than showing the old empty state.

## Draft PR description

**Results page (#92)**

Builds `/results` as a sticky story in the same layout as the kernel pages: the MVP scoreboard, throughput and time per token against max batch for the engine, the engine with CUDA Graphs, llama.cpp and HuggingFace, time per token against prompt length with split attention, the decode step profile (batch 1 eager vs graphs, batch 1 vs 16) and the whole-step roofline. One figure stays on screen and changes as the steps scroll past: dots move, lines draw in, profile segments grow and slide, numbers change with NumberFlow; reduced motion shows still steps.

The charts are hand-built SVG with a crosshair that snaps on hover, tap or arrow keys and lists the exact value of every series. Every number is read at build time from results/, docs/results.md and docs/profiling.md, and the build fails if the CSV medians and the docs/results.md table disagree. Each step names the GPU, commit, date, source and the command to reproduce it, and links the kernel story that explains its number. `/results` is back in the nav; `/dashboard` stays hidden.

- `StickyStory` gains a `morph` option (one figure mounted across steps).
- sync-data reads the batch-16 launch gap line and the CUDA Graphs op table.
- New social card `/og/results.png`.

Checked: web gate per commit, headless Chrome screenshots of every step (dark, light, desktop, 390 px phone, reduced motion), 0 console errors, 65 readout values against docs/results.md and docs/profiling.md, animations sampled over CDP.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
