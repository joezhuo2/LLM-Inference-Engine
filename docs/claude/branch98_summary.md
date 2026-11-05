# Branch 98: web/sandbox-perf

Worktree ~/code/llm-inference-engine-perf, branch `web/sandbox-perf` (from origin/main 392c02d, origin/main 0c65225 merged in). Not pushed, no PR.

## Conclusion

The sandboxes were slow mainly because of NumberFlow. Every value of a glide or drag started a NumberFlow FLIP animation on every readout, and each one measured every digit with getBoundingClientRect between DOM writes. That forced layout over and over and caused 500 to 600 ms of style recalculation per second. Three things added to it. CSS transitions with 500 to 700 ms ease-in-out curves restarted on every frame, so the dot and bars trailed the slider by 0.5 to 0.7 s. Story figure loops kept running off screen (223 Web Animations on the paged attention page). The paged KV replay restarted motion width animations on 72 rects every frame. A slow-starting 600 ms ease-in-out curve on clicks made all of this feel worse.

After the fixes, every sandbox glide, drag, preset and reset has p95 frame time under 9 ms at 1x (target was 16.7 ms) with 0 to 1 dropped frames, apart from the sampling top-k drag (13 dropped frames at 120 Hz, p95 8.5 ms). At 4x the p95 falls from 350 to 717 ms to 17 to 117 ms.

## Commits (first parent, oldest first)

ee8c8a2 Share one glide helper for SmoothRange and every sandbox tween
9d8be8a Hold NumberFlow animation while a slider or sandbox tween glides
5beb134 Follow glides with a 150 ms ease-out instead of restarting long transitions
01a78c9 Show plain readout text during glides instead of NumberFlow
39a8921 Glide the roofline batch through fractional rows so the dot slides
db9f600 Create the reduced-motion media query list once
8aae0be Unmount sticky story figures while they are off screen
a8b2e8a Keep sandbox readout rows at a constant height while values glide
e805f75 Glide sliders, presets and resets with a 300 ms ease-out
b4d9972 Pick the GEMM sandbox tile with a Toggle instead of a slider
29d8060 Pick the chunked prefill budget with a Toggle instead of a slider
c1894e3 Draw the paged KV replay blocks without per-frame motion animations
fc9e9da Measure the Toggle highlight only when the selection or labels change
c66ec76 Slide the Toggle highlight with the 300 ms glide ease-out
60c656f Merge origin/main into web/sandbox-perf
9984403 Hold the chunked prefill readouts as text while the playground plays
66ef4be Compute the RoPE score curve on first render instead of at import

## Root causes, with evidence

1. NumberFlow on every intermediate value. Ablation test: stubbing Element.animate dropped the foundations glide style recalc from 520 to 110 ms/s and removed every long task (CSS transitions off: no change). In the CPU profiles getBoundingClientRect took 1.1 to 2.3 s per drag at 1x, plus NumberFlow `set data` and `didUpdate`. Turning NumberFlow animation off was not enough, because NumberFlow's `pop` still reads offsetTop/offsetLeft for every digit it removes. Fix: `web/src/lib/glide.ts` counts holds (glides, held slider pointers, chunked play). While any hold is active, `web/src/components/number-flow.tsx` renders plain Intl-formatted text in NumberFlow's own box, then mounts NumberFlow again when the hold ends.
2. Transitions restarting every frame. Sampled: the roofline dot settled 500 ms after the slider stopped and the bars 709 ms after. Fix: while `data-gliding` is set on the root element, `.bar-ease`, `.dot-move` and `.glide-snap` follow with a 150 ms ease-out. Now 142 ms and 150 ms. Discrete changes keep their long eases.
3. Roofline rows stepped on a log scale. At the low end one row step is a doubling, a 91 px jump. Fix: glides run through fractional steps, and the dot and bars use `rowsBetween` (geometric interpolation, with a unit test). Readouts still show whole rows. On a 1 to 512 jump at 1x: 17 to 76 moving frames, biggest single-frame jump 21% to 4%, worst frame gap 108 to 17 ms.
4. Off-screen story loops. Running at idle: 223 animations on the paged attention page, 21 on foundations. Now 0 (verified: the figure mounts when you scroll back).
5. Paged KV replay. About 59 steps per second while playing or dragging, each restarting motion width animations. Now plain SVG. Replay glide at 1x: 789 to 141 ms main-thread time per second.
6. Smaller fixes: matchMedia was called in every reduced-motion snapshot (10 to 20 ms/s at 4x). Toggle re-measured on every render when its choices array was inline. The RoPE curve was computed at import, on every kernel page (nav-speed finding).
7. Feel. The home scrubber's native smooth scroll measured as ease-in-out: t10 150 ms, t90 533 ms, end 958 ms on a 3,379 px jump. So what reads well there is that frames are never dropped, not a fast start. The old SmoothRange curve (600 ms ease-in-out) took 183 ms to reach 10%. With 300 ms ease-out: t10 9 ms, t50 42 ms, t90 109 ms, end 259 ms. The roofline dot uses the same timing for an operation change (`--duration-glide`), and so does the Toggle highlight (now 300 ms ease-out, was 500 ms ease-in-out).

## Measurements

Headless Windows Chrome 156 over CDP at 1280x900. rAF deltas, longtask and LoAF observers, Performance.getMetrics. vsync is 120 Hz (8.3 ms), so "dropped" counts missed 120 Hz frames. Before is out-base (392c02d). After is the build after c66ec76; the chunked rows were re-measured after 9984403. Logs: ~/.cache/llmie-perf/before.jsonl, after.jsonl, cp-play.jsonl, base.jsonl (first baseline). Script: ~/.cache/llmie-perf/cdp-sandbox.mjs.

| sandbox | action | 1x p95 ms (before / after) | 1x dropped | 4x p95 ms | 4x dropped | 4x worst long task ms |
|---|---|---|---|---|---|---|
| foundations | glide | 58.4 / 8.8 | 108 / 1 | 491.7 / 50.1 | 102 / 74 | 434 / 246 |
| foundations | drag | 58.4 / 8.8 | 273 / 0 | 716.7 / 41.7 | 2649 / 153 | 861 / 80 |
| foundations | preset 512 | 58.3 / 8.6 | 97 / 0 | 450.1 / 41.6 | 146 / 48 | 392 / 93 |
| foundations | reset | 58.6 / 8.8 | 104 / 0 | 591.7 / 25.1 | 103 / 44 | 536 / 89 |
| paged-attention | glide | 8.5 / 8.5 | 7 / 0 | 158.3 / 41.6 | 124 / 50 | 154 / 112 |
| paged-attention | drag | 8.4 / 8.4 | 3 / 0 | 100 / 33.4 | 439 / 122 | 142 / 57 |
| paged-attention | reset | 8.4 / 8.4 | 0 / 1 | 75 / 16.7 | 115 / 21 | 61 / 55 |
| paged-attention | play | 16.6 / 16.7 | 41 / 36 | 116.6 / 100.1 | 226 / 226 | 288 / 459 |
| sampling | glide T | 41.6 / 8.4 | 56 / 0 | 424.9 / 41.8 | 144 / 80 | 500 / 291 |
| sampling | glide top-k | 25.1 / 8.4 | 47 / 0 | 350 / 33.3 | 157 / 58 | 289 / 185 |
| sampling | drag T | 75 / 8.4 | 232 / 1 | 683.3 / 108.4 | 1958 / 329 | 963 / 192 |
| sampling | drag top-k | 75.1 / 8.5 | 383 / 13 | 775 / 116.6 | 3161 / 589 | 890 / 167 |
| sampling | reset | 83.3 / 8.4 | 105 / 0 | 508.3 / 8.5 | 85 / 46 | 421 / 117 |
| chunked-prefill | glide | 16.6 / 8.4 | 12 / 0 | 183.4 / 25.1 | 145 / 35 | 151 / 80 |
| chunked-prefill | drag | 8.4 / 8.4 | 1 / 0 | 166.7 / 25.1 | 729 / 103 | 224 / 89 |
| chunked-prefill | preset run it | 8.4 / 8.4 | 0 / 0 | 25.2 / 25 | 122 / 51 | 96 / 60 |
| chunked-prefill | play +2,048 | 16.7 / 8.4 | 56 / 0 | 383.3 / 16.6 | 311 / 83 | 691 / 162 |
| gemm | glide k loop | 8.6 / 8.4 | 0 / 0 | 41.7 / 50 | 58 / 65 | 0 / 180 |
| gemm | drag k loop | 8.4 / 8.4 | 1 / 0 | 58.4 / 58.4 | 190 / 228 | 58 / 95 |
| gemm | reset | 8.4 / 8.4 | 0 / 0 | 25 / 16.7 | 33 / 37 | 0 / 92 |
| gemm | play | 8.5 / 8.4 | 0 / 0 | 16.7 / 16.7 | 61 / 50 | 0 / 0 |
| rope | glide | 33.4 / 8.4 | 56 / 0 | 541.6 / 50 | 160 / 65 | 432 / 190 |
| rope | drag | 33.4 / 8.4 | 143 / 1 | 491.6 / 50.1 | 2251 / 214 | 536 / 116 |
| rope | drag query | 16.7 / 8.4 | 35 / 0 | 200 / 41.7 | 838 / 195 | 213 / 111 |
| rope | pair 20 button | 41.7 / 8.4 | 70 / 0 | 441.6 / 33.2 | 143 / 41 | 332 / 94 |
| rope | reset | 50.1 / 8.5 | 100 / 0 | 716.7 / 25 | 144 / 36 | 611 / 73 |
| paged-kv | glide replay | 8.8 / 8.8 | 4 / 0 | 66.7 / 33.3 | 123 / 39 | 52 / 0 |
| paged-kv | drag replay | 8.5 / 8.4 | 1 / 0 | 83.3 / 33.5 | 382 / 128 | 112 / 0 |
| paged-kv | visualizer + L1 | 8.5 / 8.4 | 0 / 0 | 33.4 / 25.1 | 191 / 149 | 0 / 0 |
| paged-kv | visualizer cache toggle | 8.4 / 8.4 | 0 / 0 | 41.7 / 25 | 96 / 57 | 0 / 0 |

The sampling T drag that Joe reported improved the most: p95 75 to 8.4 ms at 1x and 683 to 108 ms at 4x. T changes every bar and all 18 readouts at once, so it set off the most NumberFlow animations. The model itself costs very little: `stages()` runs over about 300 groups, not 32k tokens.

## Layout shift (every frame of a glide and of a drag; all rows, cells, separators, the section and the body)

Before (main): foundations rows had the measured table grow and shrink by 48.5 px when passing 1 or 16 rows; the context slider (attention) had 29 px; paged attention's block table changed height by 52 px; sampling top-k and top-p shifted everything below by 4 px when the value switched between "off" and a number. Everything else was still. After: no movement on any slider, for both glide and drag. Sliders covered: foundations rows and context, paged attention context, sampling T, top-k and top-p, chunked length, GEMM K loop, RoPE position, pair, query and key, paged KV replay, dashboard scrubber. Also checked at 1024 and 1536 px widths for a subset. Script: ~/.cache/llmie-perf/cdp-shift.mjs. Results: shift-base.txt and shift-cur.txt.

## Verified

- Every commit was gated on its own with ~/.cache/llmie-perf/verify-web.sh (audit, lint, tests, typecheck, build, "VERIFY WEB OK"). The merge commit used a copy that skips the "engine files staged" refusal (those engine files are origin/main's own).
- Screenshots in ~/.cache/llmie-perf/shots/: all seven sandbox pages in dark and light at desktop width, and foundations, GEMM, chunked prefill, sampling and paged attention in a 390 px iframe. No console errors, no horizontal overflow at 390 px (GEMM, chunked, sampling, paged attention).
- Chunked budget Toggle: picking 16 then reset restores 256; "run it" selects 256 and plays. Readouts are text while playing and NumberFlow once it settles.

## Not verified

- Reduced motion was not exercised in a browser this time. It was checked by reading the code: glides are skipped, and the `data-gliding` rule only changes duration, so it adds no movement.
- Keyboard use of the new Toggles: they reuse Toggle's radiogroup keys; not driven in a browser.
- Real hardware feel: all numbers come from headless Chrome on a busy shared machine, where other agents were running GPU and CPU jobs. Idle background cost varied between runs (the first baseline showed much higher idle cost than the final "before" run).
- No engine build or ctest for the merge (no engine files changed by this branch).

## Open questions for Joe

1. The feel changes (300 ms ease-out for glides, the dot on operation change, and the Toggle highlight) follow the brief and Emil's guidance. Your plan notes say you like slower motion, though. Each is one constant: `glideSeconds`/`glideEase` in glide.ts, `--duration-glide` in globals.css. c66ec76 (Toggle) can be dropped on its own.
2. Discrete NumberFlow updates are still expensive at 4x. A paged attention play step re-measures about 12 numbers (4x worst long task about 300 to 450 ms). Also, mounting NumberFlow again at the end of a glide costs a one-off long task at 4x (GEMM K loop glide: 180 ms, which was 0 before because that glide changed few numbers). Options: drop NumberFlow from dense tables, or wrap a sandbox's readouts in NumberFlowGroup. I did not do either.
3. Chunked budget choices are now 16, 64, 256, 512, 2048. The old 8, 32, 128 and 1024 stops did not fit at 390 px.
4. Reset in the chunked sandbox now also restores budget 256 (it used to keep the budget). "run it" now always starts playing; before, it stayed paused when no glide was needed.

## Draft PR description

Title: Make the sandboxes and sliders smooth

Measured first in headless Chrome at 1x and 4x CPU throttling (rAF frame times, long tasks, CPU profiles, ablations), then fixed one cause per commit:
- NumberFlow no longer animates intermediate glide or drag values. Readouts show formatted text in NumberFlow's box while a slider, preset, reset or the chunked playground is moving, then NumberFlow comes back.
- Transitions that follow a slider use a 150 ms ease-out while gliding, instead of restarting 500 to 700 ms ease-in-out curves every frame.
- One `glide()` helper (300 ms ease-out) for SmoothRange and every sandbox tween. The roofline dot and the Toggle highlight use the same timing.
- The roofline dot glides through fractional rows on the log axis.
- Story figures unmount while off screen. The paged KV replay draws without per-frame motion animations. The reduced-motion query is created once. Toggle measures only on a real change. The RoPE curve is computed lazily.
- Readout rows keep a constant height (no layout shift on any slider).
- GEMM tile and chunked budget are Toggles now.
Result: every sandbox glide, drag, preset and reset has p95 under 9 ms at 1x with 0 to 1 dropped frames (sampling top-k drag: 13 dropped at 120 Hz). At 4x, p95 drops from 350 to 717 ms to 17 to 117 ms. Full table in branch98_summary.md.
