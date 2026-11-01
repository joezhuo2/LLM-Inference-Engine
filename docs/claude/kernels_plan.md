# Kernels section plan

Written 2026-10-25 after Joe asked to expand `/kernels`: every kernel gets a scroll-through with a visual model of how it works, and a timeline of its changes and key features. Builds on `web_plan.md` (tokens, motion rules, libraries) and the story-layout prototype of branch 51 (Joe picked Sticky). His answers to the design questions are under Decisions.

## Pages

| Slug | Kernel(s) | Source | Core idea to show | Headline numbers (from docs/profiling.md, docs/results.md) |
| --- | --- | --- | --- | --- |
| `paged-attention` | v1, v2, split + merge | `paged_attention.cu`, `paged_attention_v2.cu`, `paged_attention_split.cu` | block table lookup into the KV pool; v1 scores in shared memory (context cap); v2 online softmax with a running max; split: one context cut across blocks, partials merged in order | 883 to 272 us per decode step (v1 to v2); 14.0 to 9.1 ms per token at a 1,984-token prompt with graphs (split) |
| `add-rmsnorm` | fused residual add + RMSNorm, tuned | `add_rmsnorm.cu` | two kernels and a DRAM round trip become one; sum of squares by warp shuffles; 128-bit loads, row kept in registers | 2/3 of the DRAM bytes; 177 to 73 us per step after tuning; launches 267 to 224 per step (with rope, swiglu wiring) |
| `rope` | fast RoPE | `rope.cu`, `rope_fast.cu` | each pair of dims rotates by an angle that depends on position; thread mapping (first draft serial over heads, final over pairs and head groups); host `inv_freq` table for bit identity | 9.0 to 6.5 us at 1 token, 36.1 to 17.9 us at 512; 96.7 to 56.1 us per step |
| `swiglu` | vectorized SwiGLU + small-batch dispatch | `swiglu.cu`, `swiglu_fast.cu` | gate and up halves; 8 bf16 lanes per 16-byte load; flat 1D grid (65,535 row cap); scalar path for small launches | 650 to 358 us at 4,096 tokens (about 330 GB/s, the measured peak) |
| `gemm` | cuBLAS GEMM/GEMV (not hand-written, said plainly) | `gemm.cpp`, `scripts/gemm_peak.cu` | row-major trick (ask cuBLAS for the transposed product); batch 1 is a GEMV that streams every weight once; roofline point moving as the batch grows | 91 to 102% of the 323.6 GB/s roof at batch 1; 34.5 TFLOP/s compute roof |
| `cuda-graphs` | DecodeGraphs (runtime, not a kernel) | `decode_graphs.cpp` | launch gaps on an Nsight-style timeline closing up; one graph per (batch bucket, split count) | GPU idle 11.3% to 0.9%; 9.49 to 8.51 ms per token |
| `paged-kv` | write_kv + block manager | `write_kv.cu`, `block_manager.cpp` | slot = block table[pos / 16] * 16 + pos % 16; blocks taken and freed as requests come and go (driven by the recorded trace) | 16 concurrent = 16 solo; peak KV blocks from the trace |

The naive oracles (embedding, add, rmsnorm, rope, swiglu, attention) appear inside each page as "v0", never as pages of their own.

## Index page `/kernels`

- A decode-step map: one TinyLlama layer drawn as the sequence of kernels it launches (rmsnorm, QKV GEMM, rope, write_kv, attention, O GEMM, add+rmsnorm, gate/up GEMM, swiglu, down GEMM), the width of each box its share of the batch-1 step time from `results/decode_step*.md`, toggle for batch 16. Each box links to its page.
- Below it, one card per page: name, one-line idea, the headline before/after, the number of versions.

## Page template (every kernel page)

1. Header: name, what it computes as one formula, where it sits in the step (mini map highlighting it), the current version's headline number.
2. Scroll-through: a figure that stays on screen while 4 to 6 short text steps scroll past; each step changes the figure (see "Visual models"). Layout follows the story prototype Joe picks.
3. Timeline: every version from the naive oracle to the current kernel, one entry per change (date, branch, PR links, what changed, why, the number it moved), same component family as the landing page timeline, with version tags (v0 naive, draft, fast, tuned). Includes dead ends that taught something (e.g. RoPE first draft slower than naive; `__powf` breaking bit identity; fused add+norm not bit-identical).
4. Measurements: the Nsight Compute table from `docs/profiling.md` as a chart (naive vs versions across token counts, results chart style from the charts pick), with GPU, commit, date and the `ncu` command.
5. Correctness: how it is tested (bit identity with the oracle, tolerance, HF check, number of tests, sanitizer).
6. Code: the key excerpt (shiki at build time) and links to the source file and its tests on GitHub.

## Visual models (scroll steps)

- Paged attention: KV pool grid with a request's scattered blocks and its block table arrow-linking them; v1 a score row filling shared memory until the cap; v2 a running max line and rescaled accumulator updating as keys stream past; split: the context cut into colored splits processed side by side, then merged (m, l, acc) in order; graphs: one captured graph per split count.
- Add + RMSNorm: two kernels with arrows to and from DRAM vs one kernel with the round trip removed (bytes counter); a warp of 32 lanes doing the shuffle reduction tree; 16-byte loads lighting 8 values at once.
- RoPE: 2D vectors (one per dim pair) rotating by position-dependent angles, faster for low dims; thread grid of the first draft (serial) vs final (parallel) with a time bar.
- SwiGLU: gate and up halves zipped into one output; scalar lanes vs 8-wide vector lanes; a small batch leaving most SMs idle (26 SM tiles) and the fallback filling them.
- GEMM: matrix-vector product streaming weight rows from DRAM (bytes-per-FLOP gauge); roofline with the point sliding from bandwidth bound toward compute as the batch grows; row-major vs column-major transpose trick.
- CUDA Graphs: a kernel timeline with CPU launch gaps between bars that close when the graph replays.
- Paged KV: blocks allocated and freed live from the recorded trace (same data as the rain).
- All figures: SVG plus `motion`, driven by scroll progress or step index, `transform`/`opacity` only; reduced motion shows each step as a still figure; numbers in figures come from the data files.

## Data

- Extend `web/scripts/sync-data.mjs` to parse `docs/profiling.md` sections (by heading) into `web/src/data/profiling.json` (tables as rows, plus GPU and date lines), so charts and figure numbers are never hand-copied. Sections that are prose only stay prose.
- Kernel registry `web/src/content/kernels/<slug>.ts`: title, formula, source paths, profiling section ids, timeline entries (text written by Claude, Joe edits), excerpt line ranges. Timeline numbers point at a profiling section where one exists; the rest are marked as historical.
- Code excerpts read at build time from `src/` by path and line range (or marker-free function name lookup), so they never drift from the code.

## Branches (in order)

1. `web/kernels-data`: profiling.md parser + tests, kernel registry types, excerpt loader, shiki at build time. DONE (branch 53).
2. `web/kernels-template`: page template, index decode-step map, the paged attention page end to end. Stop for Joe's review of look and depth. DONE (branch 54, 4 commits); Joe's fixes so far: smooth batch toggle, scroll to top on navigation, label overflow in step 2, first figure not animating.
3. `web/kernel-add-rmsnorm` (branch 55): the add+RMSNorm page alone, at Joe's request. DONE (2 commits). `web/kernel-rope` (branch 56): the RoPE page alone, at Joe's request. DONE (2 commits). `web/kernel-swiglu` (branch 57). DONE (1 commit).
4. `web/kernel-gemm` (branch 58): the GEMM page alone, at Joe's request. DONE (2 commits; also parses docs/results.md, `doc: "results"` on measurements). `web/kernel-cuda-graphs` (branch 59). DONE (2 commits; real-trace JSON from `web/scripts/step-timeline.py`).
5. `web/kernel-paged-kv` (branch 61). DONE (2 commits; build-time trace replay checked against every step's free count).
6. `web/kernels-polish`: reduced motion stills, phone layout (figure above text instead of sticky beside it), break-ui pass, delete the story prototype.

Each branch: from the latest base, small one-line commits, changelog, blank learning-log terms, web gate, browser check (dark, light, 390 px, reduced motion) over CDP.

## Decisions (Joe, 2026-10-25)

| Topic | Decision |
| --- | --- |
| Pages | The core six (paged attention, add+RMSNorm, RoPE, SwiGLU, GEMM, CUDA Graphs) plus `paged-kv` (write_kv + block manager, driven by the recorded trace). Sampling is out. |
| Layout | Sticky: the figure stays pinned beside the text and changes as the steps scroll past. Each step also plays an animation of the kernel at work (threads loading, reductions, softmax updating, blocks merging), not only a static change between steps. On phones the figure sits above the text. The story prototype is deleted once this lands. |
| Interaction | Scroll drives the steps; the last step of each page adds one or two controls (token count, batch, split count) over the measured numbers. |
| Authorship | Every timeline entry is tagged: "Joe" on his Part C implementations, "drafted with Claude" on Claude's drafts, stubs, tuning done at his request and wiring, following `docs/ownership.md` and the changelog's Part tags. |

## Unscramble on titles (Joe, 2026-10-25)

The landing hero's `Scramble` (letters, digits and symbols resolving left to right) becomes the standard entrance for titles, used where it helps and not everywhere:

- Page titles (`h1`) on every page (`/kernels`, each kernel page, `/results`, `/dashboard`) unscramble when the page loads, after the intro loader on a full load or right away on a client-side navigation.
- Each kernel page's headline number in the header unscrambles from random digits, like the hero numbers (`numeric`, `.` `x` `%` and units stay fixed).
- Kernel names on the index page's cards unscramble once as they scroll into view, staggered.
- Not on: body text, timeline entries, chart labels, figure labels, nav, buttons, or anything seen repeatedly in a session (the motion rules of `web_plan.md`). Section headings inside a page (h2) stay plain unless Joe asks.
- Durations stay close to the hero's (titles about 1.6 s, numbers about 1.4 s). Reduced motion shows the final text, and screen readers always get it from the hidden copy.
- Build work: `Scramble` gains a `trigger` (`load` or `in-view`) and moves from the hero into a shared heading component; added in `web/kernels-template` (branch 2) and applied to the existing pages' titles there.

## Still open (ask when it comes up)

- Code excerpt length, whether pages chain with a "next kernel" link, URL slugs.
- Whether the index map's batch-16 toggle is worth it.

## How the template is built (for the next pages)

- Registry entry: `web/src/content/kernels/<slug>.ts` (`KernelStory`: title, short, formula, headline, sources, tests, testing, excerpts, measurements, timeline), added to `kernelStories` in `index.ts` and flipped to `ready` in `kernelCards` automatically.
- Numbers: `measured(sectionId, {column: value}, column, tableIndex)` reads `docs/profiling.md` tables (section ids are slugged headings, see `web/src/data/profiling.json`); a missing row or column fails the build.
- Story: a client component in `web/src/components/kernels/<slug>/story.tsx` passing steps, captions and a `figure(step)` function to `StickyStory`, registered in `story-view.tsx`. Figures are SVG (viewBox 560 x 400, aspect 14/10) using `useLoop(seconds)` from `loop.ts`: `on(start, end, still)` fades an element in a time window of the loop, `grow(start, end, axis)` scales a bar from its edge, `pulse(offset)`; with reduced motion they show `still` frames. Label illustrations as illustrations in the caption.
- Shared pieces (branch 55): `Frame`/`Label` in `components/kernels/frame.tsx`, `Toggle` in `toggle.tsx`, and `useLoop().move(start, end, dx, dy)` for boxes sliding between two points.
- Checks: `~/.cache/llmie-web/cdp-stepshot.mjs <url> <prefix> <w> <h> <ms>...` (figure-panel shots per step at several times; `THEME=light`, `REDUCED=1`), `~/.cache/llmie-web/cdp-overflow.mjs <url>` (text past the figure edge, outside its box, overlapping), `cdp-steps.mjs` (active step = visible text), `cdp-shot.mjs` (screenshots at scroll positions, 390 px for phones), `cdp-first.mjs` (first figure animates without a detour). Headless Chrome with `--remote-debugging-port=9333` (see `technical_notes.md`).

## Extra pages (Joe, 2026-10-31)

Joe asked for five more story pages and answered the design questions:

| Topic | Decision |
| --- | --- |
| Sampling | Reverses the earlier "sampling is out". New page `/kernels/sampling` (Gumbel-max, binary-searched top-k/top-p, why no full vocab sort). Numbers come from a new Part A bench first (`tools/sample-bench`: `scripts/sample_bench.cu`, greedy / Gumbel / top-k / top-p plus a full-sort top-p baseline at vocab 32,000, rows 1 and 16, profiling.md section). Timeline tags the sampler commits "Joe" (git authorship), despite ownership.md listing it as Part B. |
| Embedding | Not a page: collapsed cards in the layer map before and after the layer (embedding in, final norm + LM head out), expanding in place with formula, row-gather animation, source. |
| Split attention | Own page `/kernels/split-attention` (few CTAs at batch 1, SM occupancy grid, merge animation, branches 40/42/43 tables). Paged attention is trimmed to one short linking step. |
| Attention v1 to v2 | Own short page `/kernels/attention-v2` (883 to 272 us, shared-memory cap hit and removed). Paged attention trimmed to one short linking step. |
| Dead ends | `/kernels/dead-ends`, last card on the index, each entry linking back to its kernel page. Sources: docs/profiling.md AND git history / branch summaries (entries without numbers allowed, marked). |
| Branches | One per page, each from origin/main, stop for review after each: tools/sample-bench, web/kernel-split-attention, web/kernel-attention-v2, web/layer-map-embedding, web/kernel-sampling, web/dead-ends. |
