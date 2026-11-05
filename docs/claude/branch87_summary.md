# Branch 87: web/prefill-decode-prologue (issue #136)

Worktree ~/code/llm-inference-engine-pd, from origin/main cbffd5b, not pushed.

Commits (each gated alone with ~/.cache/llmie-pd/verify.sh, VERIFY WEB OK):
- 141c16d Add prefill and decode steps to the paged attention prologue
- 2b69082 Add the side by side step: recomputing the prefill against reading the cache
- cd5c10f Add the bridge step from the cache to paged attention blocks

Design: the page template's existing `lead` slot (registry `lead` + `KernelLeadView`) renders a "Prefill vs. Decode (the token lifecycle)" section right before "how it works", holding a second StickyStory with 4 new figures in web/src/components/kernels/attention/prologue-{story,figures}.tsx. Numbers: TTFT 18.3 / 1,611.6 ms and per-token 0.143 / 0.787 ms (docs/results.md prefill section), 243 GB/s and 75% of roof (roofline whole decode step, engine, 1 row), 9.13 ms decode step at 1,984 tokens with graphs (docs/profiling.md branch 43), all through measured(); 177x is computed at build time.

Verified over CDP (port 9482): desktop 1280 and phone 390, dark and light, reduced motion; 0 console errors in every run; transforms sampled over time (distinct values per step, none under reduced motion); existing 4 paged attention steps still scroll and animate.
Screenshots: ~/.cache/llmie-pd/s2 and s3 (d, dl, p, pl, r prefixes).
Not verified: OG image visually (unchanged code path), real touch scrolling, Safari/Firefox.
