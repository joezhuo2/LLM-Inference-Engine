# Branch 85: web/chunked-prefill-story

Branch `web/chunked-prefill-story` in ~/code/llm-inference-engine-story, from origin/main cbffd5b, not pushed. Commits (each passed lint, tests, typecheck, build with no warnings): b7bd02b timeline entry and scene, 08536b0 kernel story /kernels/chunked-prefill, 8919cb4 spread figure tidy, 994ea52 new M6 refinements milestone holding the entry (Joe's change; M5 dates back to Oct 25).

Verified over CDP on ports 8477/9477: story steps dark, light, 390 px, reduced motion; 0 console errors; overflow script ok (two "box" reports are false positives from label next to a progress rect); home timeline shows M0..M6, scrubber labels fit at 1280 and 390 px; scene bar transforms sampled over time (scaleX 0.02 to 1). Screenshots in ~/.cache/llmie-story/shots/. Not verified: story page in light at phone checked only for errors (screenshots taken, not all viewed); no Lighthouse; the 14-mutant claim comes from the changelog.

Numbers used: 1611.6 ms (prefill table, engine, 2,048), goodput 6.1 vLLM at QPS 6 vs 3.2 engine + graphs at QPS 4, all read from docs/results.md via sync-data. No speedup is claimed; the 96-token, 3-decoder example is labelled illustrative and computed by `chunkedSteps`.

Open questions: timeline author tags (stub "claude", impl "both"); story index 11 placed after sampling, dead-ends stays last card; the "113 ms prefill" figure is prose only in profiling.md so it is not shown.
