# Branch 95: web/sandbox-gemm

Branch web/sandbox-gemm from origin/main d13ff65, two local commits, not pushed.

- c0b1e12 Add a tiling byte and reuse model for the GEMM page, with tests
- 172b1e8 Add the tile-it-yourself sandbox to the GEMM page

Files: web/src/components/kernels/gemm/tiling.ts, web/src/components/kernels/gemm/sandbox.tsx, web/scripts/tiling.test.mjs, wiring in web/src/content/kernels/types.ts, gemm.ts, web/src/components/kernels/story-view.tsx (KernelSandboxView), web/src/app/(content)/kernels/[slug]/page.tsx (Section id="sandbox"), changelog.md, docs/exercises/learning_log.md.

Verified: each commit passed verify-web.sh alone (audit, lint, 7 tiling tests plus existing, typecheck, build with no warnings). CDP checks at 1280 and 390, dark and light, reduced motion, no console errors, no horizontal overflow. Screenshots in ~/.cache/llmie-sb-gemm/shots/.

Not verified: hover styling (hover+fine pointer not emulated), real touch dragging, screen reader output.
