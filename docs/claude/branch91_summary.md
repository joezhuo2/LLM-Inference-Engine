# Branch 91: web/sandbox-foundations

Worktree ~/code/llm-inference-engine-sb-foundations, from origin/main d13ff65, three local commits, not pushed.

Commits:
- f581963 web: sync the model shape and roofline context into the site data
- 6839ebd web: add the roofline sandbox model and its tests
- e330cef web: add the roofline sandbox to the foundations page

Each commit passed `verify-web.sh` (audit, lint, npm test, typecheck, build with no warnings). The three commit messages end with the Co-Authored-By trailer, as Joe asked, which overrides the no-trailer rule in the standing rules.

What it does: a "place an op on the roofline" section after "how it works" on /kernels/foundations. Eight ops, batch 1 to 512 on a log scale, context slider for attention, a ring on a log-log roofline with computed readouts and measured readouts (where results.md has the op), a predict-first box, presets, reset.

Verification: CDP at 1280 and 390, dark and light, plus reduced motion, with no console errors. I drove the controls and the readouts changed correctly. Keyboard: slider arrows and Enter on the op buttons work. The dot's transform transition was sampled over time (500 ms ease-in-out). Under reduced motion only opacity transitions. Screenshots are in ~/.cache/llmie-sb-foundations/shots/.

Not verified: real touch input, a screen reader, Safari and Firefox.

Open questions:
1. The model's shapes are parsed from the kTinyLlama fixture in tests/unit/test_config.cpp, because models/ is gitignored and Vercel cannot see it. Is that acceptable, or should the config be committed under results/?
2. The existing gemm and foundations stories still hardcode 11264 and 2048. I did not touch them.
3. RMSNorm, SwiGLU and RoPE FLOPs per element (5, 5, 3) are my counting assumptions and have no measured point.

Update: merged origin/main (099b87b), then ba78f56 "web: glide the roofline sandbox sliders and presets": both sandbox sliders use SmoothRange, jump buttons and reset tween over 600 ms. Both gated; a track click, reset and the 512 jump were sampled over time over CDP and glide on an ease-in-out curve to the target, with no exceptions.
