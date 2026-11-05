# Branch 86: web/memory-wall-story (story 00 "foundations", /kernels/foundations)

Worktree ~/code/llm-inference-engine-mw, from origin/main cbffd5b, 5 local commits, nothing pushed. Issue #135.

Commits: bafd405 parse docs/hardware.md into hardware.json; 7731548 add the story as 00 with execution and memory hierarchy steps; 0c77031 compute-bound and memory-bound steps; 9fdf88f fusion step and links; 4496d8a rename to foundations. Each gated alone with verify.sh (lint, tests, typecheck, build, audit): VERIFY WEB OK.

Numbering: KernelCard gets `index` (stories' own index; dead-ends = stories.length) and kernel-cards.tsx prints it, so foundations is 00 and the rest keep 01 to 10, dead-ends 11. Page template hides timeline, tested, code sections when empty.

Verified over CDP (ports 9481/8481): card numbers, crumb "kernels / 00", nav (next: paged attention), 0 console errors, no horizontal overflow at 1280 and 390, dark and light, reduced motion stills, sampled transforms moving over time (zoom group, packets, roofline ring). Shots: ~/.cache/llmie-mw/shots.

Not verified: OG image content visually (route builds, foundations.png emitted); the last light/dark 1280 shot batch finished in background only partly reviewed; Lighthouse.

Open questions: register file size, shared/register bandwidth and latency are not in the repo, so the memory figure states no numbers for them (only DRAM 323.6 GB/s measured). Rows 64/512 in the roofline and picker are computed from the two ceilings with FLOP/byte = rows x the measured 1.0, labelled not measured. The 5060 Laptop uses GDDR, so the page says global memory (DRAM) and notes it is not HBM. SwiGLU and RoPE are linked as fused-style kernels though they are single-op kernels. Dated lines "SwiGLU first draft 22 blocks of 256" come from the swiglu story.
