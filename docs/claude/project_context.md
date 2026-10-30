# LLM Inference Engine: Project Context and Handoff

This is Section 1 of the handoff doc Joe pasted into the first Claude Code session (planning conversation of Sep 23 to 24, 2026), reproduced verbatim. Section 2 is in `build_plan.md` and Section 3 in `setup_guide.md`. Note that Joe is now the solo developer; the partner described below has dropped out.

## 1. Project context and decisions

### 1.1 Who and what

- **Team:** Joe Zhuo (first-year Computer Engineering, University of Waterloo, GitHub joezhuo2) plus one partner. Both are new-ish to C++ and CUDA.
- **Project:** a single-GPU LLM inference engine in C++20 and CUDA, serving TinyLlama-1.1B-Chat-v1.0 with custom fused kernels, a paged KV cache, a paged decode attention kernel, continuous batching, GPU sampling, and CUDA Graphs. Repo name: `llm-inference-engine`.
- **Purpose:** a resume project for systems/performance roles (NVIDIA, inference startups, QNX-style C++ roles) and the long-term goal of new-grad AI inference / systems performance work. It replaces CrystalOS on the systems-focused resume. Joe's separate involvement in efim1 (a KV-cache reuse research project) is minor and not his own; this project is fully theirs.
- **Timeline:** MVP (Part 1) due Sunday Nov 15, 2026. Milestones: M1 Foundations Oct 4, M2 Correct naive engine Oct 18, M3 Paged KV + batching Nov 1, M4 Fast Nov 8, M5 Shipped Nov 15. If behind at M3, push the MVP to early December rather than let AI write the core; that still lands before the January 2027 WaterlooWorks main round. Part 2 (polish, UI/UX, harder kernels) has no deadline.

### 1.2 Hardware and environment

- **Joe:** Windows 11 laptop with an RTX 5060 Laptop GPU (Blackwell, compute capability 12.0 / `sm_120`, 8 GB GDDR7, 128-bit bus, ~3,328 CUDA cores / 26 SMs, 45 to 100 W TGP). Needs CUDA 12.8+. Setup: WSL2 + Ubuntu 24.04, CUDA toolkit only from NVIDIA's WSL-Ubuntu repo (13.3 planned), GCC 13, CMake 3.28, Ninja. Exact memory bandwidth is unconfirmed; measure it with a hand-written `bandwidth_test.cu` in week 1 and use that as the "% of peak" denominator.
- **Partner:** Mac (no CUDA possible). Builds and tests pure C++ components natively with the `cpu` CMake preset; runs GPU work over SSH into Joe's WSL2 (Tailscale + mirrored networking, port 2222, own Linux user) or on free Colab T4 as a fallback.
- **Cloud:** rent an L4 or RTX 4090 for one or two evenings near the end of Part 1 for portability benchmarks only.

### 1.3 Key technical decisions

- Model: TinyLlama-1.1B-Chat-v1.0 for the MVP (open weights, no gating, plain Llama, GQA 32 query heads / 4 KV heads, no RoPE scaling). Open question left as a doc comment: keep TinyLlama or start with Llama-3.2-1B. Default is TinyLlama.
- Numerics: BF16 weights and activations, FP32 accumulation. Match HF details exactly (rotate_half RoPE pairing i with i+32, BF16 residual rounding before RMSNorm, BF16 rounding before the weight multiply).
- Correctness criteria: teacher-forced argmax match 99%+ vs HF, mean abs logit error under 0.05, free-running greedy match for the first 64 tokens on 18 of 20 prompts.
- GEMMs via cuBLAS (row-major trick: `CUBLAS_OP_T, CUBLAS_OP_N`, m=out, n=B, k=in). Fuse QKV and gate/up weights at load time.
- KV cache: block size 16, layout `[num_blocks, kv_heads, 16, 64]` per layer, block 0 reserved as a null block, 90% of free VRAM after weights. Recompute-style preemption in the MVP.
- Paged attention: v1 stores scores in shared memory (context capped at 2,048), v2 uses online softmax and one block per KV head (GQA-aware).
- Sampler: all on GPU; Gumbel-max for temperature sampling; top-k/top-p via binary-searched thresholds (no full sort).
- CUDA Graphs: capture decode for batch buckets {1, 2, 4, 8, 16, 32}; prefill runs eagerly.
- MVP steps are all-prefill or all-decode; chunked prefill is a Part 2 fix, and the stall should be measured in the MVP so the fix can be shown later.
- Team split: Person A owns kernels and numerics, Person B owns the runtime (loader, KV manager, scheduler, graphs), with cross-over tasks so each can defend the whole project.
- Structural rule: CUDA headers only in `src/kernels/` and `src/runtime/`; public headers in `include/engine/` are plain C++ so the Mac build works.

### 1.4 Critique of the original blueprint (why the plan changed)

The original nine-subsystem blueprint (fused kernels, PagedAttention, FP8/INT4 GEMM, block manager, zero-copy loader, CUDA Graphs, continuous batching, sampling, telemetry + web API) was judged the right domain but too broad for the timeline and containing claims that would not survive an interview:

- "Fusing RMSNorm + GEMM bias + activation into one elementwise kernel" is not elementwise once a GEMM is involved; realistic fusion is into a GEMM epilogue.
- `mmap` pages are not pinned, so `cudaMemcpyAsync` from them is staged by the driver; use pinned bounce buffers or `cudaHostRegister`, and do not call it "zero-copy".
- A full `cub::DeviceRadixSort` of the vocabulary each step is overkill for top-k; use selection/thresholding.
- CUDA Graph argument updates need `cudaGraphExecKernelNodeSetParams` or fixed device buffers, not "replacing pointers".
- A hand-written GEMM will not beat cuBLAS; frame it as a learning kernel benchmarked against cuBLAS.
- FP8 `mma.sync` needs sm_89+; the 5060 (sm_120) supports it, so FP8 moved to Part 2 rather than being impossible.
- The web dashboard and WebSocket API add little for perf roles, so they moved to Part 2.

### 1.5 How to build it (AI usage policy)

Chosen approach: **AI as tutor and reviewer; hand-write the core; AI writes boilerplate the team reads and understands.**

- Hand-write: all kernels (RMSNorm, RoPE, SwiGLU, paged attention, sampler), block manager, scheduler and preemption, model runner forward pass, CUDA Graph capture, safetensors parser, performance analysis and results writeup.
- AI may write (and the team must understand every line): CMake and CI config, `.clang-format`, CLI parsing, CSV output, Python reference/dump scripts, plotting, test data generators, benchmark harness plumbing, and the Part 2 web dashboard.
- Rule of thumb: if it appears in a resume bullet or an interview question, the team writes it.
- Per-component loop: learn the concept, write the design note yourself and have AI critique it, implement it yourself (time-box being stuck to ~45 minutes, then ask for hints not solutions), have AI review for bugs/races/missed optimizations, then explain it to the partner.
- Rejected: "AI builds it, we study it" (illusion of understanding, fails deep interview follow-ups, conflicts with Joe's resume-accuracy standard). Also rejected: pure hand-writing with no AI help (too slow for beginners on the timeline).

Note: Joe's later delegation prompt (see `delegation.md`) deliberately overrides this list for the Part B items (cuBLAS wrappers, forward pass wiring, sampler, naive oracle kernels), which Claude drafts and Joe reviews line by line.

### 1.6 Resume impact analysis (honest assessment)

- **Strong for:** systems/performance roles (NVIDIA, inference startups, QNX-style C++, HPC-adjacent). Few first-years have hand-written CUDA kernels with profiler-backed numbers. It compounds across the second and third co-ops and fits the new-grad goal.
- **Smaller payoff for:** generic first-co-op SWE postings (Shopify, Wealthsimple, RBC/TD), where screening favors a real project, clean bullets, and passing LeetCode-style interviews. Keep the software-general resume version and do not let this crowd out Neetcode prep.
- **Risk:** "mini-vLLM" projects are increasingly common and AI makes shallow versions easy, so the title alone does not impress.
- **What makes it stand out, in order:** depth under 20 minutes of "why" questions; rigor (honest baselines vs HF and llama.cpp, reproducible scripts, a limitations section); a results writeup or blog post with ncu screenshots and plots; clear individual ownership in a two-person project.
- Verdict: worth doing if built the hand-written way; a liability if AI-built.

### 1.7 Learning plan (before and during the build)

Do not try to learn everything up front. Do a focused ~10-day foundation phase (Sep 24 to Oct 4), then learn each component just before building it.

**Phase 1 foundations (each ends with a proving exercise):**

| Track | Learn | Resource | Exercise |
| --- | --- | --- | --- |
| C++ | RAII, `unique_ptr`, move semantics, references vs pointers, basic templates, headers vs .cpp, compile/link | learncpp.com | Move-only `Buffer` wrapping malloc/free with a GoogleTest proving no double free |
| C++ | Binary files, mmap, basic CMake | learncpp + CMake tutorial steps 1 to 3 | Read the 8-byte header length of the TinyLlama safetensors file and print the JSON header |
| CUDA | Threads, blocks, warps, memory hierarchy, coalescing | PMPP ch. 1 to 6; NVIDIA "An Even Easier Introduction to CUDA" | Vector add, then `bandwidth_test.cu` reporting GB/s |
| CUDA | Reductions, shared memory, warp shuffles | PMPP ch. 10 | Sum 16M floats three ways (atomics, shared memory, warp shuffle) and time each |
| CUDA | Profiling | GPU MODE profiling lecture; Nsight Compute docs | Run ncu on the reduction and explain DRAM throughput |
| Transformers | Attention, RMSNorm, RoPE, SwiGLU, KV cache | 3Blue1Brown attention videos; Karpathy llama2.c run.c | Explain one token through one layer without notes |
| Transformers | TinyLlama real shapes and numerics | HF modeling_llama.py | Key exercise below |

**Key exercise:** rewrite TinyLlama's forward pass in plain PyTorch (no nn.Modules, ~100 lines), load real weights, match HF hidden states layer by layer, then add a KV cache and generate greedily. It becomes the reference oracle and turns the C++ build into translation of something fully understood.

**Phase 2 just-in-time learning:** before elementwise kernels, reread the concepts primer and port the PyTorch RMSNorm to naive CUDA; before paged attention, read the PagedAttention paper sections 3 to 4 and prototype block tables in PyTorch; before batching, skim Orca and nano-vllm's scheduler and simulate it in Python; before optimized kernels, read Simon Boehm's CUDA matmul post and add float4 loads to the bandwidth test; before CUDA Graphs, read the programming guide section and graph 100 tiny kernels. Pattern: prototype in Python, then build in C++/CUDA.

**Making it stick:** predict then measure; use AI as a Socratic tutor ("quiz me, don't give answers"); 15-minute weekly teach-back between partners; a learning log in `docs/notes.md`.

**Ready to start the engine when both can:** build and run a CUDA kernel with CMake and debug it with compute-sanitizer; explain why batch-1 decode is memory bound with the arithmetic; show a warp-shuffle reduction with its ncu bandwidth; generate correct text from the hand-written PyTorch TinyLlama with a KV cache; state every tensor shape in one decoder layer without notes.

### 1.8 Working preferences for Claude on this project

- Ask for specification or clarification before starting to write code.
- Never use em dashes in code or text; write paragraphs as continuous lines without mid-paragraph line breaks.
- Comments only for non-obvious edge cases; concise idiomatic names; no redundant defensive checks.
- Do not commit or push unless explicitly asked.
- Prefer direct recommendations with reasoning over menus of options.
- Guide and review the hand-written core; do not write the core kernels or runtime for them.
- Docs should be Google-Docs-copyable (no Mermaid diagrams or LaTeX blocks).
- Resume bullets use Jake's template and the Google XYZ format, filled only from real benchmark CSVs; each partner claims only what they built.

Note: several of these were later changed by Joe (Claude now commits but never pushes, and writes no code comments at all). `working_rules.md` has the current rules.

### 1.9 Status at handoff (Sep 24, 2026)

- Build plan doc and setup guide doc created (full text in `build_plan.md` and `setup_guide.md`).
- CPU-only bootstrap (CMake, presets, version lib, unit test, clang-format config) was verified to configure, build, pass tests, and pass clang-format on Ubuntu 24.04 with GCC 13 and CMake 3.28. The CUDA path (smoke kernel, GPU test) was not compiled in that environment and must be verified on the laptop (setup guide smoke test 6).
- Next steps: complete the setup guide checklist, then Phase 1 learning, then M1 by Oct 4.

The current status is in `status.md`.
