# Delegation

Joe's original delegation prompt from the first session (Oct 3, 2026), verbatim, followed by every amendment Joe made since. Where an amendment conflicts with the original, the amendment wins. `working_rules.md` is the merged, current version of the rules.

## Original prompt (verbatim)

You are working in my repo llm-inference-engine (C++20 and CUDA, single-GPU LLM inference engine serving TinyLlama-1.1B). Read docs/, the build plan and the setup guide in the repo before doing anything, then ask me for clarification on anything ambiguous before you write code. I am the solo developer, I use WSL2 on an RTX 5060 Laptop (sm_120), and I will build and test on my machine, so tell me exactly which commands to run to verify each section.

### Your job

Implement the parts of the project I have delegated to you, and leave clearly marked stubs for the parts I will write myself. Do not write the stubbed parts, even partially, even if it seems easy or helpful.

### Part A: implement fully (I review and test only)

CMake, CMakePresets, CI workflow, clang-format and clangd config; the CLI and benchmark harness plumbing; CSV metrics output; plotting scripts; the Python reference and golden scripts (dump_reference.py, compare_logits.py, checksum.py); config and JSON parsing; the weight loader beyond the safetensors header parse; the SentencePiece tokenizer wrapper and chat template helper (verify against the HF tokenizer in a test); test boilerplate and the checksum tool; and README and docs formatting.

### Part B: implement as a draft (I will read and explain every line)

The cuBLAS GEMM wrappers (row-major trick, leading dimensions), the model runner forward pass wiring, the sampler (greedy, temperature via Gumbel-max, top-k, top-p), and the naive baseline kernels used as test oracles (naive attention over contiguous KV, naive RMSNorm, naive RoPE). Keep these short and readable, and add a short comment only where the logic is genuinely subtle (for example the HF rotate_half pairing, BF16 rounding order, GQA head mapping).

### Part C: leave for me, stub only

The optimized RMSNorm and RoPE kernels (the fused add_rmsnorm and the vectorized RoPE), the paged decode attention kernel, the safetensors header parser, the KV block manager, the scheduler and preemption logic, the KV cache layout and write_kv slot mapping, and CUDA Graph capture and bucket logic.

For every Part C item, create the file, the public header, the function or class signature, and the build wiring so the whole project compiles and links. The body must be a stub that fails loudly at runtime (throw std::logic_error("unimplemented: <name>") on the host, and for kernels a launcher that aborts with that message). Above the stub, put one large block comment headed TODO(JOE) containing, as continuous prose without line breaks inside each paragraph: what the component does and why it exists, the exact interface and tensor shapes and layouts, the numerical or HF-matching requirements, the edge cases and off-by-one boundaries to handle, what the stub must be compared against, and which test proves it correct. Do not include any implementation hints beyond the spec, and do not include pseudocode.

You may and should write the tests for Part C items (unit tests, golden tests, compute-sanitizer runs, boundary cases such as context lengths 1, 15, 16, 17 and 2048). Mark these tests with the GoogleTest label "todo_joe" so they are expected to fail until I implement the component, and make sure ctest -LE todo_joe passes cleanly. Also keep naive oracle versions in place so I can compare against them.

### Commits

Make small, atomic commits: one logical change per commit (for example "Add CUDA_CHECK macro", "Add CSV writer for bench results", "Add stub for block manager with spec"). Build and run ctest -LE todo_joe before every commit, and only commit when they pass. Write imperative commit messages under 72 characters that say what and why. Keep the default Co-Authored-By trailer and do not hide that you wrote the code. Do not backdate commits, do not set custom author dates, do not squash or rewrite history, and never push. I will push myself.

### Working style

Work milestone by milestone following the plan (M1 first), and stop after each milestone with a short summary of what you did, what I need to run, and which Part C stubs now exist. Do not start Part 2 items. No em dashes anywhere in code, comments or docs. Do not comment obvious code, and use concise idiomatic names. Do not add redundant defensive null checks or boilerplate error handling where standard propagation is fine. Paragraphs in docs and comments should be single continuous lines with no mid-paragraph line breaks.

## Clarifications Joe answered in the first session

- The build plan and setup guide were not in the repo; Joe pasted the handoff doc into the chat (now in `project_context.md`, `build_plan.md`, `setup_guide.md`).
- DeviceBuffer, Tensor and DType, CUBLAS_CHECK and the debug launch-check macro, and the pinned host buffer are Part A (Claude writes them).
- Kernels not named in the prompt split the same way as RMSNorm and RoPE: the naive versions (embedding, SwiGLU, residual add, naive contiguous attention, which also covers causal prefill for M2) are Part B; the optimized versions (fused or vectorized SwiGLU, paged prefill attention) are Part C.
- Branches: one per task, branched from main; Claude commits, Joe pushes and opens the PRs.
- GPU tests: Claude runs them outside the sandbox when the GPU is reachable; otherwise it runs CPU-only checks and tells Joe exactly which GPU tests to run.
- Python environment: `~/.venvs/llmie` (`scripts/requirements.txt` pins it). Model path: `models/tinyllama` (symlink to `~/models/tinyllama`).

## Amendments since (in order)

1. Oct 3: M1 target moved to Oct 7 because Joe is solo. Before committing the `parse_safetensors_header` stub, show Joe the full TODO(JOE) block and signature (done, approved). The cuBLAS wrapper and the pinned double-buffer upload are line-by-line review items: list their subtle lines (file:line) in the chat. Stop after each branch with a summary.
2. Oct 3: Never push; leave everything as local commits.
3. Oct 3: Work in a separate git worktree (`~/code/llm-inference-engine-claude`) because Joe uses the main checkout at the same time.
4. Oct 4: Commit messages are one line only. Do not add the Co-Authored-By trailer (this overrides the original "keep the default Co-Authored-By trailer"). Document every change in `changelog.md` at the repo root.
5. Oct 4: Write no code comments at all; leave explanatory notes in the chat instead. Exempt: clang-format's namespace closers and the TODO(JOE) spec blocks above Part C stubs.
6. Oct 4: Instead of comments, add the concepts each change relies on to `docs/exercises/learning_log.md`, and from then on update the learning log with new things Joe should familiarize himself with. Entries are terms only, with no meaning after `=>`; Joe fills the definitions himself.
7. Oct 5: Pull main before starting a new branch (Joe squash-merges PRs).
8. Oct 7: CI hardening went on its own branch (`build/ci-timeouts`).
9. Oct 8: Branch 6 was started in parallel with Joe writing the parser. Branch 7 (checksum and golden test) waits until the parser is merged.
10. Oct 8: Keep everything needed to continue without the chat in `docs/claude/` (this folder), gitignored.
11. Oct 8: PRs are merged with rebase and merge instead of squash (keeps the atomic commits on a linear `main`, so `git bisect` can find the exact change that breaks HF matching in M2). Supersedes the setup guide's squash convention and amendment 7's reason ("Joe squash-merges PRs"); branching new work from the latest `main` stays the rule.
12. Oct 9: Branch 8's README was rewritten at Joe's request as a project overview (goals, architecture, design decisions, hardware, short build section, layout, ownership) instead of a status and workflow page; milestone status, test labels, the todo_joe workflow and commit conventions do not belong in the README.
13. Oct 9: Housekeeping (untrack the `models` symlink and the `bandwith_test` binary) went on its own branch; Joe fixed his learning-log definitions himself.
14. Oct 9: M2 decisions, "yes to all" (see `m2_plan.md`): SentencePiece found or fetched; Claude drafts every naive kernel as Part B; forward-pass order; safetensors golden format; M2 scope is batch-1 greedy with a contiguous KV cache, temperature sampling optional.
15. Oct 9: Update all context in `docs/claude/` when a branch finishes (not only `status.md`).
16. Oct 9: M2 free-running criterion, option 1: identical or first divergence at most 2 BF16 ulps apart in HF's logits, on all 20 prompts; no minimum identical count.
17. Oct 25: Part 2 web site: Claude builds all of `web/` (Next.js static export on Vercel, in the engine repo) and writes all copy; Joe reviews and changes what he wants. Prototypes first (3 variants each), then promote his picks. Web-only commits use the web gate (`verify-web.sh`) instead of the engine gate.
18. Oct 25: The kernels section gets one page per kernel (core six + paged KV) with an animated sticky scroll-through and a timeline whose entries say who wrote each version (Joe for his Part C implementations, "drafted with Claude" for Claude's drafts, stubs, tuning done at his request and wiring).
19. Oct 26: "Update all context from this chat into claude/ and any memory": at the end of a session, bring `docs/claude/` (status snapshot, plans, summaries, rules, technical notes) and the auto-memory up to date.
