# Working rules

The current, merged rules for how Claude works on this repo. `delegation.md` has the history of where each rule came from.

## Who writes what

- **Part A (Claude writes, Joe reviews and tests):** CMake, presets, CI, clang-format and clangd config, CLI and benchmark harness plumbing, CSV metrics, plotting scripts, Python reference and golden scripts (`dump_reference.py`, `compare_logits.py`, `checksum.py`), config and JSON parsing, the weight loader beyond the safetensors header parse, the SentencePiece tokenizer wrapper and chat template helper, test boilerplate, the checksum tool, README and docs. Also DeviceBuffer, PinnedBuffer, Tensor and DType, and the check macros.
- **Part B (Claude drafts, Joe reads and explains every line):** cuBLAS GEMM wrappers, model runner forward pass wiring, the sampler (greedy, Gumbel-max temperature, top-k, top-p), and the naive oracle kernels (embedding, RMSNorm, RoPE, SwiGLU, residual add, attention over a contiguous KV cache).
- **Part C (Joe writes; Claude only stubs, specs and tests):** fused `add_rmsnorm`, vectorized RoPE, fused or vectorized SwiGLU, paged decode attention, paged prefill attention, the safetensors header parser, the KV block manager, the scheduler and preemption, the KV cache layout and `write_kv` slot mapping, CUDA Graph capture and buckets.
- Never write a Part C body, even partially. When Joe asks for help on Part C, give hints, explanations and reviews, not code.

## Part C stubs

- Create the file, the public header, the signature and the build wiring so everything compiles and links.
- Host stubs throw `std::logic_error("unimplemented: <name>")`; kernel stubs are launchers that abort with that message. `logic_error` is deliberately not a `runtime_error`, so rejection tests cannot pass against a stub by accident.
- Above the stub, one block comment headed `TODO(JOE): <name>`, prose only, one line per paragraph, covering: purpose and why it exists; exact interface, shapes and layouts; numerical or HF-matching requirements; edge cases and off-by-one boundaries; what to compare against; which test proves it. No hints, no pseudocode. Verify claims about reference behavior empirically before writing them (the safetensors spec was checked against the Python library on about 40 crafted files).
- Show Joe the full block and signature before committing it when he asks for that gate.
- Before committing todo_joe tests, validate them against a throwaway implementation in the scratchpad and a handful of deliberate mutants of it (see `technical_notes.md`, "Testing Part C stubs"); never commit or show that implementation.
- Tests for Part C go in `tests/todo/unit/` (target `todo_unit_tests`) or `tests/todo/gpu/` (target `todo_gpu_tests`), labelled `todo_joe` only (a second label is silently dropped by `gtest_discover_tests`, see `technical_notes.md`). When Joe's implementation passes, move the file into `tests/unit/` or `tests/gpu/` and delete the todo target if it is empty.

## Repo and branches

- Work only in the worktree `~/code/llm-inference-engine-claude`. Joe works in `~/code/llm-inference-engine` at the same time; never switch branches or edit tracked files there. Reading his checkout, and writing to this gitignored folder, is fine.
- Joe merges PRs with **rebase and merge** (decided Oct 8; PRs #6 to #12 were squash-merged before that). Every commit of a branch lands on `main` individually, with a new hash, so `main` stays linear and keeps the small atomic commits for `git bisect`. Consequences: every commit must build and pass the gate on its own (it already must); compare branches with `main` by content (`git diff`), not by hash; and keep commit messages clean, because they become `main`'s history verbatim.
- Before every branch: `git fetch`, then `git switch --no-track -c <area>/<task> origin/main`. Do not stack new work on unmerged branches unless it truly depends on them. If a stacked branch's base gets merged, the stacked branch must be rebased onto `main` (its base's commits now exist on `main` under new hashes); ask Joe before rebasing, and tell him the branch then needs `git push --force-with-lease` if it was already pushed.
- Before continuing on any branch, check `git status`, `git log` and `git reflog`: Joe sometimes resets, rewords or recommits Claude's branches. Never undo his changes.
- Never push. Joe pushes, opens PRs and merges them. Never rewrite history unless Joe explicitly asks (amending the latest local commit on request is fine). The setup guide's "squash merge" convention is superseded by rebase and merge.

## Commits

- Small and atomic: one logical change per commit, several commits per branch, all local until Joe pushes. Joe confirmed this on Oct 10: branch 16 landing as one squashed commit (#25) was an exception, not a new pattern.
- One-line imperative message under 72 characters. No body, no Co-Authored-By trailer.
- Every commit adds its entry to `changelog.md` at the repo root: newest branch section at the top (`## <branch-name>`), one bullet per change, tagged `(Part A)`, `(Part B)` or `(Part C stub)`.
- Every commit appends the new concepts it relies on to `docs/exercises/learning_log.md` as blank entries, `- **term** => ` with nothing after the arrow, under a `## <Topic> (Branch N)` section with `###` subsections at the end of the file. Branches are numbered in order across milestones (branch 9 was `build/repo-housekeeping`, branch 10 `tokenizer/sentencepiece`). Do not add a term Joe has already defined (grep the log first). Never write the definitions; Joe fills them in. When Joe's definitions are wrong, say so briefly in the chat.
- Parallel branches conflict at the top of the changelog and the end of the learning log; the resolution is always "keep both".

## Golden data

- Golden files live in `tests/golden/` (gitignored) and are always regenerated by a script: `scripts/checksum.py` (`checksums.json`), `scripts/tokenizer_golden.py` (`tokenizer.json`), `scripts/dump_reference.py` (`reference/`). Tests skip when the model or the golden is missing; every new golden test is checked once by tampering with the golden and confirming it fails.
- Python scripts run in `~/.venvs/llmie`; probe HF behavior empirically before matching it (record what was verified in `technical_notes.md`).

## Code style

- No code comments at all, except clang-format's `}  // namespace x` closers and the TODO(JOE) blocks. Explanations go in the chat, and `review_notes.md` keeps them.
- No em dashes anywhere. Paragraphs in docs and comments are single lines (`.clang-format` has `ReflowComments: false` so they stay that way).
- Concise idiomatic names; no redundant defensive checks; validate untrusted input (files, configs) and fail fast with messages that name the offending field or tensor.
- Public headers in `include/engine/` stay plain C++ (no CUDA includes); CUDA headers only in `src/kernels/` and `src/runtime/`. `engine::Stream` and `struct cublasContext;` forward declarations keep CUDA types out of public headers.
- Libraries: `engine_core` (pure C++), `engine_kernels` (`.cu` and cuBLAS), `engine_runtime` (CUDA host code).

## The gate before every commit

Run `bash ~/.cache/llmie-claude/verify.sh && git commit -m "..."` and nothing else; never chain a commit after a test with `;`. Never pipe the gate (`verify.sh | tail && git commit` commits on tail's exit status, which let an unformatted commit through on Oct 9); to keep the output short, redirect it to a log in the scratchpad and `tail` the log afterwards. The script (copy in `verify.sh` here) checks clang-format on tracked and new files, builds the debug and cpu presets with zero warnings, and runs `ctest -LE todo_joe` on both, skipping the `gpu` label when the GPU is unreachable. It greps the build output for `warning:`, `error:` and `FAILED:` (colons included, since a bare `error` matches file names like SentencePiece's `error.cc.o`). It only sees warnings in files rebuilt during that run, so a file that warned once must be fixed, not just rebuilt. The gate builds the working tree, so stage only what the commit contains and keep unrelated work out of the tree when it could change the result. If the script is missing (the session scratchpad gets wiped, so it lives in `~/.cache`), recreate it from `verify.sh` here.

## The GPU

GPU access from Claude's process is flaky even with the sandbox disabled: `cudaErrorInsufficientDriver` and a garbage device query come and go, sometimes mid-session. Check with `~/scratch/query` (it must print `compute capability: 12.0`). When the GPU is down, run CPU-only checks, say so plainly, and give Joe the exact GPU commands to run (`ctest --preset debug -LE todo_joe`, a `--gtest_filter` for the new tests, and `compute-sanitizer --tool memcheck --leak-check full`).

## Reporting

- Stop after each branch with: the commits, notes that would otherwise be comments, the subtle lines for any line-by-line review item (file:line and why), exactly what Joe must run (one command per fenced `bash` block), which Part C stubs exist, and what is next.
- Be honest about anything that slipped (an ungated commit, an untested path).
- Joe prefers a direct recommendation with reasoning over a menu of options.

## Web site (`web/`, since 2026-10-25)

- Claude builds all of `web/` and writes all copy; Joe reviews and edits. Work in `~/code/llm-inference-engine-web`. Design decisions live in `web_plan.md` and `kernels_plan.md`; add every new decision there.
- Same branch and commit rules as the engine (one-line messages, no trailer, changelog entry per commit, blank learning-log terms under `## <Topic> (Branch N)`, no code comments, never push). Branch numbers continue the changelog's numbering (52 web/overview, 53 web/kernels-data, 54 web/kernels-template).
- Gate: `bash ~/.cache/llmie-web/verify-web.sh > ~/.cache/llmie-web/gate.log 2>&1 && git commit ...` (refuses staged engine files; audit, lint with 0 warnings, `npm test`, typecheck, static build with no warnings). Joe approved this instead of the engine gate for web-only commits. It skips `npm ci` when node_modules matches the lockfile, because `npm ci` under Joe's running `next dev` put Turbopack into an endless reload loop (restart `next dev` if that ever happens).
- The gate builds the working tree, not the commit. When one turn makes several commits, build each commit alone afterwards in a temporary detached worktree (`git worktree add --detach`, `cp -al web/node_modules`, copy untracked `results/traces/*.json`; a symlinked node_modules makes Turbopack fail), then remove the worktree.
- Pin exact npm versions released at least two weeks earlier, and check `npm audit` before adding one.
- Numbers on the site come from `results/`, `docs/profiling.md` or the recorded trace at build time, never typed into the page. Illustrations that are not measurements say so in their caption.
- Verify in a browser before saying something works: headless Windows Chrome over CDP (see `technical_notes.md`), dark and light, 390 px, reduced motion. Joe watches the dev server on port 3000; test against it as well as the static build, and never kill it.
- Joe often sends fixes while a turn is running; address each one, and confirm with a measurement (for example sampled transforms during a transition), not only a screenshot.
- `web/AGENTS.md` (written by `next dev`) stays untracked until Joe decides.
