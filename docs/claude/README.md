# Claude handoff notes

This folder holds everything a new Claude Code session needs to continue work on this repo without the original chat. It is gitignored and local to this machine. A new session should read `status.md` first, then `working_rules.md` and `delegation.md`, and open the plan files only for the milestone it is working on.

| File | What it contains |
| --- | --- |
| `status.md` | Where the project stands right now and the exact next steps. Update it at the end of every session. |
| `working_rules.md` | How Claude works on this repo: worktree, branches, commits, changelog, learning log, comments, the verify gate, the flaky GPU. |
| `delegation.md` | Joe's original delegation prompt (Part A, B and C) verbatim, plus every amendment Joe made since. |
| `m1_plan.md` | The M1 branch plan and what each branch delivered (M1 closed Oct 9). |
| `m2_plan.md` | The M2 decisions Joe confirmed, the branch plan, and the reference (golden) data format (M2 closed Oct 9). |
| `m3_plan.md` | M3 ownership, the proposed branch order (branch 15 onward), and the decisions made in each branch. |
| `review_notes.md` | The explanations that would otherwise be code comments, per branch, including the lines Joe asked to review line by line. |
| `technical_notes.md` | Decisions, gotchas and verified facts (safetensors reference behavior, HF tokenizer and model behavior, SentencePiece versions, CMake and CI pitfalls, cuBLAS details). |
| `project_context.md` | Section 1 of Joe's handoff doc: who, what, hardware, key decisions, AI usage policy, learning plan. |
| `build_plan.md` | Section 2 of Joe's handoff doc: the full build plan (milestones, component specs, testing, Part 2). |
| `setup_guide.md` | Section 3 of Joe's handoff doc: the full setup guide. |
| `web_plan.md` | The web site plan (Part 2): look, stack, pages, motion rules, branch order, and every design decision Joe made since. |
| `kernels_plan.md` | The kernels section plan: the 7 kernel pages, page template, visual models, data sources, branch order, Joe's answers, and how to add the next page. |
| `branchNN_summary.md` | PR summaries Claude writes when a branch is done (what each commit does, how it was checked, what is open for Joe). The newest: `branch52_summary.md` (landing page), `branch53_54_summary.md` (kernels). |
| `verify.sh` | The pre-commit gate script (a copy also lives at `~/.cache/llmie-claude/verify.sh`). |

Claude works in separate worktrees, not in Joe's checkout: `~/code/llm-inference-engine-claude` (engine), `~/code/llm-inference-engine-api` (a second engine worktree for parallel agents) and `~/code/llm-inference-engine-web` (the web site). Joe's checkout is `~/code/llm-inference-engine`; read these files from `~/code/llm-inference-engine/docs/claude/` (gitignored files are not shared between worktrees).
