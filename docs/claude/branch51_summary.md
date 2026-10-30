# Branch 51 `web/prototypes` (worktree ~/code/llm-inference-engine-web), 5 commits STACKED on web/scaffold (d5d50f6), not pushed

Rebase onto main after #50 (web/scaffold) merges. Each commit passed the web gate.

1. 66eac99 picker harness (the prototype skill's PICKER.md, CSS verbatim incl. its one comment) + trace replay helpers + pinned libs (motion 14.1.0, @number-flow/react 0.6.2, recharts 3.10.1, liveline 0.0.7; npm audit clean).
2. 23b23ac `/prototypes/rain`: Cascade (classic rain, density = token rate), Lanes (one row per request, answer streams in), Pulse (one column per request, newest token on top, amber prefill bar).
3. eb03532 `/prototypes/tiles`: Readout (big NumberFlow numbers), Spark (number + 10 s Liveline trend), Gauge (16-slot batch bar + capacity bars).
4. be6f45d `/prototypes/story` (paged attention v1/v2/split, numbers from docs/profiling.md): Sticky, Inline, Stepper.
5. 1faf8ab `/prototypes/charts` (dataviz emphasis form, contrast validated): Lines, Bars, Speedup.

How to view: `cd web && npm run dev`, open /prototypes/rain (tiles, story, charts); 1-3 or arrows switch, R replays, `?v=2` in the URL. The rain and tiles need the chat trace from branch 49: until #49 merges, copy `results/traces/chat_b16_rate4.json` from the api worktree into `results/traces/` (it is untracked in the web worktree for that reason).

Open for Joe:
- Pick one variant per prototype (or ask for a riff around one); then Claude promotes the winners into the real pages and deletes `src/prototypes` and `/prototypes`.
- `web/AGENTS.md`: `next dev` writes an agent-instructions file and says to commit it. Not committed. Commit it, or set `agentRules: false` in next.config.ts to stop it.
- Verified with headless Windows Chrome (Claude in Chrome not connected): dark and light, reduced-motion still frames for the rain; fixed during checks: Pulse lost columns after a jump (slots freed only between frames), split figure bars misplaced (CSS px transforms in a scaled SVG), chart labels clipped, legend not following the theme.
