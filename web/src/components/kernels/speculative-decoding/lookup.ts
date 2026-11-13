import { rowTimes, spec } from "@/content/kernels/speculative-decoding";

export type Knobs = { maxDraft: number; maxNgram: number; adaptive: boolean };
export type ReplayStep = { at: number; draft: string[]; kept: number };

const example = spec.example;
const prompt = example.prompt_tokens;
const output = example.output_tokens;

export function proposePromptLookup(context: string[], maxNgram: number, maxDraft: number): string[] {
  const n = context.length;
  if (n < 2) return [];
  for (let g = Math.min(maxNgram, n - 1); g >= 1; --g) {
    for (let i = n - 1 - g; i >= 0; --i) {
      let same = true;
      for (let j = 0; j < g && same; ++j) same = context[i + j] === context[n - g + j];
      if (same) return context.slice(i + g, Math.min(i + g + maxDraft, n));
    }
  }
  return [];
}

export function replay({ maxDraft, maxNgram, adaptive }: Knobs): ReplayStep[] {
  const steps: ReplayStep[] = [];
  let emitted = 1;
  let limit = maxDraft;
  while (emitted < output.length) {
    const want = Math.min(adaptive ? limit : maxDraft, example.max_batch - 1);
    const draft = want > 0 ? proposePromptLookup([...prompt, ...output.slice(0, emitted)], maxNgram, want) : [];
    let agreed = 0;
    while (agreed < draft.length && draft[agreed] === output[emitted + agreed]) ++agreed;
    const kept = Math.min(agreed, output.length - emitted - 1);
    steps.push({ at: prompt.length + emitted, draft, kept });
    emitted += kept + 1;
    if (adaptive && draft.length > 0) limit = agreed === draft.length ? Math.min(maxDraft, limit * 2) : Math.max(1, Math.floor(limit / 2));
  }
  return steps;
}

const recorded = replay({ maxDraft: example.max_draft, maxNgram: example.max_ngram, adaptive: false });
if (recorded.length !== example.steps.length || recorded.some((s, i) => s.kept !== example.steps[i].kept || s.draft.join("\u0000") !== example.steps[i].draft.join("\u0000"))) throw new Error("speculative sandbox: the TypeScript lookup no longer rebuilds the recorded steps of results/speculative_example.json");

const sorted = [...rowTimes].sort((a, b) => a.rows - b.rows);

export function stepMs(rows: number): number {
  const hi = sorted.findIndex((r) => r.rows >= rows);
  if (hi === 0) return sorted[0].ms;
  if (hi < 0) return sorted[sorted.length - 1].ms;
  const a = sorted[hi - 1];
  const b = sorted[hi];
  return a.ms + ((b.ms - a.ms) * (rows - a.rows)) / (b.rows - a.rows);
}

export function totals(steps: ReplayStep[]) {
  const drafted = steps.reduce((sum, s) => sum + s.draft.length, 0);
  const kept = steps.reduce((sum, s) => sum + s.kept, 0);
  const ms = steps.reduce((sum, s) => sum + stepMs(1 + s.draft.length), 0);
  return { steps: steps.length, drafted, kept, wasted: drafted - kept, tokensPerStep: (steps.length + kept) / steps.length, msPerToken: ms / (steps.length + kept) };
}
