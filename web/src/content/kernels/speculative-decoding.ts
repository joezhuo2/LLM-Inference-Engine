import { profilingTable, speculativeExample } from "@/lib/data";
import { measured, type KernelStory } from "./types";

export const specFmt = (v: number, d = 0) => v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });

const section = "speculative-decoding";
const table = profilingTable(section, 0, "results");
const column = (name: string) => {
  const i = table.header.indexOf(name);
  if (i < 0) throw new Error(`docs/results.md Speculative decoding: no column "${name}"`);
  return i;
};
const sameOutput = (row: string[]) => {
  const match = /^(\d+) of (\d+)$/.exec(row[column("same output")]);
  if (!match) throw new Error('docs/results.md Speculative decoding: "same output" must read "N of M"');
  return { same: Number(match[1]), prompts: Number(match[2]) };
};
const number = (row: string[], name: string) => {
  const v = Number.parseFloat(row[column(name)]);
  if (Number.isNaN(v)) throw new Error(`docs/results.md Speculative decoding: no number in "${name}"`);
  return v;
};

export type SweepRow = { set: "repeat" | "chat"; k: number; tokens: number; steps: number; drafted: number; accepted: number; acceptedPct: number; tokensPerStep: number; ms: number; speedup: number; same: number; prompts: number };

export const sweep: SweepRow[] = table.rows.map((r) => ({
  set: r[column("prompts")] === "repeat" ? "repeat" : "chat",
  k: number(r, "max draft"),
  tokens: number(r, "output tokens"),
  steps: number(r, "steps"),
  drafted: number(r, "draft tokens"),
  accepted: number(r, "accepted"),
  acceptedPct: number(r, "accepted %"),
  tokensPerStep: number(r, "tokens per step"),
  ms: number(r, "ms per token"),
  speedup: number(r, "speedup"),
  ...sameOutput(r),
}));

const of = (set: SweepRow["set"]) => sweep.filter((r) => r.set === set).sort((a, b) => a.k - b.k);
export const repeatRows = of("repeat");
export const chatRows = of("chat");
if (repeatRows.length < 2 || chatRows.length < 2 || repeatRows[0].k !== 0 || chatRows[0].k !== 0) throw new Error("docs/results.md Speculative decoding: each prompt set needs a row without drafts and at least one with");

const fastest = (rows: SweepRow[]) => rows.reduce((best, r) => (r.ms < best.ms ? r : best));
const slowest = (rows: SweepRow[]) => rows.reduce((worst, r) => (r.ms > worst.ms ? r : worst));

const example = speculativeExample();
const rowsTable = profilingTable(section, 2, "results");
export const rowTimes = rowsTable.rows.map((r) => ({ rows: Number.parseInt(r[0], 10), steps: Number.parseInt(r[1].replace(/,/g, ""), 10), ms: Number.parseFloat(r[2]) }));
if (rowTimes.length < 2 || rowTimes.some((r) => Number.isNaN(r.rows + r.steps + r.ms))) throw new Error("docs/results.md Speculative decoding: the third table must list rows, steps and median ms");
const drafted = example.steps.reduce((sum, s) => sum + s.draft.length, 0);
const kept = example.steps.reduce((sum, s) => sum + s.kept, 0);
const bestStep = example.steps.reduce((best, s) => (s.kept > best.kept ? s : best));
const missStep = example.steps.find((s) => s.draft.length > 0 && s.kept < s.draft.length && s.kept > 0) ?? example.steps.find((s) => s.draft.length > 0 && s.kept < s.draft.length) ?? bestStep;

const step1 = measured("throughput-and-latency", { system: "engine + graphs", "max batch": "1" }, "TPOT p50 ms", 0, "results");
const step16 = measured("throughput-and-latency", { system: "engine + graphs", "max batch": "16" }, "TPOT p50 ms", 0, "results");

export const piece = (token: string) => token.replace(/▁/g, " ").replace(/<0x0A>/g, "↵");

export const spec = {
  example,
  date: example.date.slice(0, 10),
  maxDraft: example.max_draft,
  maxNgram: example.max_ngram,
  maxBatch: example.max_batch,
  promptTokens: example.prompt_tokens.length,
  outputTokens: example.output_tokens.length,
  decodeSteps: example.steps.length,
  drafted,
  kept,
  keptPct: drafted ? (kept / drafted) * 100 : 0,
  bestStep,
  missStep,
  step1,
  step16,
  rowCost: step16 / step1,
  plain: repeatRows[0],
  best: fastest(repeatRows),
  most: repeatRows[repeatRows.length - 1],
  chatPlain: chatRows[0],
  chatWorst: slowest(chatRows),
  compared: fastest(repeatRows).prompts + (chatRows.find((r) => r.k === fastest(repeatRows).k)?.prompts ?? 0),
  changed: fastest(repeatRows).prompts - fastest(repeatRows).same + (chatRows.find((r) => r.k === fastest(repeatRows).k)?.prompts ?? 0) - (chatRows.find((r) => r.k === fastest(repeatRows).k)?.same ?? 0),
  chatAt: (k: number) => chatRows.find((r) => r.k === k),
  repeatAt: (k: number) => repeatRows.find((r) => r.k === k),
};

const chatAtBest = spec.chatAt(spec.best.k);
if (!chatAtBest || chatAtBest.ms >= spec.chatPlain.ms) throw new Error("docs/results.md Speculative decoding: the chapter says chat prompts gain at the best draft length of the repeat prompts; rewrite its last step");

export const speculativeDecoding: KernelStory = {
  slug: "speculative-decoding",
  index: 17,
  book: true,
  about: true,
  title: "speculative decoding",
  short: `A decode step has rows to spare. Fill them with guesses of the next tokens, copied from the context, and one step can yield several tokens: ${specFmt(spec.best.tokensPerStep, 2)} per step on prompts whose answers repeat them, and the output is the one plain decoding writes.`,
  formula: "tokens per step = 1 + the guesses the model agrees with",
  headline: {
    value: `${specFmt(spec.best.speedup, 2)}×`,
    label: `tokens per second on answers that repeat their prompt, with up to ${spec.best.k} guessed tokens per step (measured, one request at a time)`,
  },
  sources: [
    { path: "src/sched/speculative.cpp", label: "the guess, the count of agreed guesses, giving cache slots back" },
    { path: "src/sched/scheduler.cpp", label: "which rows get guesses, what a step keeps" },
    { path: "src/sched/paged_batch.cpp", label: "a sequence as several rows of one step" },
    { path: "src/runtime/serve.cpp", label: "one step: schedule, forward, sample, update" },
  ],
  tests: [
    { path: "tests/unit/test_speculative.cpp", label: "the lookup, the accept count, truncate" },
    { path: "tests/unit/test_scheduler_speculative.cpp", label: "the same answers as plain decoding" },
    { path: "tests/gpu/test_serve.cpp", label: "the same tokens on the GPU" },
  ],
  testing: "",
  excerpts: [
    {
      file: "src/sched/speculative.cpp",
      from: "std::vector<int32_t> propose_prompt_lookup(std::span<const int32_t> context, int max_ngram,",
      to: "    return {};",
      caption: "The whole guess: find where the last few tokens appeared before, and copy what followed",
    },
  ],
  measurements: [{ doc: "results", section, table: 0, caption: "one request at a time with up to 0 to 15 guessed tokens per step: steps, accepted guesses and time per token", command: "scripts/speculative_sweep.sh, then scripts/speculative_report.py" }],
  timeline: [],
};
