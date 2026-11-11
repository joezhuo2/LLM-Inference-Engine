import { goodputSeries, largestGap, loadProvenance, loadRates, loadValue, peakGoodput, ttftTailSeries } from "@/content/results";
import { ceilings, decodeSteps, launchGaps, loadSlo, modelShape, profilingTable } from "@/lib/data";
import { sweepOps, sweepPoints } from "./cuda-graphs";
import { flashLengths, flashLong, flashNumbers, flashPath, flashPaths, flashPerToken, flashProfile, flashShape, flashSystems, flashTokens, flashTtft } from "./flash-prefill";
import historyJson from "./step-history.json";
import { type KernelStory, measured } from "./types";

type Profile = { kernels: number; span_us: number; busy_us: number; idle_pct: number; gaps: number; tpot_ms: number | null; sample_us: number | null; ops: [string, number][] };
type Replay = { span_us: number; busy_us: number; idle_pct: number; tpot_ms: number | null };
type HistoryStage = { commit: string; date: string; eager: Profile; graphs: Replay | null };

const history = (historyJson as unknown as { stages: HistoryStage[] }).stages;
const regenerate = "run node web/scripts/step-history.mjs in a full clone and commit web/src/content/kernels/step-history.json";
if (history.length === 0) throw new Error(`step-history.json is empty: ${regenerate}`);

export const timeFmt = (v: number, digits = 0) => v.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });

export const stepGroups = [
  { id: "gemm", label: "GEMMs", ops: ["gate_up", "down", "qkv", "o_proj", "lm_head"] },
  { id: "attention", label: "attention", ops: ["attention"] },
  { id: "norm", label: "norm and add", ops: ["norm1", "norm2", "add1", "add2", "add1_norm2", "add2_norm1", "final_norm"] },
  { id: "rope", label: "RoPE", ops: ["rope"] },
  { id: "swiglu", label: "SwiGLU", ops: ["swiglu"] },
  { id: "rest", label: "write_kv, embedding", ops: ["write_kv", "embedding"] },
] as const;

export type GroupId = (typeof stepGroups)[number]["id"];

export type StepStage = { id: string; commit: string; date: string; title: string; what: string; kernels: number; span: number; busy: number; idle: number; idlePct: number; tpot: number | null; sampleUs: number | null; groups: Record<GroupId, number>; kernelSum: number; replay: boolean };

const known = new Set<string>(stepGroups.flatMap((g) => g.ops));

function groupsOf(profile: Profile): Record<GroupId, number> {
  const unknown = profile.ops.filter(([op]) => !known.has(op));
  if (unknown.length) throw new Error(`step-history.json has ops no group covers: ${unknown.map(([op]) => op).join(", ")}`);
  const us = (ops: readonly string[]) => profile.ops.filter(([op]) => ops.includes(op)).reduce((sum, [, v]) => sum + v, 0);
  return Object.fromEntries(stepGroups.map((g) => [g.id, us(g.ops)])) as Record<GroupId, number>;
}

const day = (date: string) => new Date(`${date}T00:00:00Z`).toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "UTC" });

function stage(h: HistoryStage, id: string, title: string, what: string, replay = false): StepStage {
  const groups = groupsOf(h.eager);
  const span = replay && h.graphs ? h.graphs.span_us : h.eager.span_us;
  const busy = replay && h.graphs ? h.graphs.busy_us : h.eager.busy_us;
  return {
    id,
    commit: h.commit,
    date: day(h.date),
    title,
    what,
    kernels: h.eager.kernels,
    span,
    busy,
    idle: span - busy,
    idlePct: replay && h.graphs ? h.graphs.idle_pct : h.eager.idle_pct,
    tpot: replay && h.graphs ? h.graphs.tpot_ms : h.eager.tpot_ms,
    sampleUs: h.eager.sample_us,
    groups,
    kernelSum: Object.values(groups).reduce((a, b) => a + b, 0),
    replay,
  };
}

const landed: { commit: string; id: string; title: string; what: string }[] = [
  { commit: "c6eb1f0", id: "first", title: "the first profile", what: "naive kernels, attention v1, one launch per kernel" },
  { commit: "72e666f", id: "fast", title: "fast RoPE and SwiGLU, add fused with the norm", what: "43 launches fewer per step" },
  { commit: "334736d", id: "v2", title: "attention v2", what: "online softmax, 8 lanes per key" },
  { commit: "f701cd2", id: "tuned", title: "add + RMSNorm tuned", what: "128-bit loads, the row kept in registers" },
];

const latest = history[history.length - 1];
const live = decodeSteps.b1;
const liveGaps = launchGaps("eager");
const liveOps = (live?.rows ?? []).map((r) => `${r[0]}=${Number(r[live!.header.indexOf("GPU us/step")])}`).join(",");
if (liveOps !== latest.eager.ops.map(([op, us]) => `${op}=${us}`).join(",") || liveGaps.span_us !== latest.eager.span_us) throw new Error(`results/decode_step.md is newer than step-history.json: ${regenerate}`);
if (!latest.graphs || latest.graphs.tpot_ms === null) throw new Error(`step-history.json has no graph replay profile for ${latest.commit}: ${regenerate}`);

export const stepStages: StepStage[] = [
  ...landed.flatMap((l) => {
    const h = history.find((x) => x.commit === l.commit);
    return h && h !== latest ? [stage(h, l.id, l.title, l.what)] : [];
  }),
  stage(latest, "eager", "today's kernels, launched one by one", "the same kernels as the step before, measured again"),
  stage(latest, "graphs", "the same step replayed from a CUDA Graph", "one launch for the whole step", true),
];

export const firstStage = stepStages[0];
export const eagerStage = stepStages[stepStages.length - 2];
export const nowStage = stepStages[stepStages.length - 1];
export const stageById = (id: string) => stepStages.find((s) => s.id === id);
export const maxSpan = Math.max(...stepStages.map((s) => s.kernelSum + s.idle));
export const smallOf = (s: StepStage) => s.groups.norm + s.groups.rope + s.groups.swiglu + s.groups.rest;
export const noisy = stepStages.filter((s) => !s.replay && s.idlePct > 1.5 * eagerStage.idlePct);

const shape = modelShape();
const headDim = shape.hidden / shape.heads;
const layerParams = shape.hidden * (shape.hidden + 2 * shape.kv_heads * headDim) + shape.hidden * shape.hidden + shape.hidden * 2 * shape.intermediate + shape.intermediate * shape.hidden;
const weightBytes = (shape.layers * layerParams + shape.vocab * shape.hidden) * 2;
const peak = ceilings();
const floorMs = (weightBytes / (peak.gbs * 1e9)) * 1e3;
const tokenMs = nowStage.tpot as number;
const spanMs = nowStage.span / 1e3;
const busyMs = nowStage.busy / 1e3;

export const token = {
  weightGb: weightBytes / 1e9,
  peakGbs: peak.gbs,
  ms: tokenMs,
  floorMs,
  aboveMs: busyMs - floorMs,
  gapsMs: spanMs - busyMs,
  hostMs: tokenMs - spanMs,
  floorPct: (floorMs / tokenMs) * 100,
  abovePct: ((busyMs - floorMs) / tokenMs) * 100,
  hostPct: ((tokenMs - spanMs) / tokenMs) * 100,
  spanPct: (floorMs / spanMs) * 100,
  firstMs: firstStage.tpot,
  eagerMs: eagerStage.tpot as number,
  gemmShare: (nowStage.groups.gemm / nowStage.kernelSum) * 100,
  firstGemmShare: (firstStage.groups.gemm / firstStage.kernelSum) * 100,
};

export const tokenParts = [
  { id: "floor", label: "the weights at the bandwidth peak", ms: token.floorMs },
  { id: "above", label: "kernel time above that", ms: token.aboveMs },
  { id: "gaps", label: "launch gaps", ms: token.gapsMs },
  { id: "host", label: "between steps: host work and sampling", ms: token.hostMs },
];

export const prefillPaths = [
  { id: "paged", label: "decode kernels, one walk per row", ms: flashLengths.map((p) => flashPath(flashPaths.paged, p)) },
  { id: "first", label: "first flash kernel", ms: flashLengths.map((p) => flashPath(flashPaths.first, p)) },
  { id: "tiled", label: "tiled flash kernel", ms: flashLengths.map((p) => flashPath(flashPaths.tiled, p)) },
  { id: "llama", label: "llama.cpp", ms: flashLengths.map((p) => flashTtft(flashSystems.llama, p)) },
];
export const prefillLengths = flashLengths;
export const prefillTokens = flashLengths.map(flashTokens);
export const prefillPerToken = prefillPaths.map((p) => p.ms.map((ms, i) => ms / prefillTokens[i]));
export const prefill = {
  long: flashLong,
  paged: flashNumbers.paged,
  first: flashNumbers.first,
  tiled: flashNumbers.tiled,
  llama: flashNumbers.llama,
  ratio: flashNumbers.ratio,
  firstLoss: flashNumbers.firstLoss,
  ahead: flashNumbers.ahead,
  perTokenLow: flashNumbers.perTokenLow,
  perTokenHigh: flashNumbers.perTokenHigh,
  pagedPerTokenShort: flashPerToken(flashSystems.paged, flashLengths[0]),
  pagedPerTokenLong: flashPerToken(flashSystems.paged, flashLong),
  attentionMs: flashProfile.whole.attention,
  gemmMs: flashProfile.whole.gemms,
  spanMs: flashProfile.whole.span,
  attentionPct: (flashProfile.whole.attention / flashProfile.whole.span) * 100,
  gemmPct: (flashProfile.whole.gemms / flashProfile.whole.span) * 100,
};

const present = (id: string) => goodputSeries.some((s) => s.id === id && s.values.some((v) => v !== undefined));
const engineIds = ["chunked", "budget-2048", "engine"].filter(present);
if (engineIds.length === 0 || !present("vllm")) throw new Error("docs/results.md Under load has no engine or no vLLM rows that content/results.ts knows");
const labelOf = (id: string) => goodputSeries.find((s) => s.id === id)?.label ?? id;
const peaks = engineIds.map((id) => ({ id, label: labelOf(id), ...peakGoodput(id) }));
const best = peaks.reduce((a, b) => (b.value > a.value ? b : a));
const vllmPeak = peakGoodput("vllm");
const topRate = loadRates[loadRates.length - 1];
const ratio = vllmPeak.value / best.value;
const slo = loadSlo();
const flashRow = profilingTable("kernels", 0, "results").rows.find((r) => r[0] === "Prefill attention");
const flashSince = /default since ([\d-]+)/.exec(flashRow?.[flashRow.length - 1] ?? "")?.[1] ?? null;
const sweepDay = loadProvenance.date.slice(0, 10);
const tailAt = (id: string, qps: number) => loadValue(ttftTailSeries, id, qps);
const within = (series: string, qps: number) => tailAt(series, qps) <= slo.ttft_ms;
const lastWithin = (id: string) => [...loadRates].reverse().find((q) => within(id, q));
const gapIds = engineIds.filter((id) => id === "engine" || id === "chunked");

export const load = {
  rates: loadRates,
  topRate,
  slo,
  date: loadProvenance.date,
  commit: loadProvenance.commit,
  gpu: loadProvenance.gpu,
  series: [...engineIds.filter((id) => id !== "budget-2048").reverse(), "vllm"].map((id) => ({ id, label: labelOf(id), goodput: loadRates.map((q) => loadValue(goodputSeries, id, q)), ttftP99: loadRates.map((q) => tailAt(id, q)) })),
  best,
  vllm: { ...vllmPeak, label: labelOf("vllm") },
  ratio,
  verdict: ratio >= 1.05 ? "vllm" : ratio <= 1 / 1.05 ? "engine" : "level",
  vllmStillRising: vllmPeak.qps === topRate,
  engineStillRising: best.qps === topRate,
  hasFlash: flashSince === null ? null : sweepDay >= flashSince,
  flashSince,
  tailRate: best.qps,
  tailEngine: tailAt(best.id, best.qps),
  tailVllm: tailAt("vllm", best.qps),
  engineHolds: lastWithin(best.id),
  vllmHolds: lastWithin("vllm"),
  gaps: gapIds.map((id) => ({ id, label: labelOf(id), largest: largestGap(id, best.qps) })),
} as const;

export const loadVerdict =
  load.verdict === "vllm"
    ? `the best goodput of vLLM is ${timeFmt(load.ratio, 1)} times the engine's`
    : load.verdict === "engine"
      ? `the engine's best goodput is ${timeFmt(1 / load.ratio, 1)} times vLLM's`
      : "the two reach the same goodput within 5%";

export const loadTitle = load.verdict === "vllm" ? "under load, vLLM still serves more" : load.verdict === "engine" ? "under load, the engine now serves more" : "under load, level with vLLM";

const sweepAt = (batch: number) => {
  const p = sweepPoints.find((x) => x.batch === batch);
  if (!p) throw new Error(`docs/results.md Batch size past 16 has no max batch ${batch}`);
  return p;
};
const opsAt = (batch: number) => {
  const p = sweepOps.find((x) => x.batch === batch);
  if (!p) throw new Error(`docs/results.md Batch size past 16 has no profile at max batch ${batch}`);
  return p;
};
const loadBatch = 16;
const widest = sweepOps.reduce((a, b) => (b.share > a.share ? b : a));
const half = sweepOps.find((x) => x.share >= 40) ?? widest;
const gumbel = "sampling-branch-65";
const cellRuns = (cell: string) => cell.split("/").map((p) => Number(p.trim())).filter(Number.isFinite);
const gumbelTable = profilingTable(gumbel, 2);
const floatRow = gumbelTable.rows.find((r) => r[0].startsWith("float"));
const floatRuns = cellRuns(floatRow?.[1] ?? "");
if (floatRuns.length === 0) throw new Error("docs/profiling.md Sampling: no float variant of gumbel() in the third table");

export const next = {
  loadBatch,
  kneeBatch: half.batch,
  throughputAtLoad: sweepAt(loadBatch).graphs,
  throughputAtKnee: sweepAt(half.batch).graphs,
  throughputGain: sweepAt(half.batch).graphs / sweepAt(loadBatch).graphs,
  kvPeakBlocks: measured("batch-size-past-16", { "max batch": String(sweepPoints[sweepPoints.length - 1].batch) }, "peak KV blocks", 0, "results"),
  largestBatch: sweepPoints[sweepPoints.length - 1].batch,
  attentionShare: half.share,
  attentionUs: opsAt(half.batch).attention,
  busyUs: opsAt(half.batch).busy,
  widestShare: widest.share,
  widestBatch: widest.batch,
  group: flashShape.group,
  requestedMb: flashProfile.step.requestedMb,
  neededMb: flashProfile.neededMb,
  tensorPct: flashProfile.step.tensor,
  prefillAttentionPct: prefill.attentionPct,
  hostMs: token.hostMs,
  hostPct: token.hostPct,
  roofNow: token.floorPct,
  roofWithout: token.spanPct,
  chunkCostPct: flashProfile.plainAfter.cost,
  chunkBudget: /budget ([\d,]+)/.exec(flashSystems.chunked)?.[1] ?? "",
  gumbelUs: measured(gumbel, { rows: "1" }, "temperature"),
  gumbelFloatLow: Math.min(...floatRuns),
  gumbelFloatHigh: Math.max(...floatRuns),
};

export const nextItems = [
  {
    id: "batch",
    title: `serve more than ${next.loadBatch} at once`,
    evidence: `${timeFmt(next.throughputAtLoad)} to ${timeFmt(next.throughputAtKnee)} tokens/s from ${next.loadBatch} to ${next.kneeBatch} rows, offline`,
    moves: "the goodput peak, and a fair run against vLLM",
    bar: Math.min(1, next.throughputGain - 1),
    value: `${timeFmt(next.throughputGain, 2)}x`,
  },
  {
    id: "kv",
    title: "read K and V once per KV head",
    evidence: `attention is ${timeFmt(next.attentionShare)}% of a ${next.kneeBatch}-row step, and ${timeFmt(next.prefillAttentionPct)}% of a ${flashLong}-token prefill`,
    moves: `the step at ${next.kneeBatch} rows and more, the first token of a long prompt`,
    bar: next.attentionShare / 100,
    value: `${timeFmt(next.attentionShare)}%`,
  },
  {
    id: "host",
    title: "prepare the next step while this one runs",
    evidence: `${timeFmt(next.hostMs, 2)} of the ${timeFmt(token.ms, 2)} ms of a token pass between two steps`,
    moves: `every decoded token: ${timeFmt(next.roofNow)}% of the roof to at most ${timeFmt(next.roofWithout)}%`,
    bar: next.hostPct / 100,
    value: `${timeFmt(next.hostPct)}%`,
  },
];

export const whereTheTimeWent: KernelStory = {
  slug: "where-the-time-went",
  index: 13,
  book: true,
  about: true,
  title: "where the time went",
  short: "The same decode step profiled after every kernel in this book, the first token of a long prompt, the engine under load against vLLM, and the three changes the measurements point to next.",
  formula: "time per token = weights ÷ bandwidth + what is left",
  headline: {
    value: `${timeFmt(token.floorPct)}%`,
    label: "of a batch-1 token is the read of the weights at this GPU's peak bandwidth",
  },
  sources: [
    { path: "docs/results.md", label: "every table this chapter reads" },
    { path: "docs/profiling.md", label: "the profiles, kernel by kernel" },
    { path: "results/decode_step.md", label: "the decode step profile, one version per commit" },
    { path: "results/decode_step_graphs.md", label: "the same step replayed from a graph" },
    { path: "scripts/profile_report.py", label: "kernels grouped by the op that launched them" },
    { path: "web/scripts/step-history.mjs", label: "reads every committed version of the profile" },
    { path: "docs/baselines.md", label: "how llama.cpp and vLLM were run" },
  ],
  tests: [],
  testing: "",
  excerpts: [],
  measurements: [
    { doc: "results", section: "gpu-time-per-op-in-a-decode-step", table: 0, caption: "today's batch-1 decode step, GPU time per op, launched one by one", command: "scripts/profile.sh nsys decode_step --max-batch 1 (run by scripts/reproduce.sh)" },
    { doc: "results", section: "prefill-against-prompt-length", table: 0, caption: "TTFT in ms against prompt tokens, one request per run, median of 5 (min to max)", command: "python scripts/prefill_ttft.py --llama-batched-bench <path> --token-budget 256" },
    { doc: "results", section: "under-load", table: 0, caption: `Poisson arrivals at each rate, medians of the runs; engine runs measured ${load.date} at commit ${load.commit}`, command: "scripts/load_sweep.sh, then scripts/load_report.py" },
    { doc: "results", section: "batch-size-past-16", table: 0, caption: "all requests submitted at once, twice as many as the max batch, median of 3 runs", command: "scripts/batch_sweep.sh, then scripts/batch_report.py" },
  ],
  timeline: [],
};
