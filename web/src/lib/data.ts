import decodeStepsJson from "@/data/decode_steps.json";
import environmentJson from "@/data/environment.json";
import hardwareJson from "@/data/hardware.json";
import kvDemoJson from "@/data/kv_demo.json";
import kvReplayJson from "@/data/kv_replay.json";
import launchGapsJson from "@/data/launch_gaps.json";
import modelShapeJson from "@/data/model_shape.json";
import onchipJson from "@/data/onchip.json";
import prefillStepJson from "@/data/prefill_step.json";
import profilingJson from "@/data/profiling.json";
import resultsDocJson from "@/data/results_doc.json";
import samplingLogitsJson from "@/data/sampling_logits.json";
import summariesJson from "@/data/summaries.json";
import tracesJson from "@/data/traces.json";

export type Summary = {
  system: string;
  batch: number;
  run: number;
  output_throughput: number;
  tpot_s_p50: number;
  ttft_s_p50: number;
  [key: string]: string | number;
};

export type TraceInfo = {
  file: string;
  version: number;
  meta: Record<string, string | number | boolean>;
  requests: number;
  steps: number;
  duration_s: number;
  max_batch_seen: number;
  commit: string | null;
};

export type ProfilingTable = { header: string[]; rows: string[][] };

export type ProfilingSection = { id: string; title: string; level: number; branch: number | null; tables: ProfilingTable[] };

export const summaries = summariesJson as unknown as Summary[];
export const environment = environmentJson as Record<string, string | number>;
export const traces = tracesJson as unknown as TraceInfo[];

export function preferredTrace(): TraceInfo | undefined {
  return traces.find((t) => t.meta.prompts !== undefined && t.meta.prompts !== "random") ?? traces[0];
}
export const profiling = profilingJson as unknown as ProfilingSection[];
export const decodeSteps = decodeStepsJson as unknown as Partial<Record<"b1" | "b16" | "graphs", ProfilingTable & { file: string }>>;

export type LaunchGaps = { file: string; steps: number; kernels: number; span_us: number; busy_us: number; idle_pct: number; gaps: number; gap_us: number };

export type GapProfile = "eager" | "graphs" | "b16";

const gapFiles: Record<GapProfile, string> = { eager: "decode_step.md", graphs: "decode_step_graphs.md", b16: "decode_step_b16.md" };

export function launchGaps(mode: GapProfile): LaunchGaps {
  const gaps = (launchGapsJson as Partial<Record<GapProfile, LaunchGaps>>)[mode];
  if (!gaps) throw new Error(`results/${gapFiles[mode]} has no decode launch gap line`);
  return gaps;
}

export type PrefillGemms = { file: string; kernel: string; count: number; ms: number; maxBatch: number; promptTokens: number; rows: number; spanUs: number };

type PrefillStepData = Partial<{ file: string; kernels: ProfilingTable; prefill: { steps: number; span_us: number }; max_batch: number }>;

export function prefillGemms(): PrefillGemms {
  const data = prefillStepJson as unknown as PrefillStepData;
  const { kernels, prefill, max_batch: maxBatch } = data;
  if (!data.file || !kernels || !prefill || !maxBatch) throw new Error("results/decode_step_b16.md is missing its Top 12 kernels table, its prefill launch gap line or its max_batch line");
  if (prefill.steps !== 1) throw new Error(`${data.file} profiled ${prefill.steps} prefill steps, so the rows of one prefill step cannot be read from it`);
  const prompt = /(\d+) prompt tokens/.exec(String(environment.workload ?? ""));
  if (!prompt) throw new Error("results/environment.txt has no workload line with a prompt length");
  const [kernelCol, countCol, msCol] = ["kernel", "count", "total ms"].map((name) => kernels.header.indexOf(name));
  if (kernelCol < 0 || countCol < 0 || msCol < 0) throw new Error(`${data.file} Top 12 kernels table has no kernel, count and total ms columns`);
  const layers = modelShape().layers;
  const found = kernels.rows.filter((r) => cellNumber(r[countCol]) === 4 * layers && r[kernelCol].includes("s16816gemm"));
  const ms = cellNumber(found[0]?.[msCol]);
  if (found.length !== 1 || ms === undefined) throw new Error(`${data.file} has ${found.length} tensor-op GEMM rows launched 4 x ${layers} times, expected one (qkv, o_proj, gate_up and down in every layer)`);
  const promptTokens = Number(prompt[1]);
  return { file: data.file, kernel: found[0][kernelCol], count: 4 * layers, ms, maxBatch, promptTokens, rows: maxBatch * promptTokens, spanUs: prefill.span_us };
}

export type KvRequest = { id: number; prompt: number; output: number; arrival_s: number; admitted: number; decodes: number; finished: number; final_length: number; blocks: { id: number; step: number }[] };

export type KvReplay = {
  file: string;
  meta: { kv_blocks: number; block_size: number; max_batch: number; num_requests: number; request_rate: number; gpu: string; date: string };
  duration_s: number;
  peak: { blocks: number; step: number; t: number; running: number; tokens: number };
  max_block: number;
  max_running: number;
  requests: KvRequest[];
  steps: [number, number, number, number][];
};

export function kvReplay(): KvReplay {
  const replay = kvReplayJson as unknown as Partial<KvReplay>;
  if (!replay.steps) throw new Error("results/traces/chat_b16_rate4.json is missing, so the KV replay is empty");
  return replay as KvReplay;
}

export type KvDemoRequest = { name: string; prompt: string; answer: string; prompt_tokens: string[]; output_tokens: string[] };

export type KvDemo = { date: string; gpu: string; commit: string; max_new_tokens: number; temperature: number; requests: KvDemoRequest[] };

export function kvDemo(): KvDemo {
  const demo = kvDemoJson as unknown as Partial<KvDemo>;
  if (!demo.requests) throw new Error("results/kv_demo.json is missing, so the KV visualizer has no requests (scripts/kv_demo.py writes it)");
  return demo as KvDemo;
}

export type SamplingContext = { name: string; note: string; user: string; answer_so_far: string; vocab: number; tokens: string[]; logits: number[]; tail: [number, number][] };

export type SamplingLogits = { date: string; device: string; commit: string; top: number; bin: number; contexts: SamplingContext[] };

export function samplingLogits(): SamplingLogits {
  const dump = samplingLogitsJson as unknown as Partial<SamplingLogits>;
  if (!dump.contexts) throw new Error("results/sampling_logits.json is missing, so the sampling sandbox has no logits (scripts/dump_logits.py writes it)");
  return dump as SamplingLogits;
}

export type LoadRun = { date: string; commit: string; gpu: string };

export const resultsDoc = resultsDocJson as unknown as { sections: ProfilingSection[]; ceilings: { gbs: number; tflops: number } | null; load: LoadRun | null; slo: { ttft_ms: number; tpot_ms: number } | null };

export type Doc = "profiling" | "results";

export const docPath: Record<Doc, string> = { profiling: "docs/profiling.md", results: "docs/results.md" };

export function docSections(doc: Doc): ProfilingSection[] {
  return doc === "results" ? resultsDoc.sections : profiling;
}

export function profilingTable(sectionId: string, index = 0, doc: Doc = "profiling"): ProfilingTable {
  const table = docSections(doc).find((s) => s.id === sectionId)?.tables[index];
  if (!table) throw new Error(`${docPath[doc]} has no table ${index} in section ${sectionId}`);
  return table;
}

export type EmbeddingLaunch = { date: string; maxBatch: number; promptTokens: number; bytesPerThread: number; valuesPerLoad: number; decodeUs: { naive: number; fast: number }; prefillTokens: number; prefillUs: { naive: [number, number]; fast: [number, number] }; underPct: number };

export function embeddingLaunch(): EmbeddingLaunch {
  const { header, rows } = profilingTable("kernels", 0, "results");
  const row = rows.find((r) => r[header.indexOf("Kernel")] === "Embedding lookup");
  const kernel = /one thread per (\d+) bytes.*not a multiple of (\d+)/.exec(row?.[header.indexOf("Optimized version")] ?? "");
  const m = /max batch (\d+) with ([\d,]+)-token prompts, ([\d-]+): a decode step ([\d.]+) to ([\d.]+) us, a ([\d,]+)-token prefill step ([\d.]+) to ([\d.]+) us to ([\d.]+) to ([\d.]+) us; under ([\d.]+)% of a step/.exec(row?.[header.indexOf("Nsight Compute before and after")] ?? "");
  if (!kernel || !m) throw new Error("docs/results.md Kernels: the Embedding lookup row has no \"one thread per N bytes\" or no \"a decode step A to B us, a T-token prefill step ... under P% of a step\" text");
  const n = (text: string) => cellNumber(text) as number;
  return { date: m[3], maxBatch: n(m[1]), promptTokens: n(m[2]), bytesPerThread: n(kernel[1]), valuesPerLoad: n(kernel[2]), decodeUs: { naive: n(m[4]), fast: n(m[5]) }, prefillTokens: n(m[6]), prefillUs: { naive: [n(m[7]), n(m[8])], fast: [n(m[9]), n(m[10])] }, underPct: n(m[11]) };
}

export type Hardware = { gpu: string; compute_capability: number; sms: number; vram_gb: number; l2_kb: number; shared_kb_per_sm: number };

export function hardware(): Hardware {
  const hw = hardwareJson as Partial<Hardware>;
  if (!hw.sms || !hw.shared_kb_per_sm || !hw.l2_kb || !hw.vram_gb) throw new Error("docs/hardware.md is missing SMs, VRAM, L2 or shared mem per SM");
  return hw as Hardware;
}

export type ModelShape = { hidden: number; intermediate: number; layers: number; heads: number; kv_heads: number; vocab: number; max_positions: number; roofline_context: number };

export const modelShape = (): ModelShape => modelShapeJson;

export function loadRun(): LoadRun {
  if (!resultsDoc.load) throw new Error("docs/results.md has no \"Engine runs measured\" line in its Under load section");
  return resultsDoc.load;
}

export function loadSlo(): { ttft_ms: number; tpot_ms: number } {
  if (!resultsDoc.slo) throw new Error("docs/results.md has no \"SLO: TTFT at most ... and TPOT at most ...\" line in its Under load section");
  return resultsDoc.slo;
}

export function ceilings(): { gbs: number; tflops: number } {
  if (!resultsDoc.ceilings) throw new Error("docs/results.md has no Ceilings line in its Roofline section");
  return resultsDoc.ceilings;
}

type OnchipRow = { metric: string; unit: string; run: number; value: number; sm_mhz: number | ""; date: string; commit: string; gpu: string };

const onchipRows = onchipJson as unknown as OnchipRow[];

export type Onchip = { median: number; min: number; max: number; mhz: number | undefined };

export function onchip(metric: string, unit: string): Onchip {
  const rows = onchipRows.filter((r) => r.metric === metric && r.unit === unit);
  if (rows.length === 0) throw new Error(`results/onchip/onchip.csv has no ${metric} in ${unit} (scripts/onchip_report.py writes it)`);
  const values = rows.map((r) => r.value);
  const clocks = rows.flatMap((r) => (typeof r.sm_mhz === "number" ? [r.sm_mhz] : []));
  return { median: median(values) as number, min: Math.min(...values), max: Math.max(...values), mhz: median(clocks) };
}

export function onchipRun(): { date: string; commit: string; gpu: string; runs: number } {
  if (onchipRows.length === 0) throw new Error("results/onchip/onchip.csv is missing (scripts/onchip_report.py writes it)");
  const { date, commit, gpu } = onchipRows[0];
  return { date, commit, gpu, runs: Math.max(...onchipRows.map((r) => r.run)) };
}

export function cellNumber(cell: string | undefined): number | undefined {
  const match = /^-?[\d,]*\.?\d+/.exec((cell ?? "").replace(/\s/g, ""));
  return match ? Number(match[0].replace(/,/g, "")) : undefined;
}

export const systems = [
  { id: "engine_graphs", label: "engine + CUDA Graphs" },
  { id: "engine", label: "engine" },
  { id: "llamacpp", label: "llama.cpp" },
  { id: "hf", label: "HuggingFace" },
];

export function median(values: number[]): number | undefined {
  if (values.length === 0) return undefined;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

export function medianOf(system: string, batch: number, key: keyof Summary): number | undefined {
  return median(
    summaries
      .filter((s) => s.system === system && s.batch === batch)
      .map((s) => Number(s[key]))
      .filter(Number.isFinite),
  );
}

export const batches = [...new Set(summaries.map((s) => s.batch))].sort((a, b) => a - b);

export function format(value: number | undefined, digits = 1): string {
  return value === undefined ? "–" : value.toFixed(digits);
}

export function headlineStats(): { label: string; value: string }[] {
  const graphs1 = medianOf("engine_graphs", 1, "output_throughput");
  const graphs16 = medianOf("engine_graphs", 16, "output_throughput");
  const hf16 = medianOf("hf", 16, "output_throughput");
  const tpot1 = medianOf("engine_graphs", 1, "tpot_s_p50");
  return [
    { label: "tokens/s at batch 16", value: format(graphs16, 0) },
    { label: "batch 16 over batch 1", value: graphs1 && graphs16 ? `${format(graphs16 / graphs1)}x` : "–" },
    { label: "over HuggingFace", value: graphs16 && hf16 ? `${format(graphs16 / hf16)}x` : "–" },
    { label: "ms per token at batch 1", value: format(tpot1 === undefined ? undefined : tpot1 * 1e3, 2) },
  ];
}
