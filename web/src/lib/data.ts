import decodeStepsJson from "@/data/decode_steps.json";
import environmentJson from "@/data/environment.json";
import kvDemoJson from "@/data/kv_demo.json";
import kvReplayJson from "@/data/kv_replay.json";
import launchGapsJson from "@/data/launch_gaps.json";
import profilingJson from "@/data/profiling.json";
import resultsDocJson from "@/data/results_doc.json";
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
export const decodeSteps = decodeStepsJson as unknown as Partial<Record<"b1" | "b16", ProfilingTable & { file: string }>>;

export type LaunchGaps = { file: string; steps: number; kernels: number; span_us: number; busy_us: number; idle_pct: number; gaps: number; gap_us: number };

export function launchGaps(mode: "eager" | "graphs"): LaunchGaps {
  const gaps = (launchGapsJson as Partial<Record<"eager" | "graphs", LaunchGaps>>)[mode];
  if (!gaps) throw new Error(`results/${mode === "eager" ? "decode_step.md" : "decode_step_graphs.md"} has no decode launch gap line`);
  return gaps;
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

export const resultsDoc = resultsDocJson as unknown as { sections: ProfilingSection[]; ceilings: { gbs: number; tflops: number } | null };

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

export function ceilings(): { gbs: number; tflops: number } {
  if (!resultsDoc.ceilings) throw new Error("docs/results.md has no Ceilings line in its Roofline section");
  return resultsDoc.ceilings;
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
