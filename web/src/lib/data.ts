import decodeStepsJson from "@/data/decode_steps.json";
import environmentJson from "@/data/environment.json";
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
};

export type ProfilingTable = { header: string[]; rows: string[][] };

export type ProfilingSection = { id: string; title: string; level: number; branch: number | null; tables: ProfilingTable[] };

export const summaries = summariesJson as unknown as Summary[];
export const environment = environmentJson as Record<string, string | number>;
export const traces = tracesJson as unknown as TraceInfo[];
export const profiling = profilingJson as unknown as ProfilingSection[];
export const decodeSteps = decodeStepsJson as unknown as Partial<Record<"b1" | "b16", ProfilingTable & { file: string }>>;

export type LaunchGaps = { file: string; steps: number; kernels: number; span_us: number; busy_us: number; idle_pct: number; gaps: number; gap_us: number };

export function launchGaps(mode: "eager" | "graphs"): LaunchGaps {
  const gaps = (launchGapsJson as Partial<Record<"eager" | "graphs", LaunchGaps>>)[mode];
  if (!gaps) throw new Error(`results/${mode === "eager" ? "decode_step.md" : "decode_step_graphs.md"} has no decode launch gap line`);
  return gaps;
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
