import environmentJson from "@/data/environment.json";
import profilingJson from "@/data/profiling.json";
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

export function profilingTable(sectionId: string, index = 0): ProfilingTable {
  const table = profiling.find((s) => s.id === sectionId)?.tables[index];
  if (!table) throw new Error(`docs/profiling.md has no table ${index} in section ${sectionId}`);
  return table;
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
