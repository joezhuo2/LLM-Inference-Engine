import { measured } from "@/content/kernels/types";
import { cellNumber, decodeSteps, environment, launchGaps, medianOf, profilingTable, type GapProfile } from "@/lib/data";

export type Tone = "accent" | "series-2" | "series-3" | "baseline";

export type Series = { id: string; label: string; tone: Tone; dash?: string; values: (number | undefined)[]; notes?: (string | undefined)[] };

export type Provenance = { gpu: string; commit: string; date: string; source: string; command: string };

const gpu = String(environment.gpu ?? "unknown GPU").replace(/^NVIDIA GeForce /, "");

export const benchRun: Provenance = {
  gpu,
  commit: String(environment.commit ?? "unknown"),
  date: String(environment.date ?? "").slice(0, 10),
  source: "results/*_summary.csv, docs/results.md",
  command: "scripts/reproduce.sh",
};

export const workload = String(environment.workload ?? "unknown workload");

export const batchSizes = [1, 2, 4, 8, 16];

const benchSystems: { id: string; doc: string; label: string; tone: Tone; dash?: string }[] = [
  { id: "engine_graphs", doc: "engine + graphs", label: "engine + CUDA Graphs", tone: "accent" },
  { id: "engine", doc: "engine", label: "engine", tone: "series-2" },
  { id: "llamacpp", doc: "llama.cpp", label: "llama.cpp", tone: "baseline", dash: "6 4" },
  { id: "hf", doc: "HF transformers", label: "HuggingFace", tone: "baseline", dash: "2 4" },
];

const benchTable = profilingTable("throughput-and-latency", 0, "results");

function docCell(system: string, batch: number, column: string): number | undefined {
  const col = benchTable.header.indexOf(column);
  const row = benchTable.rows.find((r) => r[0] === system && r[1] === String(batch));
  return row ? cellNumber(row[col]) : undefined;
}

function checkedMedian(system: (typeof benchSystems)[number], batch: number, key: "output_throughput" | "tpot_s_p50"): number | undefined {
  const value = medianOf(system.id, batch, key);
  const column = key === "output_throughput" ? "output tok/s" : "TPOT p50 ms";
  const shown = value === undefined ? undefined : key === "output_throughput" ? Number(value.toFixed(1)) : Number((value * 1e3).toFixed(2));
  if (shown !== docCell(system.doc, batch, column)) throw new Error(`results/ CSVs and docs/results.md disagree on ${system.doc} batch ${batch} ${column}: ${shown} against ${docCell(system.doc, batch, column)}`);
  return shown;
}

function benchSeries(key: "output_throughput" | "tpot_s_p50"): Series[] {
  return benchSystems.map((system) => ({
    id: system.id,
    label: system.label,
    tone: system.tone,
    dash: system.dash,
    values: batchSizes.map((b) => checkedMedian(system, b, key)),
  }));
}

export const throughput = benchSeries("output_throughput");
export const tpot = benchSeries("tpot_s_p50");

export function valueAt(series: Series[], id: string, batch: number): number {
  const v = series.find((s) => s.id === id)?.values[batchSizes.indexOf(batch)];
  if (v === undefined) throw new Error(`no ${id} value at batch ${batch}`);
  return v;
}

export type Criterion = { id: string; name: string; target: string; measured: string; status: string; gauge?: Gauge };

export type Gauge =
  | { kind: "bar"; value: number; target: number; max: number; format: (v: number) => string; higher: boolean }
  | { kind: "cells"; exact: number; ties: number; total: number };

function match(text: string, pattern: RegExp, what: string): number[] {
  const m = pattern.exec(text);
  if (!m) throw new Error(`docs/results.md scoreboard: cannot read ${what} from "${text}"`);
  return m.slice(1).map(Number);
}

function gaugeFor(id: string, target: string, measuredText: string): Gauge | undefined {
  switch (id) {
    case "1a": {
      const [limit] = match(target, /error under ([\d.]+)/, "the error limit");
      const [, , error] = match(measuredText, /([\d.]+)% argmax over (\d+) positions, mean abs logit error ([\d.]+)/, "the teacher-forced result");
      return { kind: "bar", value: error, target: limit, max: limit * 1.2, format: String, higher: false };
    }
    case "1b": {
      const [exact, total, ties] = match(measuredText, /exact on (\d+) of (\d+); the other (\d+)/, "the free-running result");
      return { kind: "cells", exact, ties, total };
    }
    case "2": {
      const [pct] = match(target, /at least (\d+)%/, "the bandwidth target");
      const [, share] = match(measuredText, /(\d+) GB\/s = (\d+)%/, "the bandwidth share");
      return { kind: "bar", value: share, target: pct, max: 100, format: (v) => `${v}%`, higher: true };
    }
    case "3": {
      const [times] = match(target, /at least ([\d.]+)x/, "the throughput target");
      const [, , ratio] = match(measuredText, /(\d+) vs (\d+) tokens\/s = ([\d.]+)x/, "the throughput ratio");
      return { kind: "bar", value: ratio, target: times, max: 12, format: (v) => `${v.toFixed(1)}x`, higher: true };
    }
    default:
      return undefined;
  }
}

export const scoreboard: Criterion[] = [
  ...profilingTable("mvp-scoreboard", 0, "results").rows.map(([id, name, target, measuredText, status]) => ({
    id,
    name,
    target,
    measured: measuredText,
    status,
    gauge: gaugeFor(id, target, measuredText),
  })),
  { id: "4", name: "Nsight Compute before and after per kernel", target: "a number for every optimized kernel", measured: "kernel table in docs/results.md and docs/profiling.md", status: "tracked" },
  { id: "5", name: "Reproducible from a clean clone", target: "one command", measured: "scripts/reproduce.sh", status: "tracked" },
];

export const nearTies = scoreboard.find((c) => c.gauge?.kind === "cells")?.gauge as Extract<Gauge, { kind: "cells" }> | undefined;

export const contextPrompts = [64, 512, 1024, 1984];

const wired = "split-attention-in-the-forward-pass-branch-42";
const graphsSection = "split-attention-in-cuda-graphs-branch-43";

function runs(section: string, prompt: number, column: string): { mean: number; text: string } {
  const { header, rows } = profilingTable(section);
  const row = rows.find((r) => r[header.indexOf("max batch")] === "1" && r[header.indexOf("prompt")] === prompt.toLocaleString("en-US"));
  const cell = row?.[header.indexOf(column)];
  const parts = (cell ?? "").split("/").map((p) => cellNumber(p.trim()));
  if (parts.length !== 2 || parts.some((p) => p === undefined)) throw new Error(`docs/profiling.md ${section}: no two runs for prompt ${prompt} in "${column}"`);
  const [a, b] = parts as number[];
  return { mean: (a + b) / 2, text: `${a.toFixed(2)} / ${b.toFixed(2)}` };
}

function contextSeries(id: string, label: string, tone: Tone, section: string, column: string, dash?: string): Series {
  const cells = contextPrompts.map((p) => runs(section, p, column));
  return { id, label, tone, dash, values: cells.map((c) => Number(c.mean.toFixed(3))), notes: cells.map((c) => `runs ${c.text}`) };
}

export const contextTpot: Series[] = [
  contextSeries("graphs-split", "graphs, one per split count", "accent", graphsSection, "this branch graphs"),
  contextSeries("eager-split", "split, eager", "series-2", graphsSection, "eager"),
  contextSeries("graphs-v2", "graphs, v2 captured", "series-3", graphsSection, "main graphs"),
  contextSeries("eager-v2", "v2, eager", "baseline", wired, "v2 eager", "6 4"),
];

export const splitsAt = contextPrompts.map((p) => measured(graphsSection, { "max batch": "1", prompt: p.toLocaleString("en-US") }, "splits"));

export const contextRun: Provenance = {
  gpu,
  commit: "e298faf (branch 42), 7770e78 (branch 43)",
  date: "2026-10-23 and 24",
  source: "docs/profiling.md, split attention in the forward pass and in CUDA Graphs",
  command: "build/release/bin/engine bench --model models/tinyllama --max-batch 1 --num-requests 1 --prompt-len 1984 --output-len 64 --warmup 1 --graphs",
};

export const profileRun: Provenance = { ...benchRun, source: "results/decode_step.md, decode_step_graphs.md", command: "scripts/profile.sh nsys decode_step --max-batch 1, then with --graphs (both run by scripts/reproduce.sh)" };

export const profileB16Run: Provenance = { ...benchRun, source: "results/decode_step.md, decode_step_b16.md", command: "scripts/profile.sh nsys decode_step_b16 --max-batch 16 (run by scripts/reproduce.sh)" };

export type Segment = { id: string; label: string; group: "gemm" | "attention" | "other" | "idle" | "graph"; us: number; detail: string };

export type StepRow = { id: GapProfile; label: string; span: number; busy: number; idlePct: number; gaps: number; gapUs: number; kernels: number; segments: Segment[] };

const gemmOps: Record<string, string> = { gate_up: "gate/up", down: "down", qkv: "QKV", o_proj: "output", lm_head: "LM head" };

function opSegments(table: NonNullable<(typeof decodeSteps)["b1"]>): Segment[] {
  const col = (name: string) => table.header.indexOf(name);
  const ops = table.rows.map((r) => ({ op: r[0], kernels: r[col("kernels/step")], usText: r[col("GPU us/step")], us: cellNumber(r[col("GPU us/step")]) ?? 0, pct: r[col("% of kernel time")] }));
  const detail = (o: (typeof ops)[number]) => `${o.usText} µs/step, ${o.pct}% of kernel time, ${o.kernels} kernels per step`;
  const big = ops.filter((o) => o.op in gemmOps || o.op === "attention" || o.op === "forward_decode");
  const rest = ops.filter((o) => !big.includes(o));
  const segments: Segment[] = big.map((o) => ({
    id: o.op,
    label: o.op === "forward_decode" ? "one graph replay" : (gemmOps[o.op] ?? o.op),
    group: o.op === "attention" ? "attention" : o.op === "forward_decode" ? "graph" : "gemm",
    us: o.us,
    detail: detail(o),
  }));
  if (rest.length) {
    const us = rest.reduce((s, o) => s + o.us, 0);
    segments.push({
      id: "other",
      label: `${rest.length} other ops`,
      group: "other",
      us,
      detail: rest.map((o) => `${o.op} ${o.usText} µs`).join(", "),
    });
  }
  return segments;
}

function stepRow(id: GapProfile, label: string, table: (typeof decodeSteps)["b1"]): StepRow {
  if (!table) throw new Error(`results/ has no op table for the ${label} profile`);
  const gaps = launchGaps(id);
  const idle = gaps.span_us - gaps.busy_us;
  return {
    id,
    label,
    span: gaps.span_us,
    busy: gaps.busy_us,
    idlePct: gaps.idle_pct,
    gaps: gaps.gaps,
    gapUs: gaps.gap_us,
    kernels: gaps.kernels,
    segments: [
      ...opSegments(table),
      { id: "idle", label: "idle", group: "idle", us: idle, detail: `median span ${gaps.span_us.toLocaleString("en-US")} µs minus median busy ${gaps.busy_us.toLocaleString("en-US")} µs, ${gaps.idle_pct}% of the span, ${gaps.gaps} gaps over ${gaps.gap_us} µs` },
    ],
  };
}

export const stepRows = {
  eager: stepRow("eager", "batch 1, eager", decodeSteps.b1),
  graphs: stepRow("graphs", "batch 1, CUDA Graphs", decodeSteps.graphs),
  b16: stepRow("b16", "batch 16, eager", decodeSteps.b16),
};

export function segmentUs(row: StepRow, id: string): number {
  const s = row.segments.find((x) => x.id === id);
  if (!s) throw new Error(`no ${id} in the ${row.label} profile`);
  return s.us;
}

const roofTable = profilingTable("roofline", 1, "results");
const roofCol = (name: string) => roofTable.header.indexOf(name);
export const roofRows = [1, 2, 4, 8, 16];

function roofSeries({ doc, ...s }: (typeof benchSystems)[number]): Series & { tflops: (number | undefined)[]; intensity: (number | undefined)[] } {
  const rows = roofRows.map((r) => roofTable.rows.find((x) => x[0] === doc && x[1] === String(r)));
  const at = (name: string) => rows.map((r) => (r ? cellNumber(r[roofCol(name)]) : undefined));
  const gbs = at("GB/s");
  const pct = at("% of roof");
  return {
    ...s,
    values: at("TFLOP/s"),
    tflops: at("TFLOP/s"),
    intensity: at("FLOP/byte"),
    notes: rows.map((r, i) => (r ? `${gbs[i]} GB/s, ${pct[i]}% of roof` : undefined)),
  };
}

export const roofline = benchSystems.map(roofSeries);

export const roofIntensity = roofRows.map((_, i) => {
  const x = roofline.map((s) => s.intensity[i]).find((v) => v !== undefined);
  if (x === undefined) throw new Error(`docs/results.md roofline: no point at ${roofRows[i]} rows`);
  return x;
});

export function roofPct(id: string, rows: number): number {
  const v = cellNumber(roofTable.rows.find((x) => x[0] === benchSystems.find((s) => s.id === id)?.doc && x[1] === String(rows))?.[roofCol("% of roof")]);
  if (v === undefined) throw new Error(`docs/results.md roofline: no % of roof for ${id} at ${rows} rows`);
  return v;
}

export const gemmRoofRange = (() => {
  const ops = Object.keys(gemmOps).map((op) => measured("roofline", { op, rows: "1" }, "% of roof", 0, "results"));
  return { min: Math.min(...ops), max: Math.max(...ops) };
})();

export const v2Blocks = measured("split-attention-occupancy-branch-66", { kernel: "v2", splits: "1" }, "grid blocks");

export function opRank(row: StepRow, id: string): number {
  const ops = row.segments.filter((s) => s.group !== "idle" && s.group !== "other").sort((a, b) => b.us - a.us);
  return ops.findIndex((s) => s.id === id) + 1;
}
