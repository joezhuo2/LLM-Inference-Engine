"use client";

import { useTheme } from "next-themes";
import type { ReactNode } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  LabelList,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { batches, environment, medianOf } from "@/lib/data";

type Colors = { accent: string; gray: string; grid: string; text: string; muted: string; surface: string };

function useColors(): Colors {
  const { resolvedTheme } = useTheme();
  return resolvedTheme === "light"
    ? { accent: "#15803d", gray: "#7c817c", grid: "rgba(20,20,10,0.08)", text: "#1a1c1a", muted: "#5f645f", surface: "#fbfaf6" }
    : { accent: "#4ade80", gray: "#8b918b", grid: "rgba(255,255,255,0.06)", text: "#e6e8e6", muted: "#8b918b", surface: "#111311" };
}

const series = [
  { id: "engine_graphs", label: "engine + CUDA Graphs", dash: undefined, emphasis: true },
  { id: "engine", label: "engine", dash: undefined, emphasis: false },
  { id: "llamacpp", label: "llama.cpp", dash: "6 4", emphasis: false },
  { id: "hf", label: "HuggingFace", dash: "2 4", emphasis: false },
] as const;

type Row = { batch: string } & Partial<Record<(typeof series)[number]["id"], number>>;

const rows: Row[] = batches.map((b) => {
  const row: Row = { batch: String(b) };
  for (const s of series) {
    const v = medianOf(s.id, b, "output_throughput");
    if (v !== undefined) row[s.id] = Math.round(v);
  }
  return row;
});

function ChartPage({ title, note, children }: { title: string; note: string; children: ReactNode }) {
  return (
    <div className="space-y-6">
      <h1 className="font-mono text-2xl">results</h1>
      <section className="space-y-3 rounded border border-border bg-surface p-4 sm:p-6">
        <h2 className="font-mono text-sm">{title}</h2>
        {children}
        <p className="font-mono text-xs text-muted">
          {note} {String(environment.gpu ?? "")}, {String(environment.workload ?? "")}.
        </p>
      </section>
    </div>
  );
}

function Legend({ only, bars = false }: { only?: string[]; bars?: boolean }) {
  return (
    <ul className="flex flex-wrap gap-x-5 gap-y-1 font-mono text-xs text-muted">
      {series
        .filter((s) => !only || only.includes(s.id))
        .map((s) => (
          <li key={s.id} className="flex items-center gap-2">
            {bars ? (
              <svg width="10" height="10" aria-hidden="true">
                <rect
                  width="10"
                  height="10"
                  rx="2"
                  style={{ fill: s.emphasis ? "var(--accent)" : "var(--text-muted)" }}
                  fillOpacity={s.id === "hf" ? 0.45 : 1}
                />
              </svg>
            ) : (
              <svg width="20" height="8" aria-hidden="true">
                <line
                  x1="0"
                  y1="4"
                  x2="20"
                  y2="4"
                  style={{ stroke: s.emphasis ? "var(--accent)" : "var(--text-muted)" }}
                  strokeWidth="2"
                  strokeDasharray={s.dash}
                />
              </svg>
            )}
            {s.label}
          </li>
        ))}
    </ul>
  );
}

function TooltipBox({
  active,
  label,
  payload,
  colors,
  unit,
}: {
  active?: boolean;
  label?: string | number;
  payload?: readonly { name?: unknown; value?: unknown; dataKey?: unknown }[];
  colors: Colors;
  unit: string;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div
      className="rounded border border-border px-3 py-2 font-mono text-xs shadow-lg"
      style={{ background: colors.surface, color: colors.text }}
    >
      <div style={{ color: colors.muted }}>batch {label}</div>
      {payload.map((p) => (
        <div key={String(p.dataKey)} className="flex justify-between gap-4 tabular-nums">
          <span style={{ color: colors.muted }}>{String(p.name ?? "")}</span>
          <span>
            {typeof p.value === "number" ? p.value.toFixed(unit === "x" ? 1 : 0) : "–"}
            {unit === "x" ? "x" : ""}
          </span>
        </div>
      ))}
    </div>
  );
}

const axisTick = (colors: Colors) => ({ fill: colors.muted, fontSize: 11, fontFamily: "var(--font-jetbrains)" });

export function Lines() {
  const colors = useColors();
  const first = rows[0]?.engine_graphs;
  const last = rows.at(-1)?.engine_graphs;
  return (
    <ChartPage title="output tokens per second by max batch size" note="Median of the recorded runs;">
      <Legend />
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={rows} margin={{ top: 16, right: 160, bottom: 8, left: 0 }}>
            <CartesianGrid stroke={colors.grid} vertical={false} />
            <XAxis dataKey="batch" tick={axisTick(colors)} axisLine={{ stroke: colors.grid }} tickLine={false} />
            <YAxis tick={axisTick(colors)} axisLine={false} tickLine={false} width={48} />
            <Tooltip
              cursor={{ stroke: colors.muted, strokeWidth: 1 }}
              content={(props) => <TooltipBox {...props} colors={colors} unit="tok/s" />}
            />
            {series.map((s) => (
              <Line
                key={s.id}
                dataKey={s.id}
                name={s.label}
                type="linear"
                connectNulls
                stroke={s.emphasis ? colors.accent : colors.gray}
                strokeWidth={2}
                strokeDasharray={s.dash}
                dot={{ r: 4, strokeWidth: 2, stroke: colors.surface, fill: s.emphasis ? colors.accent : colors.gray }}
                activeDot={{ r: 5, strokeWidth: 2, stroke: colors.surface }}
                isAnimationActive={false}
              >
                <LabelList
                  dataKey={s.id}
                  content={(p) => {
                    const { x, y, index } = p as { x?: number; y?: number; index?: number };
                    if (index !== rows.length - 1 || x === undefined || y === undefined) return null;
                    return (
                      <text x={Number(x) + 10} y={Number(y) + 4} fill={colors.text} fontSize={11} fontFamily="var(--font-jetbrains)">
                        {s.label}
                      </text>
                    );
                  }}
                />
              </Line>
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
      {first && last ? (
        <p className="font-mono text-xs text-muted">
          engine + CUDA Graphs: {first} tok/s at batch 1, {last} at batch 16 ({(last / first).toFixed(1)}x)
        </p>
      ) : null}
    </ChartPage>
  );
}

export function Bars() {
  const colors = useColors();
  const shown = ["engine_graphs", "llamacpp", "hf"] as const;
  return (
    <ChartPage title="output tokens per second at each max batch size" note="Median of the recorded runs; engine without graphs is in the results table;">
      <Legend only={[...shown]} bars />
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={rows} margin={{ top: 24, right: 8, bottom: 8, left: 0 }} barGap={2} barCategoryGap="22%">
            <CartesianGrid stroke={colors.grid} vertical={false} />
            <XAxis dataKey="batch" tick={axisTick(colors)} axisLine={{ stroke: colors.grid }} tickLine={false} />
            <YAxis tick={axisTick(colors)} axisLine={false} tickLine={false} width={48} />
            <Tooltip
              cursor={{ fill: colors.grid }}
              content={(props) => <TooltipBox {...props} colors={colors} unit="tok/s" />}
            />
            {shown.map((id) => {
              const s = series.find((x) => x.id === id)!;
              return (
                <Bar
                  key={id}
                  dataKey={id}
                  name={s.label}
                  fill={s.emphasis ? colors.accent : colors.gray}
                  fillOpacity={id === "hf" ? 0.45 : 1}
                  radius={[4, 4, 0, 0]}
                  isAnimationActive={false}
                >
                  {s.emphasis ? (
                    <LabelList dataKey={id} position="top" fill={colors.text} fontSize={10} fontFamily="var(--font-jetbrains)" />
                  ) : null}
                </Bar>
              );
            })}
          </BarChart>
        </ResponsiveContainer>
      </div>
    </ChartPage>
  );
}

export function Speedup() {
  const colors = useColors();
  const ratios = rows.map((r) => ({
    batch: r.batch,
    vs_llamacpp: r.engine_graphs && r.llamacpp ? r.engine_graphs / r.llamacpp : undefined,
    vs_hf: r.engine_graphs && r.hf ? r.engine_graphs / r.hf : undefined,
  }));
  const lines = [
    { key: "vs_llamacpp", label: "over llama.cpp", color: colors.accent, swatch: "var(--accent)", dash: undefined },
    { key: "vs_hf", label: "over HuggingFace", color: colors.gray, swatch: "var(--text-muted)", dash: "6 4" },
  ];
  return (
    <ChartPage title="engine + CUDA Graphs speedup over each baseline" note="Ratio of median throughputs; 1x means equal;">
      <ul className="flex flex-wrap gap-x-5 gap-y-1 font-mono text-xs text-muted">
        {lines.map((l) => (
          <li key={l.key} className="flex items-center gap-2">
            <svg width="20" height="8" aria-hidden="true">
              <line x1="0" y1="4" x2="20" y2="4" style={{ stroke: l.swatch }} strokeWidth="2" strokeDasharray={l.dash} />
            </svg>
            {l.label}
          </li>
        ))}
      </ul>
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={ratios} margin={{ top: 16, right: 190, bottom: 8, left: 0 }}>
            <CartesianGrid stroke={colors.grid} vertical={false} />
            <XAxis dataKey="batch" tick={axisTick(colors)} axisLine={{ stroke: colors.grid }} tickLine={false} />
            <YAxis
              tick={axisTick(colors)}
              axisLine={false}
              tickLine={false}
              width={40}
              tickFormatter={(v: number) => `${v}x`}
              domain={[0, "auto"]}
            />
            <ReferenceLine y={1} stroke={colors.muted} strokeDasharray="2 2" />
            <Tooltip
              cursor={{ stroke: colors.muted, strokeWidth: 1 }}
              content={(props) => <TooltipBox {...props} colors={colors} unit="x" />}
            />
            {lines.map((l) => (
              <Line
                key={l.key}
                dataKey={l.key}
                name={l.label}
                connectNulls
                stroke={l.color}
                strokeWidth={2}
                strokeDasharray={l.dash}
                dot={{ r: 4, strokeWidth: 2, stroke: colors.surface, fill: l.color }}
                isAnimationActive={false}
              >
                <LabelList
                  dataKey={l.key}
                  content={(p) => {
                    const { x, y, value, index } = p as { x?: number; y?: number; value?: number; index?: number };
                    if (index !== ratios.length - 1 || x === undefined || y === undefined || value === undefined) return null;
                    return (
                      <text x={Number(x) + 10} y={Number(y) + 4} fill={colors.text} fontSize={11} fontFamily="var(--font-jetbrains)">
                        {Number(value).toFixed(1)}x {l.label}
                      </text>
                    );
                  }}
                />
              </Line>
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </ChartPage>
  );
}
