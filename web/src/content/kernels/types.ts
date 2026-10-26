import type { Metric } from "@/content/build-log";
import { cellNumber, profilingTable } from "@/lib/data";
import type { ExcerptSpec } from "@/lib/excerpt";

export type Author = "joe" | "claude" | "both";

export const authorLabel: Record<Author, string> = {
  joe: "Joe",
  claude: "drafted with Claude",
  both: "Joe, fixes drafted with Claude",
};

export type KernelEntry = {
  date: string;
  prs: number[];
  branch: string;
  version: string;
  author: Author;
  title: string;
  text: string;
  metric?: Metric;
};

export type KernelSource = { path: string; label: string };

export type KernelStory = {
  slug: string;
  index: number;
  title: string;
  short: string;
  formula: string;
  headline: { value: string; label: string };
  sources: KernelSource[];
  tests: KernelSource[];
  testing: string;
  excerpts: ExcerptSpec[];
  timeline: KernelEntry[];
};

export function measured(section: string, match: Record<string, string>, column: string, table = 0): number {
  const { header, rows } = profilingTable(section, table);
  const col = header.indexOf(column);
  if (col < 0) throw new Error(`docs/profiling.md ${section}: no column "${column}"`);
  const row = rows.find((r) => Object.entries(match).every(([k, v]) => r[header.indexOf(k)] === v));
  const value = cellNumber(row?.[col]);
  if (value === undefined) throw new Error(`docs/profiling.md ${section}: no number for ${JSON.stringify(match)} in "${column}"`);
  return value;
}
