import { execFileSync } from "node:child_process";
import { writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { number, parseLaunchGaps, parseProfiling } from "./profiling.mjs";

const web = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const out = resolve(web, "src/content/kernels/step-history.json");
const eagerFile = "results/decode_step.md";
const graphsFile = "results/decode_step_graphs.md";

const git = (...args) => execFileSync("git", args, { cwd: resolve(web, ".."), encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] });

function show(commit, path) {
  try {
    return git("show", `${commit}:${path}`);
  } catch {
    return null;
  }
}

function profile(text) {
  if (!text) return null;
  const gaps = parseLaunchGaps(text).decode;
  if (!gaps) return null;
  const sections = parseProfiling(text);
  const ops = sections.find((s) => s.id === "gpu-time-per-op-in-a-decode-step")?.tables[0];
  const kernels = sections.find((s) => s.id.startsWith("top-") && s.id.endsWith("-kernels"))?.tables[0];
  const sample = kernels?.rows.find((r) => r[0].includes("sample_kernel"));
  const tpot = /^- max_batch 1, median TPOT ([\d.]+) ms/m.exec(text);
  return {
    kernels: gaps.kernels,
    span_us: gaps.span_us,
    busy_us: gaps.busy_us,
    idle_pct: gaps.idle_pct,
    gaps: gaps.gaps,
    tpot_ms: tpot ? Number(tpot[1]) : null,
    sample_us: sample ? number(sample[kernels.header.indexOf("avg us")]) : null,
    ops: (ops?.rows ?? []).map((r) => [r[0], number(r[ops.header.indexOf("GPU us/step")])]),
  };
}

const commits = git("log", "--reverse", "--format=%h %ad", "--date=short", "--", eagerFile).trim().split("\n").map((line) => line.split(" "));
const stages = commits.flatMap(([commit, date]) => {
  const eager = profile(show(commit, eagerFile));
  if (!eager) return [];
  const graphs = profile(show(commit, graphsFile));
  return [{ commit, date, eager, graphs: graphs && { span_us: graphs.span_us, busy_us: graphs.busy_us, idle_pct: graphs.idle_pct, tpot_ms: graphs.tpot_ms } }];
});
if (stages.length === 0) throw new Error(`git log has no commits for ${eagerFile}: run this in a full clone`);
writeFileSync(out, `${JSON.stringify({ source: `git log -- ${eagerFile} ${graphsFile}`, stages }, null, 1)}\n`);
console.log(`wrote ${stages.length} decode step profiles to ${out}`);
