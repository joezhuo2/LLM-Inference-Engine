import { copyFileSync, mkdirSync, readFileSync, readdirSync, writeFileSync, existsSync } from "node:fs";
import { basename, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseProfiling } from "./profiling.mjs";

const web = resolve(fileURLToPath(import.meta.url), "../..");
const results = resolve(web, "../results");
const dataDir = join(web, "src/data");
const tracesOut = join(web, "public/traces");

function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"' && text[i + 1] === '"') {
        field += '"';
        i++;
      } else if (c === '"') {
        quoted = false;
      } else {
        field += c;
      }
    } else if (c === '"') {
      quoted = true;
    } else if (c === ",") {
      row.push(field);
      field = "";
    } else if (c === "\n" || c === "\r") {
      if (c === "\r" && text[i + 1] === "\n") i++;
      row.push(field);
      if (row.some((f) => f !== "")) rows.push(row);
      row = [];
      field = "";
    } else {
      field += c;
    }
  }
  row.push(field);
  if (row.some((f) => f !== "")) rows.push(row);
  return rows;
}

function value(text) {
  return text !== "" && Number.isFinite(Number(text)) ? Number(text) : text;
}

function readSummaries() {
  const pattern = /^(.+)_b(\d+)_r(\d+)_summary\.csv$/;
  const summaries = [];
  for (const name of readdirSync(results).sort()) {
    const match = pattern.exec(name);
    if (!match) continue;
    const [header, ...rows] = parseCsv(readFileSync(join(results, name), "utf8"));
    for (const row of rows) {
      const entry = { system: match[1], batch: Number(match[2]), run: Number(match[3]), file: name };
      header.forEach((key, i) => {
        entry[key] = value(row[i] ?? "");
      });
      summaries.push(entry);
    }
  }
  return summaries;
}

function readEnvironment() {
  const path = join(results, "environment.txt");
  if (!existsSync(path)) return {};
  const env = {};
  for (const line of readFileSync(path, "utf8").split("\n")) {
    const eq = line.indexOf("=");
    if (eq > 0) env[line.slice(0, eq).trim()] = value(line.slice(eq + 1).trim());
  }
  return env;
}

function syncTraces() {
  mkdirSync(tracesOut, { recursive: true });
  const dir = join(results, "traces");
  if (!existsSync(dir)) return [];
  const index = [];
  for (const name of readdirSync(dir).sort()) {
    if (!name.endsWith(".json")) continue;
    copyFileSync(join(dir, name), join(tracesOut, name));
    const trace = JSON.parse(readFileSync(join(dir, name), "utf8"));
    const steps = trace.steps ?? [];
    index.push({
      file: `/traces/${basename(name)}`,
      version: trace.version,
      meta: trace.meta,
      requests: (trace.requests ?? []).length,
      steps: steps.length,
      duration_s: steps.length ? steps[steps.length - 1].end_s : 0,
      max_batch_seen: steps.reduce((m, s) => Math.max(m, s.seqs.length), 0),
    });
  }
  return index;
}

mkdirSync(dataDir, { recursive: true });
const summaries = readSummaries();
const environment = readEnvironment();
const traces = syncTraces();
const profilingPath = resolve(web, "../docs/profiling.md");
const profiling = existsSync(profilingPath) ? parseProfiling(readFileSync(profilingPath, "utf8")) : [];
const resultsPath = resolve(web, "../docs/results.md");
const resultsText = existsSync(resultsPath) ? readFileSync(resultsPath, "utf8") : "";
const ceilings = /Ceilings: ([\d.]+) GB\/s and ([\d.]+) TFLOP\/s/.exec(resultsText);
const resultsDoc = { sections: parseProfiling(resultsText), ceilings: ceilings ? { gbs: Number(ceilings[1]), tflops: Number(ceilings[2]) } : null };
const decodeSteps = {};
for (const [key, name] of [["b1", "decode_step.md"], ["b16", "decode_step_b16.md"]]) {
  const path = join(results, name);
  if (!existsSync(path)) continue;
  const ops = parseProfiling(readFileSync(path, "utf8")).find((s) => s.id === "gpu-time-per-op-in-a-decode-step");
  if (ops?.tables[0]) decodeSteps[key] = { file: `results/${name}`, ...ops.tables[0] };
}
writeFileSync(join(dataDir, "summaries.json"), JSON.stringify(summaries));
writeFileSync(join(dataDir, "environment.json"), JSON.stringify(environment));
writeFileSync(join(dataDir, "traces.json"), JSON.stringify(traces));
writeFileSync(join(dataDir, "profiling.json"), JSON.stringify(profiling));
writeFileSync(join(dataDir, "decode_steps.json"), JSON.stringify(decodeSteps));
writeFileSync(join(dataDir, "results_doc.json"), JSON.stringify(resultsDoc));
console.log(`sync-data: ${summaries.length} summaries, ${traces.length} traces, ${profiling.length} profiling sections, ${resultsDoc.sections.length} results sections, ${Object.keys(decodeSteps).length} decode step profiles, environment ${environment.commit ?? "missing"}`);
