import { execFileSync } from "node:child_process";
import { copyFileSync, mkdirSync, readFileSync, readdirSync, writeFileSync, existsSync } from "node:fs";
import { basename, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { replayKv } from "./kv-replay.mjs";
import { parseModelShape, parseRooflineContext } from "./model-shape.mjs";
import { parseLaunchGaps, parseProfiling } from "./profiling.mjs";

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

function parseHardware(text) {
  const number = (pattern) => {
    const match = pattern.exec(text);
    return match ? Number(match[1]) : undefined;
  };
  return {
    gpu: text.split("\n")[0].trim(),
    compute_capability: number(/compute capability: ([\d.]+)/),
    sms: number(/SMs: (\d+)/),
    vram_gb: number(/VRAM: ([\d.]+) GB/),
    l2_kb: number(/L2: (\d+) KB/),
    shared_kb_per_sm: number(/shared mem per SM: (\d+) KB/),
  };
}

function readOnchip() {
  const path = join(results, "onchip/onchip.csv");
  if (!existsSync(path)) return [];
  const [header, ...rows] = parseCsv(readFileSync(path, "utf8"));
  return rows.map((row) => Object.fromEntries(header.map((key, i) => [key, value(row[i] ?? "")])));
}

function git(...args) {
  return execFileSync("git", args, { cwd: web, encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] }).trim();
}

function recordedIn(path) {
  try {
    if (git("rev-parse", "--is-shallow-repository") === "true") return null;
    return git("log", "-1", "--format=%h", "--", path) || null;
  } catch {
    return null;
  }
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
      commit: recordedIn(join(dir, name)),
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
const loadRun = /Engine runs measured ([\d-]+(?: to [\d-]+)?) at commit (\w+) on ([^;.]+)/.exec(resultsText);
const slo = /SLO: TTFT at most ([\d.]+) ms and TPOT at most ([\d.]+) ms/.exec(resultsText);
const resultsDoc = {
  sections: parseProfiling(resultsText),
  ceilings: ceilings ? { gbs: Number(ceilings[1]), tflops: Number(ceilings[2]) } : null,
  load: loadRun ? { date: loadRun[1], commit: loadRun[2], gpu: loadRun[3] } : null,
  slo: slo ? { ttft_ms: Number(slo[1]), tpot_ms: Number(slo[2]) } : null,
};
const modelShape = { ...parseModelShape(readFileSync(resolve(web, "../tests/unit/test_config.cpp"), "utf8")), roofline_context: parseRooflineContext(resultsText) };
const hardwarePath = resolve(web, "../docs/hardware.md");
const hardware = existsSync(hardwarePath) ? parseHardware(readFileSync(hardwarePath, "utf8")) : {};
const decodeSteps = {};
for (const [key, name] of [["b1", "decode_step.md"], ["b16", "decode_step_b16.md"], ["graphs", "decode_step_graphs.md"]]) {
  const path = join(results, name);
  if (!existsSync(path)) continue;
  const ops = parseProfiling(readFileSync(path, "utf8")).find((s) => s.id === "gpu-time-per-op-in-a-decode-step");
  if (ops?.tables[0]) decodeSteps[key] = { file: `results/${name}`, ...ops.tables[0] };
}
const launchGaps = {};
for (const [key, name] of [["eager", "decode_step.md"], ["graphs", "decode_step_graphs.md"], ["b16", "decode_step_b16.md"]]) {
  const path = join(results, name);
  if (!existsSync(path)) continue;
  const gaps = parseLaunchGaps(readFileSync(path, "utf8"));
  if (gaps.decode) launchGaps[key] = { file: `results/${name}`, ...gaps.decode };
}
const kvTrace = join(results, "traces/chat_b16_rate4.json");
const kvReplay = existsSync(kvTrace) ? replayKv(JSON.parse(readFileSync(kvTrace, "utf8")), "results/traces/chat_b16_rate4.json") : {};
writeFileSync(join(dataDir, "summaries.json"), JSON.stringify(summaries));
writeFileSync(join(dataDir, "environment.json"), JSON.stringify(environment));
writeFileSync(join(dataDir, "traces.json"), JSON.stringify(traces));
writeFileSync(join(dataDir, "profiling.json"), JSON.stringify(profiling));
writeFileSync(join(dataDir, "decode_steps.json"), JSON.stringify(decodeSteps));
writeFileSync(join(dataDir, "model_shape.json"), JSON.stringify(modelShape));
writeFileSync(join(dataDir, "hardware.json"), JSON.stringify(hardware));
writeFileSync(join(dataDir, "results_doc.json"), JSON.stringify(resultsDoc));
writeFileSync(join(dataDir, "launch_gaps.json"), JSON.stringify(launchGaps));
writeFileSync(join(dataDir, "kv_replay.json"), JSON.stringify(kvReplay));
const onchip = readOnchip();
writeFileSync(join(dataDir, "onchip.json"), JSON.stringify(onchip));
const kvDemoFile = join(results, "kv_demo.json");
const kvDemo = existsSync(kvDemoFile) ? JSON.parse(readFileSync(kvDemoFile, "utf8")) : {};
writeFileSync(join(dataDir, "kv_demo.json"), JSON.stringify(kvDemo));
const samplingLogitsFile = join(results, "sampling_logits.json");
const samplingLogits = existsSync(samplingLogitsFile) ? JSON.parse(readFileSync(samplingLogitsFile, "utf8")) : {};
writeFileSync(join(dataDir, "sampling_logits.json"), JSON.stringify(samplingLogits));
console.log(`sync-data: ${summaries.length} summaries, ${traces.length} traces, ${profiling.length} profiling sections, ${resultsDoc.sections.length} results sections, ${Object.keys(decodeSteps).length} decode step profiles, ${Object.keys(launchGaps).length} launch gap lines, KV replay ${kvReplay.steps?.length ?? 0} steps, KV demo ${kvDemo.requests?.length ?? 0} requests, sampling logits ${samplingLogits.contexts?.length ?? 0} contexts, ${onchip.length} on-chip rows, environment ${environment.commit ?? "missing"}`);
