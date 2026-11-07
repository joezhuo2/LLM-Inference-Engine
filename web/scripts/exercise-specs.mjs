import { execFileSync } from "node:child_process";
import { writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

const web = resolve(fileURLToPath(import.meta.url), "../..");
const out = resolve(web, "src/content/exercise-specs.json");

export const specSources = [
  { id: "block-manager", start: "135b3bd", file: "src/kv/block_manager.cpp", title: "BlockManager" },
  { id: "write-kv", start: "37a4712", file: "src/kernels/write_kv.cu", title: "write_kv" },
  { id: "scheduler", start: "f0d6978", file: "src/sched/scheduler.cpp", title: "Scheduler" },
  { id: "paged-attention", start: "f840222", file: "src/kernels/paged_attention.cu", title: "paged_attention_decode" },
  { id: "attention-v2", start: "7fd2390", file: "src/kernels/paged_attention_v2.cu", title: "paged_attention_decode_v2" },
  { id: "split-attention", start: "e2a6214", file: "src/kernels/paged_attention_split.cu", title: "paged_attention_decode_split" },
  { id: "add-rmsnorm", start: "8daf61d", file: "src/kernels/add_rmsnorm.cu", title: "add_rmsnorm" },
  { id: "rope", start: "efecbaa", file: "src/kernels/rope_fast.cu", title: "rope (fast)" },
  { id: "swiglu", start: "efecbaa", file: "src/kernels/swiglu_fast.cu", title: "swiglu (fast)" },
  { id: "cuda-graphs", start: "b8fef56", file: "src/runtime/decode_graphs.cpp", title: "DecodeGraphs" },
  { id: "chunked-prefill", start: "2d9ee3b", file: "src/sched/scheduler.cpp", title: "Scheduler::schedule_chunked" },
];

export function parseSpecs(source) {
  const blocks = {};
  for (const m of source.matchAll(/\/\*\s*\nTODO\(JOE\): ([^\n]+)\n([\s\S]*?)\*\//g)) {
    const paragraphs = m[2].split(/\n\s*\n/).map((p) => p.replace(/\s*\n\s*/g, " ").trim()).filter(Boolean);
    blocks[m[1].trim()] = paragraphs;
  }
  return blocks;
}

export function splitLabel(paragraph) {
  const m = /^([A-Z][A-Za-z ,'-]{1,48}?)\. (.*)$/s.exec(paragraph);
  return m ? { label: m[1], text: m[2] } : { label: "", text: paragraph };
}

function main() {
  const specs = {};
  for (const s of specSources) {
    const source = execFileSync("git", ["show", `${s.start}:${s.file}`], { cwd: web, encoding: "utf8" });
    const blocks = parseSpecs(source);
    const paragraphs = blocks[s.title];
    if (!paragraphs) throw new Error(`${s.start}:${s.file} has no TODO(JOE): ${s.title} (found ${Object.keys(blocks).join(", ")})`);
    specs[s.id] = paragraphs.map(splitLabel);
  }
  writeFileSync(out, `${JSON.stringify(specs, null, 1)}\n`);
  console.log(`wrote ${Object.keys(specs).length} specs to ${out}`);
}

if (process.argv[1] === fileURLToPath(import.meta.url)) main();
