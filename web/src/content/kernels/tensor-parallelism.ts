import { cellNumber, modelShape, profilingTable } from "@/lib/data";
import type { KernelStory } from "./types";

const forwardRows = profilingTable("tensor-parallelism-a-forward-pass-on-the-shards", 0, "results");
function forwardCell(layout: string, column: string) {
  const cell = forwardRows.rows.find((r) => r[0] === layout)?.[forwardRows.header.indexOf(column)];
  if (cell === undefined) throw new Error(`results.md forward pass on the shards: no "${column}" for ${layout}`);
  return cell;
}
function forwardNumber(layout: string, column: string) {
  const v = cellNumber(forwardCell(layout, column));
  if (v === undefined) throw new Error(`results.md forward pass on the shards: "${column}" for ${layout} is not a number`);
  return v;
}
export type ShardedForward = { ranks: number[]; msPerToken: number[]; wholeMsPerToken: number; maxLogitDiff: number; sameTokens: boolean };
export function shardedForward(): ShardedForward {
  const ranks = [1, 2, 4];
  const layout = (n: number) => (n === 1 ? "1 rank" : `${n} ranks`);
  const run = {
    ranks,
    msPerToken: ranks.map((n) => forwardNumber(layout(n), "ms per decode step")),
    wholeMsPerToken: forwardNumber("whole model, ModelRunner", "ms per decode step"),
    maxLogitDiff: Math.max(...ranks.map((n) => forwardNumber(layout(n), "largest logit difference"))),
    sameTokens: ranks.every((n) => { const [same, of] = forwardCell(layout(n), "greedy tokens equal").split(" of "); return same === of; }),
  };
  if (forwardNumber("1 rank", "largest logit difference") !== 0 || !run.sameTokens || run.msPerToken[2] <= run.wholeMsPerToken) throw new Error("results.md forward pass on the shards: the chapter says one rank matches bit for bit, the greedy tokens are the same and the ranks on one GPU are slower than the whole model; rewrite it");
  return run;
}
export function shardedForwardSentences(): string[] {
  const run = shardedForward();
  return [
    `With one rank the logits are the whole model's bit for bit. With 2 and 4 ranks they came within ${run.maxLogitDiff} of them, and every greedy token was the same (measured).`,
    `A token took ${run.ranks.map((n, i) => `${tpFmt(run.msPerToken[i], 2)} ms with ${n} ${n === 1 ? "rank" : "ranks"}`).join(", ")}, against ${tpFmt(run.wholeMsPerToken, 2)} ms for the whole model (measured). The ranks take turns on the one GPU, so more ranks is slower here.`,
  ];
}

export const tpFmt = (v: number, d = 0) => v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
export const mib = (bytes: number, d = 1) => `${tpFmt(bytes / 1048576, d)} MiB`;

const shape = modelShape();
const headDim = shape.hidden / shape.heads;
const bf16 = 2;

export type TpMatrix = { id: string; label: string; rows: number; cols: number; cut: "rows" | "cols"; bands: { label: string; rows: number }[] };

export const tpShape = { ...shape, headDim, group: shape.heads / shape.kv_heads, sums: 2 * shape.layers };

export const attentionMatrices: TpMatrix[] = [
  { id: "wqkv", label: "fused QKV weight", rows: (shape.heads + 2 * shape.kv_heads) * headDim, cols: shape.hidden, cut: "rows", bands: [{ label: "Q", rows: shape.heads * headDim }, { label: "K", rows: shape.kv_heads * headDim }, { label: "V", rows: shape.kv_heads * headDim }] },
  { id: "wo", label: "output projection wo", rows: shape.hidden, cols: shape.heads * headDim, cut: "cols", bands: [{ label: "", rows: shape.hidden }] },
];
export const mlpMatrices: TpMatrix[] = [
  { id: "wgu", label: "fused gate and up weight", rows: 2 * shape.intermediate, cols: shape.hidden, cut: "rows", bands: [{ label: "gate", rows: shape.intermediate }, { label: "up", rows: shape.intermediate }] },
  { id: "wdown", label: "wdown", rows: shape.hidden, cols: shape.intermediate, cut: "cols", bands: [{ label: "", rows: shape.hidden }] },
];
export const layerMatrices = [...attentionMatrices, ...mlpMatrices];

export const rankChoices = [1, 2, 4, 8, 16].filter((n) => shape.kv_heads % n === 0 && shape.heads % n === 0 && shape.intermediate % n === 0);
export const mostRanks = rankChoices[rankChoices.length - 1];

const layerBytes = shape.layers * layerMatrices.reduce((s, m) => s + m.rows * m.cols, 0) * bf16;
const tableBytes = shape.vocab * shape.hidden * bf16;
const normBytes = (2 * shape.layers + 1) * shape.hidden * bf16;

export function rankShare(n: number) {
  const layers = layerBytes / n;
  const total = layers + 2 * tableBytes + normBytes;
  return { n, layers, embed: tableBytes, head: tableBytes, norms: normBytes, total, pct: (total / (layerBytes + 2 * tableBytes + normBytes)) * 100, together: (n * total) / (layerBytes + 2 * tableBytes + normBytes), qHeads: shape.heads / n, kvHeads: shape.kv_heads / n, channels: shape.intermediate / n };
}
export const fullBytes = rankShare(1).total;

export const rankCopies = (n: number) => (n === 0 ? 3 + 9 * shape.layers : 3 + shape.layers * (2 + 5 + 2 * shape.hidden));

const section = "tensor-parallelism-the-weights-of-one-rank";
const layouts = profilingTable(section, 0, "results");
const history = profilingTable(section, 1, "results");
const layoutName = (n: number) => (n === 0 ? "full model" : `rank 0 of ${n}`);
const cell = (table: typeof layouts, layout: string, col: number) => {
  const v = cellNumber(table.rows.find((r) => r[0] === layout)?.[col]);
  if (v === undefined) throw new Error(`results.md Tensor parallelism: no "${table.header[col]}" for ${layout}`);
  return v;
};
const col = (name: string) => {
  const i = layouts.header.indexOf(name);
  if (i < 0) throw new Error(`results.md Tensor parallelism: no column "${name}"`);
  return i;
};
const commitOf = (i: number) => {
  const m = /^upload ms, (one call per copy|one tensor per chunk) \((\w+)\)$/.exec(history.header[i] ?? "");
  if (!m) throw new Error(`results.md Tensor parallelism: the second table's column ${i} is not "upload ms, one call per copy (commit)" or "one tensor per chunk (commit)"`);
  return m[2];
};

export const measuredRanks = [0, ...rankChoices].map((n) => ({
  n,
  layout: layoutName(n),
  copies: cell(layouts, layoutName(n), col("copies")),
  mib: cell(layouts, layoutName(n), col("MiB on the device")),
  ms: cell(layouts, layoutName(n), col("upload ms")),
  calc: rankShare(Math.max(1, n)),
}));
export const loadHistory = {
  before: commitOf(1),
  after: commitOf(2),
  rows: measuredRanks.map((r) => ({ n: r.n, layout: r.layout, copies: r.copies, perCopy: cell(history, r.layout, 1), packed: cell(history, r.layout, 2) })),
};

for (const r of measuredRanks) {
  if (Math.abs(r.calc.total / 1048576 - r.mib) > 0.1) throw new Error(`results.md Tensor parallelism: ${r.layout} holds ${r.mib} MiB, but the chapter calculates ${(r.calc.total / 1048576).toFixed(1)} MiB from the model's shape; rewrite the chapter's memory spread`);
  if (r.copies !== rankCopies(r.n)) throw new Error(`results.md Tensor parallelism: ${r.layout} has ${r.copies} copies, the chapter counts ${rankCopies(r.n)} (one per row of a column slice)`);
}
export const whole = measuredRanks[0];
export const split = measuredRanks.filter((r) => r.n > 1);
export const firstSplit = split[0];
export const slowRanks = loadHistory.rows.filter((r) => r.n > 0);
export const perCopySpan = { min: Math.min(...slowRanks.map((r) => r.perCopy)), max: Math.max(...slowRanks.map((r) => r.perCopy)) };
if (!firstSplit || firstSplit.calc.pct <= 100 / firstSplit.n || split.some((r) => r.ms >= whole.ms) || slowRanks.some((r) => r.perCopy < 10 * r.packed || r.perCopy < whole.ms))
  throw new Error("results.md Tensor parallelism: the chapter says a rank holds more than its share, loads faster than the whole model, and took ten times as long with one call per copy; rewrite the memory and load spreads");

export const tensorParallelism: KernelStory = {
  slug: "tensor-parallelism",
  index: 18,
  book: true,
  about: true,
  title: "tensor parallelism",
  short: `A model too big for one GPU is cut so that every layer's weight matrices are split across several GPUs, and the parts are added back together twice per layer. Here the cut is built and measured on one GPU: a rank of ${firstSplit.n} holds ${tpFmt(firstSplit.calc.pct)}% of the model, not ${tpFmt(100 / firstSplit.n)}%.`,
  formula: "a layer's output = the sum over ranks of (x × its rows) × its columns",
  headline: {
    value: `${tpFmt(firstSplit.calc.pct)}%`,
    label: `of the model's bytes in a rank of ${firstSplit.n}, not ${tpFmt(100 / firstSplit.n)}%: the embedding and the LM head stay whole in every rank (measured)`,
  },
  sandbox: {
    title: "cut it yourself",
    intro: `Pick a number of ranks and a rank. The four weight matrices of a layer are cut the way plan_shard_layout() in src/loader/tensor_parallel.cpp cuts them, and the bytes of one rank are calculated from TinyLlama's shapes. For ${split.map((r) => r.n).join(" and ")} ranks the calculation matches what the engine loaded, to a tenth of a MiB.`,
  },
  sources: [
    { path: "src/loader/tensor_parallel.cpp", label: "plan_shard_layout: the rows and columns of one rank" },
    { path: "include/engine/loader/tensor_parallel.h", label: "which tensors are cut, and how" },
    { path: "src/runtime/weights.cpp", label: "upload_weights: the copies of one tensor in one chunk" },
    { path: "scripts/shard_load_bench.cu", label: "the load of one rank, timed" },
  ],
  tests: [{ path: "tests/unit/test_tensor_parallel.cpp", label: "the rows and columns of every rank, and partial outputs that add up to the full layer" }],
  testing: "",
  excerpts: [
    {
      file: "src/loader/tensor_parallel.cpp",
      from: "    const int64_t shard_q_heads = config.heads / num_ranks;",
      to: "                            dst_layer.wo, rank * shard_q_rows, shard_q_rows);",
      caption: "The cut of attention: a rank's query and K/V heads are rows of the fused QKV weight and columns of wo",
    },
  ],
  measurements: [
    { doc: "results", section, table: 0, caption: "what one rank holds and how long it takes to load, every rank loaded onto the one GPU of this laptop", command: "scripts/shard_load_bench.cu, one process per layout" },
    { doc: "results", section, table: 1, caption: "the load of one rank with one call per copy and with the copies of one tensor packed into one chunk", command: "scripts/shard_load_bench.cu at each commit" },
  ],
  timeline: [],
};
