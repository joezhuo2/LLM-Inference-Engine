import { decodeSteps, kvDemo, modelShape, samplingLogits } from "@/lib/data";
import { type KernelStory, measured } from "./types";
import { eagerStage, timeFmt } from "./where-the-time-went";

export const tokenFmt = timeFmt;

const shape = modelShape();
const tokenizerSection = "tokenizer-on-the-host-branch-148";
const host = (stage: string) => measured(tokenizerSection, { stage }, "median us");

const demo = kvDemo().requests.find((r) => r.name === "capital");
const logits = samplingLogits().contexts.find((c) => c.name === "capital");
if (!demo || !logits) throw new Error("kv_demo.json or sampling_logits.json has no request named capital");

const table = decodeSteps.b1;
if (!table) throw new Error("results/decode_step.md has no batch-1 table");
const column = (name: string) => {
  const i = table.header.indexOf(name);
  if (i < 0) throw new Error(`results/decode_step.md has no column "${name}"`);
  return i;
};
const usCol = column("GPU us/step");
const countCol = column("kernels/step");
const opUs = (op: string) => Number(table.rows.find((r) => r[0] === op)?.[usCol] ?? 0);
const opCount = (op: string) => Number(table.rows.find((r) => r[0] === op)?.[countCol] ?? 0);

const chosen = logits.tokens[0];
const chosenAt = demo.output_tokens.indexOf(chosen);
if (chosenAt < 1) throw new Error(`the recorded answer does not contain the top token ${chosen}`);
const piece = (t: string) => t.replace(/▁/g, " ");

export const journey = {
  user: demo.prompt,
  answer: demo.answer,
  promptTokens: demo.prompt_tokens,
  promptCount: demo.prompt_tokens.length,
  outputTokens: demo.output_tokens,
  input: demo.output_tokens[chosenAt - 1],
  inputText: piece(demo.output_tokens[chosenAt - 1]),
  inputId: 338,
  inputPosition: demo.prompt_tokens.length + chosenAt - 1,
  chosen,
  chosenText: piece(chosen),
  chosenId: 13476,
  soFar: logits.answer_so_far,
  top: logits.tokens.slice(0, 5).map((token, i) => ({ token, logit: logits.logits[i] })),
  lead: logits.logits[0] - logits.logits[1],
  hidden: shape.hidden,
  vocab: shape.vocab,
  layers: shape.layers,
  kvValues: 2 * shape.kv_heads * (shape.hidden / shape.heads),
};

export const hostTimes = {
  template: host("chat template"),
  encode: host("encode"),
  answer: host("decode, whole answer"),
  push: host("decode, last token of the answer"),
  answerTokens: measured(tokenizerSection, { stage: "decode, whole answer" }, "tokens"),
};

const layerDefs = [
  { ops: ["add2_norm1", "norm1", "add2"], label: "add + RMSNorm", slug: "add-rmsnorm", does: "adds the last layer's result to the stream and rescales the row" },
  { ops: ["qkv"], label: "QKV projection", slug: "gemm", does: "one matrix product makes this token's query, key and value" },
  { ops: ["rope"], label: "RoPE", slug: "rope", does: "turns the query and the key by the token's position" },
  { ops: ["write_kv"], label: "write_kv", slug: "paged-kv", does: "stores the key and the value in the paged cache" },
  { ops: ["attention"], label: "paged attention", slug: "paged-attention", does: "scores the query against every cached key and mixes the values" },
  { ops: ["o_proj"], label: "output projection", slug: "gemm", does: "one matrix product brings the mix back to the stream's width" },
  { ops: ["add1_norm2"], label: "add + RMSNorm", slug: "add-rmsnorm", does: "adds attention's result to the stream and rescales the row" },
  { ops: ["gate_up"], label: "gate/up projection", slug: "gemm", does: "the largest matrix product: widens the row for the MLP" },
  { ops: ["swiglu"], label: "SwiGLU", slug: "swiglu", does: "gates one half of the wide row with the other" },
  { ops: ["down"], label: "down projection", slug: "gemm", does: "narrows the row again; the next layer's add takes it in" },
];

export const layerOps = layerDefs.map((d) => {
  const step = d.ops.reduce((sum, op) => sum + opUs(op), 0);
  return { label: d.label, slug: d.slug, does: d.does, ops: d.ops, step, layer: step / shape.layers };
});
export const layersUs = layerOps.reduce((sum, o) => sum + o.step, 0);
export const layerUs = layersUs / shape.layers;
export const layerKernels = layerDefs.flatMap((d) => d.ops).reduce((sum, op) => sum + opCount(op), 0);
export const widest = layerOps.reduce((a, b) => (b.step > a.step ? b : a));
export const gemmUs = layerOps.filter((o) => o.slug === "gemm").reduce((sum, o) => sum + o.step, 0);

const s = eagerStage;
const tpotUs = (s.tpot ?? 0) * 1e3;
const sampleUs = s.sampleUs ?? 0;

export const trip = [
  { id: "embedding", label: "embedding", where: "GPU", us: opUs("embedding"), does: `looks up row ${journey.inputId} of the embedding table: ${tokenFmt(shape.hidden)} numbers` },
  { id: "layers", label: `${shape.layers} layers`, where: "GPU", us: layersUs, does: `${tokenFmt(layerKernels)} kernels: every layer reads its weights once` },
  { id: "final-norm", label: "final norm", where: "GPU", us: opUs("final_norm"), does: "one RMSNorm over the last layer's row" },
  { id: "lm-head", label: "LM head", where: "GPU", us: opUs("lm_head"), does: `one matrix product: ${tokenFmt(shape.hidden)} numbers become ${tokenFmt(shape.vocab)} logits` },
  { id: "gaps", label: "launch gaps", where: "GPU", us: s.span - s.busy, does: "a few microseconds before each kernel, with no kernel running" },
  { id: "sample", label: "sampler", where: "GPU", us: sampleUs, does: `one kernel reads the ${tokenFmt(shape.vocab)} logits and keeps one ID` },
  { id: "host", label: "host, between steps", where: "host", us: tpotUs - s.span - sampleUs, does: "read the ID back, check for the end, schedule and build the next step" },
];
export const tripSum = trip.reduce((sum, t) => sum + t.us, 0);
export const tokenUs = tpotUs;
export const stage = (id: string) => {
  const t = trip.find((x) => x.id === id);
  if (!t) throw new Error(`no stage ${id} in the trip of a token`);
  return t;
};
export const share = (us: number) => (us / tpotUs) * 100;
export const shareText = (us: number) => (share(us) < 0.1 ? "under 0.1%" : `${tokenFmt(share(us), share(us) < 10 ? 1 : 0)}%`);
export const stepProfile = { kernels: s.kernels, span: s.span, busy: s.busy, tpotMs: s.tpot ?? 0, commit: s.commit, date: s.date };

export const stations = [
  { id: "tokenizer", label: "tokenizer", where: "host", takes: `"${demo.prompt}"`, gives: `${demo.prompt_tokens.length} IDs`, time: `${tokenFmt(hostTimes.encode, 1)} µs, once per request` },
  { id: "embedding", label: "embedding", where: "GPU", takes: `ID ${journey.inputId}, "${journey.inputText.trim()}"`, gives: `${tokenFmt(shape.hidden)} numbers`, time: `${tokenFmt(stage("embedding").us, 1)} µs` },
  { id: "layers", label: `${shape.layers} layers`, where: "GPU", takes: `${tokenFmt(shape.hidden)} numbers`, gives: `${tokenFmt(shape.hidden)} numbers, and ${shape.layers} cache entries`, time: `${tokenFmt(layersUs)} µs` },
  { id: "final-norm", label: "final norm", where: "GPU", takes: `${tokenFmt(shape.hidden)} numbers`, gives: `${tokenFmt(shape.hidden)} numbers`, time: `${tokenFmt(stage("final-norm").us, 1)} µs` },
  { id: "lm-head", label: "LM head", where: "GPU", takes: `${tokenFmt(shape.hidden)} numbers`, gives: `${tokenFmt(shape.vocab)} logits`, time: `${tokenFmt(stage("lm-head").us)} µs` },
  { id: "sample", label: "sampler", where: "GPU", takes: `${tokenFmt(shape.vocab)} logits`, gives: `ID ${journey.chosenId}`, time: `${tokenFmt(sampleUs, 1)} µs` },
  { id: "text", label: "back to text", where: "host", takes: `ID ${journey.chosenId}`, gives: `"${journey.chosenText}"`, time: `${tokenFmt(hostTimes.push, 1)} µs` },
];

export const oneToken: KernelStory = {
  slug: "one-token",
  index: 15,
  book: true,
  about: true,
  title: "one token, end to end",
  short: `One token followed through the whole engine, from the text of a question to the ID ${journey.chosenId}, with the measured time of every stage on the way.`,
  formula: "text → IDs → 22 layers → logits → one ID → text",
  headline: {
    value: `${tokenFmt(share(layersUs))}%`,
    label: `of a batch-1 token is spent in the ${shape.layers} layers; the tokenizer and the sampler together are under 1%`,
  },
  sources: [
    { path: "src/tokenizer/tokenizer.cpp", label: "encode and decode" },
    { path: "src/tokenizer/chat.cpp", label: "the chat template" },
    { path: "src/tokenizer/text_stream.cpp", label: "IDs back to text, one at a time" },
    { path: "src/runtime/paged_runner.cpp", label: "the forward pass" },
    { path: "src/runtime/sampler.cpp", label: "logits to one ID" },
    { path: "scripts/tokenizer_bench.cpp", label: "the host stages, timed" },
    { path: "results/decode_step.md", label: "the decode step, op by op" },
  ],
  tests: [],
  testing: "",
  excerpts: [],
  measurements: [
    { section: tokenizerSection, table: 0, caption: "the two host stages for this chapter's prompt and answer, median of 2,001 calls", command: "/tmp/tokenizer_bench models/tinyllama (build line in docs/profiling.md)" },
    { doc: "results", section: "gpu-time-per-op-in-a-decode-step", table: 0, caption: "the batch-1 decode step, GPU time per op, launched one by one", command: "scripts/profile.sh nsys decode_step --max-batch 1 (run by scripts/reproduce.sh)" },
  ],
  timeline: [],
};
