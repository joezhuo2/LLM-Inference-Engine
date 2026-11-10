import { measured } from "@/content/kernels/types";
import { ceilings, hardware, modelShape } from "@/lib/data";
import { chapterNumber } from "./book-order";
import { kernelStory } from "./kernels";
import { flashLong, flashProfile, flashRoofline, flashShape } from "./kernels/flash-prefill";

export type Estimate = { id: string; title: string; question: string; first: string; sources: string[]; steps: string[]; answer: string; teaches: string; chapters: string[] };

const billions = (params: number) => `${Number((params / 1e9).toPrecision(3))} billion`;

const h100 = { vendor: "NVIDIA", model: "H100 SXM", bandwidthTBs: 3.35, bf16TFlops: 989, memoryGB: 80 };
const llama2 = { vendor: "Meta", model: "Llama 2 7B", params: 6.74e9, layers: 32, kvHeads: 32, headDim: 128 };
const llama3 = { vendor: "Meta", model: "Llama 3 8B", params: 8.03e9, layers: 32, kvHeads: 8, headDim: 128 };

export const published = {
  h100: { ...h100, source: `published spec: ${h100.vendor} ${h100.model}, ${h100.bandwidthTBs} TB/s HBM3, ${h100.bf16TFlops} TFLOP/s dense BF16, ${h100.memoryGB} GB` },
  llama2: { ...llama2, source: `published spec: ${llama2.vendor} ${llama2.model}, ${billions(llama2.params)} parameters, ${llama2.layers} layers, ${llama2.kvHeads} KV heads, head dimension ${llama2.headDim}` },
  llama3: { ...llama3, source: `published spec: ${llama3.vendor} ${llama3.model}, ${billions(llama3.params)} parameters, ${llama3.layers} layers, ${llama3.kvHeads} KV heads, head dimension ${llama3.headDim}` },
};

const setup = { bytesPerWeight: 2, context: 8192, batch: 32, prompt: 2048, computeShare: 0.6, spareBytes: 1e9, lowBits: 4, highBits: 8, bookRows: [1, 16] };

const sig = (x: number) => x.toLocaleString("en-US", { minimumSignificantDigits: 2, maximumSignificantDigits: 2 });
const n2 = (x: number) => x.toLocaleString("en-US", { maximumSignificantDigits: 2 });
const whole = (x: number) => Math.round(x).toLocaleString("en-US");
const gb = (bytes: number) => `${sig(bytes / 1e9)} GB`;
const gb3 = (bytes: number) => `${(bytes / 1e9).toLocaleString("en-US", { maximumSignificantDigits: 3 })} GB`;
const ms = (seconds: number) => `${sig(seconds * 1e3)} ms`;
const us = (seconds: number) => `${sig(seconds * 1e6)} µs`;
const kvPerToken = (layers: number, kvHeads: number, headDim: number) => 2 * layers * kvHeads * headDim * setup.bytesPerWeight;
const sharePct = Math.round(setup.computeShare * 100);
const spareGB = setup.spareBytes / 1e9;

const h = published.h100;
const a = published.llama2;
const b = published.llama3;
const m = modelShape();
const peak = ceilings();
const hw = hardware();

const bandwidth = h.bandwidthTBs * 1e12;
const roof = h.bf16TFlops * 1e12;
const memory = h.memoryGB * 1e9;
const ridge = roof / bandwidth;
const weights7 = a.params * setup.bytesPerWeight;
const floor7 = weights7 / bandwidth;
const rate7 = 1 / floor7;

const roofline = (rows: number, column: string) => measured("roofline", { "whole decode step": "engine + graphs", rows: String(rows) }, column, 1, "results");
const attentionUs = (rows: number) => measured("roofline", { op: "attention", rows: String(rows) }, "us/step", 0, "results");
const [rowsLow, rowsHigh] = setup.bookRows;
const bookLow = roofline(rowsLow, "TPOT us") * 1e-6;
const bookHigh = roofline(rowsHigh, "TPOT us") * 1e-6;
const bookShare = roofline(rowsLow, "% of roof");

const batch = setup.batch;
const perByte = (rows: number) => (rows * 2) / setup.bytesPerWeight;
const arithmetic32 = (2 * a.params * batch) / roof;
const step32 = Math.max(floor7, arithmetic32);
const rate32 = batch / step32;

const kv7 = kvPerToken(a.layers, a.kvHeads, a.headDim);
const headDimTiny = m.hidden / m.heads;
const kvTiny = kvPerToken(m.layers, m.kv_heads, headDimTiny);
const cache7 = kv7 * setup.context;
const cacheTiny = kvTiny * setup.context;

const free7 = memory - weights7;
const weights8 = b.params * setup.bytesPerWeight;
const free8 = memory - weights8;
const kv8 = kvPerToken(b.layers, b.kvHeads, b.headDim);
const cache8 = kv8 * setup.context;
const fit7 = free7 / cache7;
const fit8 = free8 / cache8;

const prefillFlop = 2 * a.params * setup.prompt;
const prefillRate = setup.computeShare * roof;
const prefillTime = prefillFlop / prefillRate;
const prefillPerToken = prefillTime / setup.prompt;

const vram = hw.vram_gb * 1e9;
const room = vram - setup.spareBytes;
const bytesFit = room / b.params;
const lowBytes = (b.params * setup.lowBits) / 8;
const floorRoom = room / (peak.gbs * 1e9);
const floorLow = lowBytes / (peak.gbs * 1e9);
const weightsAt = (bits: number) => {
  const bytes = (b.params * bits) / 8;
  return `${bits}-bit weights take ${gb(bytes)}, ${bytes <= room ? `which fits with ${gb(room - bytes)} to spare` : `${gb(bytes - room)} over the limit`}.`;
};

const pagedKv = kernelStory("paged-kv")?.title ?? "paged KV cache";

const fs = flashShape;
const fr = flashRoofline;

const fromBook = {
  roofline: "measured in this book: the roofline tables in docs/results.md",
  config: "this book's model: TinyLlama config.json",
  hardware: "this book's GPU: docs/hardware.md",
  bandwidth: "measured in this book: the bandwidth ceiling in docs/results.md",
};

export const estimatesTitle = "estimate it yourself";

export const estimatesIntro = [
  "Back-of-envelope problems that use this book's formulas on models and GPUs the book did not run, and on one change to its own kernel that nobody has built. Each needs a pencil and a calculator.",
  "Try each one first, then open the working. Answers are rounded to 2 significant figures, and each working ends with what it teaches.",
  "Published specs are typed in by hand and labelled published spec, with the vendor and the model. Every other number is calculated from those and from this book's own measurements, and is marked calculated or measured.",
];

export const estimates: Estimate[] = [
  {
    id: "decode-ceiling",
    title: "decode ceiling of a 7B model",
    question: `${a.model} has ${billions(a.params)} parameters. In BF16 on one ${h.model}, decoding one request at a time, what is the most tokens per second it can reach?`,
    first: `Work it out first: a decode step reads every weight once. How many bytes is that, and how long does it take to stream them at ${h.bandwidthTBs} TB/s?`,
    sources: [h.source, a.source, fromBook.roofline],
    steps: [
      `Bytes per token = parameters × bytes per weight = ${billions(a.params)} × ${setup.bytesPerWeight} = ${gb(weights7)} (calculated).`,
      `Floor per token = bytes ÷ bandwidth = ${gb(weights7)} ÷ ${h.bandwidthTBs} TB/s = ${ms(floor7)} (calculated).`,
      `Tokens per second = 1 ÷ ${ms(floor7)} = ${sig(rate7)} (calculated).`,
      `This counts only the weights. On this book's GPU the engine reaches ${bookShare}% of the measured bandwidth roof at batch 1 (measured), so expect a real engine to land under this ceiling.`,
    ],
    answer: `At best ${ms(floor7)} per token, or ${sig(rate7)} tokens per second (calculated).`,
    teaches: "Batch-1 decode speed is the weight bytes divided by the memory bandwidth, and the arithmetic does not enter.",
    chapters: [],
  },
  {
    id: "batch-32",
    title: "the same model at batch 32",
    question: `Now the same model and GPU decode ${batch} requests in one step. Is the step still memory bound, and what happens to the step time and the throughput?`,
    first: `Work it out first: how many FLOP does a projection do per byte it reads at ${batch} rows, and where is the ${h.model} ridge?`,
    sources: [h.source, a.source, fromBook.roofline],
    steps: [
      `FLOP per byte of the projections = rows × 2 FLOP ÷ ${setup.bytesPerWeight} bytes per weight = ${batch} × 2 ÷ ${setup.bytesPerWeight} = ${perByte(batch)} (calculated). Each weight does one multiply and one add per row.`,
      `The ${h.model} ridge = compute roof ÷ bandwidth = ${h.bf16TFlops} TFLOP/s ÷ ${h.bandwidthTBs} TB/s = ${sig(ridge)} FLOP per byte (calculated). ${perByte(batch)} is far below it, so the step is still memory bound.`,
      `Step time: reading the weights still takes ${ms(floor7)}, and the arithmetic takes 2 × ${billions(a.params)} × ${batch} ÷ ${h.bf16TFlops} TFLOP/s = ${ms(arithmetic32)}. The longer one sets the step: ${ms(step32)} (calculated).`,
      `Throughput = ${batch} tokens ÷ ${ms(step32)} = ${sig(rate32)} tokens per second, ${n2(rate32 / rate7)} times batch 1 (calculated).`,
      `Compare this book's RTX 5060, whole decode step with CUDA Graphs (measured): ${rowsLow} row takes ${ms(bookLow)} and ${rowsHigh} rows take ${ms(bookHigh)}, so the step is ${n2(bookHigh / bookLow)} times longer and throughput rises ${n2((rowsHigh * bookLow) / bookHigh)} times, not ${rowsHigh}. The estimate leaves out attention, which reads every request's cache and grows from ${ms(attentionUs(rowsLow) * 1e-6)} to ${ms(attentionUs(rowsHigh) * 1e-6)} per step (measured).`,
    ],
    answer: `Still memory bound, ${perByte(batch)} FLOP per byte against a ridge of ${sig(ridge)}. The step stays near ${ms(step32)}, so throughput is about ${n2(rate32 / rate7)} times higher, ${sig(rate32)} tokens per second (calculated).`,
    teaches: "Batching is nearly free while the step stays below the ridge, because every row shares one read of the weights.",
    chapters: [],
  },
  {
    id: "kv-per-token",
    title: "KV cache per token",
    question: `${a.model} has ${a.layers} layers, ${a.kvHeads} KV heads, head dimension ${a.headDim} and no grouped-query attention. How many bytes of KV cache does one token cost in BF16, how does that compare with TinyLlama, and how big is the cache at ${whole(setup.context)} tokens?`,
    first: "Work it out first: the cache keeps one K vector and one V vector per layer, for every KV head.",
    sources: [a.source, fromBook.config],
    steps: [
      `${a.model}, per token = 2 (K and V) × ${a.layers} layers × ${a.kvHeads} KV heads × ${a.headDim} values × ${setup.bytesPerWeight} bytes = ${whole(kv7)} bytes, about ${sig(kv7 / 1e6)} MB (calculated).`,
      `TinyLlama has ${m.layers} layers, ${m.kv_heads} KV heads and head dimension ${m.hidden} ÷ ${m.heads} = ${headDimTiny}. Per token = 2 × ${m.layers} × ${m.kv_heads} × ${headDimTiny} × ${setup.bytesPerWeight} = ${whole(kvTiny)} bytes, about ${sig(kvTiny / 1e3)} KB (calculated).`,
      `Ratio = ${whole(kv7)} ÷ ${whole(kvTiny)} = ${n2(kv7 / kvTiny)}. It splits into ${n2(a.kvHeads / m.kv_heads)} from GQA (${a.kvHeads} KV heads against ${m.kv_heads}), ${n2(a.headDim / headDimTiny)} from the wider head and ${n2(a.layers / m.layers)} from more layers (calculated).`,
      `At ${whole(setup.context)} tokens: ${a.model} ${whole(kv7)} × ${whole(setup.context)} = ${gb(cache7)}, and TinyLlama ${whole(kvTiny)} × ${whole(setup.context)} = ${gb(cacheTiny)} (calculated). TinyLlama's own window is ${whole(m.max_positions)} tokens, so that figure is only for comparison.`,
    ],
    answer: `${a.model}: about ${sig(kv7 / 1e6)} MB per token and ${gb(cache7)} at ${whole(setup.context)} tokens. TinyLlama: about ${sig(kvTiny / 1e3)} KB per token and ${gb(cacheTiny)} (calculated).`,
    teaches: "The cache grows with layers × KV heads × head dimension, so cutting the KV heads with grouped-query attention shrinks it directly.",
    chapters: [],
  },
  {
    id: "requests-that-fit",
    title: "how many requests fit",
    question: `An ${h.memoryGB} GB ${h.model} holds ${a.model} in BF16. If every request reserves a full ${whole(setup.context)}-token cache and activations are ignored, how many requests fit? Then repeat it for ${b.model} (${billions(b.params)} parameters, ${b.layers} layers, ${b.kvHeads} KV heads, head dimension ${b.headDim}), which uses grouped-query attention.`,
    first: "Work it out first: free memory is the GPU's memory minus the weights. Divide it by the cache of one request.",
    sources: [h.source, a.source, b.source],
    steps: [
      `${a.model}: weights ${gb(weights7)}, so ${h.memoryGB} GB − ${gb(weights7)} leaves ${gb3(free7)}. One request's cache is ${gb3(cache7)} (problem 3).`,
      `Requests = ${gb3(free7)} ÷ ${gb3(cache7)} = about ${fit7.toFixed(1)}, so ${Math.floor(fit7)} whole requests (calculated).`,
      `${b.model}: weights ${billions(b.params)} × ${setup.bytesPerWeight} = ${gb(weights8)}, leaving ${gb3(free8)}. Cache per token = 2 × ${b.layers} × ${b.kvHeads} × ${b.headDim} × ${setup.bytesPerWeight} = ${whole(kv8)} bytes, so ${gb3(cache8)} per request.`,
      `Requests = ${gb3(free8)} ÷ ${gb3(cache8)} = about ${fit8.toFixed(1)}, so ${Math.floor(fit8)} whole requests (calculated).`,
      `Chapter ${chapterNumber("paged-kv")}, ${pagedKv}, hands out cache in blocks as a request grows, so requests that stop short leave room for more. GQA cuts each token's cache ${n2(kv7 / kv8)} times, from ${a.kvHeads} KV heads to ${b.kvHeads}.`,
    ],
    answer: `About ${Math.floor(fit7)} requests with ${a.kvHeads} KV heads and about ${Math.floor(fit8)} with ${b.kvHeads}, ${n2(Math.floor(fit8) / Math.floor(fit7))} times as many (calculated).`,
    teaches: "The KV cache, not the weights, decides how many requests share a GPU, which is why paged KV and GQA matter.",
    chapters: ["paged-kv"],
  },
  {
    id: "prefill-time",
    title: "prefill time for a prompt",
    question: `A ${whole(setup.prompt)}-token prompt goes through ${a.model} on an ${h.model} in one prefill pass. Assume the projections run at ${sharePct}% of the compute roof. How long does prefill take, and how does the cost per token compare with decode in problem 1?`,
    first: `Work it out first: prefill does 2 FLOP per weight for each prompt token, and all ${whole(setup.prompt)} tokens share one read of the weights.`,
    sources: [h.source, a.source],
    steps: [
      `Work = 2 FLOP × parameters × prompt tokens = 2 × ${billions(a.params)} × ${whole(setup.prompt)} = ${sig(prefillFlop / 1e12)} TFLOP (calculated).`,
      `Speed = ${sharePct}% × ${h.bf16TFlops} TFLOP/s = ${sig(prefillRate / 1e12)} TFLOP/s. The ${sharePct}% is an assumption, not a measurement.`,
      `Time = ${sig(prefillFlop / 1e12)} TFLOP ÷ ${sig(prefillRate / 1e12)} TFLOP/s = ${ms(prefillTime)} (calculated).`,
      `Check: ${whole(setup.prompt)} rows give ${whole(perByte(setup.prompt))} FLOP per byte, above the ridge of ${sig(ridge)}, so prefill is compute bound. Reading the weights once takes only ${ms(floor7)} of it.`,
      `Per token = ${ms(prefillTime)} ÷ ${whole(setup.prompt)} = ${us(prefillPerToken)} per prompt token. Decode in problem 1 costs ${ms(floor7)} per token, ${n2(floor7 / prefillPerToken)} times more (calculated).`,
    ],
    answer: `About ${ms(prefillTime)} for the whole prompt, ${us(prefillPerToken)} per prompt token, against ${ms(floor7)} per decoded token (calculated, at the assumed ${sharePct}%).`,
    teaches: "Prefill is compute bound and decode is memory bound, so a token costs far less when many tokens share one read of the weights.",
    chapters: [],
  },
  {
    id: "fit-this-gpu",
    title: `does ${b.model} fit this GPU`,
    question: `${b.model} has ${billions(b.params)} parameters. Does it fit in BF16 in the memory of this book's GPU? What bytes per weight would leave ${spareGB} GB for the cache, and what is the decode floor at that size?`,
    first: `Work it out first: the weights can use at most the GPU's memory minus ${spareGB} GB. Divide that by the parameter count.`,
    sources: [b.source, fromBook.hardware, fromBook.bandwidth],
    steps: [
      `BF16 weights = ${billions(b.params)} × ${setup.bytesPerWeight} = ${gb(weights8)}. This GPU has ${sig(hw.vram_gb)} GB of VRAM, so the weights are ${n2(weights8 / vram)} times too big and do not fit.`,
      `Room for weights = ${sig(hw.vram_gb)} GB − ${spareGB} GB for the cache = ${gb(room)}.`,
      `Bytes per weight = ${gb(room)} ÷ ${billions(b.params)} = ${sig(bytesFit)} bytes, or ${sig(bytesFit * 8)} bits (calculated). ${weightsAt(setup.highBits)} ${weightsAt(setup.lowBits)}`,
      `Decode floor = weight bytes ÷ measured bandwidth. At the largest size that fits: ${gb(room)} ÷ ${peak.gbs} GB/s = ${ms(floorRoom)}. At ${setup.lowBits} bits: ${gb(lowBytes)} ÷ ${peak.gbs} GB/s = ${ms(floorLow)}, or ${sig(1 / floorLow)} tokens per second (calculated from the measured bandwidth).`,
      `This book does not cover quantization yet. The estimate only counts bytes, and real ${setup.lowBits}-bit files also store scale factors, so they are a little larger.`,
    ],
    answer: `No: ${gb(weights8)} of BF16 weights against ${sig(hw.vram_gb)} GB. They must average under ${sig(bytesFit)} bytes (${sig(bytesFit * 8)} bits) per weight to leave ${spareGB} GB, so ${setup.highBits}-bit is too big and about ${setup.lowBits}-bit fits. At ${setup.lowBits} bits the floor is ${ms(floorLow)} per token, ${sig(1 / floorLow)} tokens per second (calculated).`,
    teaches: "Bytes against VRAM decide whether a model runs at all, and fewer bytes per weight lower the decode floor in the same proportion.",
    chapters: [],
  },
  {
    id: "shared-kv",
    title: `${fs.group} query heads, one read of the keys`,
    question: `TinyLlama has ${fs.heads} query heads and ${fs.kvHeads} KV heads. This book's prefill kernel gives each query head its own warp, and each warp loads every tile of K and V for itself. How many FLOP does it do per byte of K and V it loads, what would that be if one thread block served the ${fs.group} query heads of a KV head and loaded each tile once, and where do both sit against this book's ridge?`,
    first: `Work it out first: one tile pair is ${fs.tile} query rows against ${fs.tile} keys of one head. Count its FLOP in the two products, then the bytes of its K tile and V tile in BF16.`,
    sources: [fromBook.config, fromBook.roofline, "this book's kernel: src/kernels/flash_prefill.cu", "measured in this book: Nsight Compute and Nsight Systems on that kernel, docs/profiling.md"],
    steps: [
      `FLOP per tile pair = 2 products (Q·K and P·V) × 2 FLOP × ${fs.tile} rows × ${fs.tile} keys × ${fs.headDim} dimensions = ${whole(fr.pairFlop)} (calculated).`,
      `Bytes per tile pair = 2 (K and V) × ${fs.tile} keys × ${fs.headDim} values × ${setup.bytesPerWeight} bytes = ${whole(fr.pairBytes)}, so ${whole(fr.pairFlop)} ÷ ${whole(fr.pairBytes)} = ${fr.now} FLOP per byte (calculated). It is the ${fs.tile} rows of the tile: every row that shares a load adds 1 FLOP per byte.`,
      `Check the count against the profiler: a ${flashLong}-token prompt has ${whole(fr.tilePairs)} tile pairs per head, × ${fs.heads} heads × ${whole(fr.pairBytes)} bytes = ${whole(fr.loadedMb)} MB per layer (calculated). Nsight Compute counts ${whole(flashProfile.step.requestedMb)} MB requested in 32-byte sectors (measured), so a thread uses about ${n2(fr.sectorBytes)} of the 32 bytes of each sector it asks for.`,
      `Shared by ${fs.group} heads, the FLOP stay and the bytes fall ${fs.group} times: ${fr.now} × ${fs.group} = ${fr.shared} FLOP per byte (calculated). The ridge is ${peak.tflops} TFLOP/s ÷ ${peak.gbs} GB/s = ${whole(fr.ridge)} FLOP per byte (measured ceilings), so the point crosses it.`,
      `What crossing buys: at ${fr.now} FLOP per byte DRAM could feed ${fr.now} × ${peak.gbs} GB/s = ${n2(fr.dramRoofNow)} TFLOP/s. The kernel already runs at ${fr.achieved.toFixed(1)} TFLOP/s (calculated: ${whole(fr.promptGflop)} GFLOP over ${fr.layers} layers in the measured ${fr.kernelMs} ms). It can because ${flashProfile.step.l1Hit}% of its loads are in L1 and DRAM runs at ${flashProfile.step.dram}% of its throughput (measured). The caches already serve the rereads, so DRAM bandwidth was not the limit. Whether the cache traffic is, the stall reasons would say, and they are not measured.`,
      `The ceiling: ${whole(fr.promptGflop)} GFLOP ÷ ${peak.tflops} TFLOP/s = ${n2(fr.floorMs)} ms against ${fr.kernelMs} ms measured. Against that measured ceiling no change to this kernel gains more than ${n2(fr.maxGain)} times, ${n2(fr.maxSavedMs)} ms or ${n2(fr.maxSavedPct)}% of the ${flashProfile.whole.span} ms that one prefill spans in Nsight Systems (calculated bound). The tensor cores run at ${flashProfile.step.tensor}% of their peak rate (measured).`,
      `Registers push the other way: ${fs.group} warps × 32 threads × ${flashProfile.launchAfter.registers} registers leave room for ${fr.sharedBlocksPerSm} blocks in the ${whole(fr.smRegisters)} registers of an SM, ${fr.warpsShared} warps where ${fr.warpsNow} run today (calculated, if the count stays at ${flashProfile.launchAfter.registers}).`,
    ],
    answer: `${fr.now} FLOP per byte loaded today and ${fr.shared} with one read for ${fs.group} heads, left and right of the ridge at ${whole(fr.ridge)} (calculated). The time it saves is not measured: nobody has built it, and the bound is ${n2(fr.maxGain)} times on the kernel.`,
    teaches: "Arithmetic intensity counts the rows that share a read. Crossing the ridge pays only if the bytes were the limit, and here the profiler says DRAM was not; what the extra cache traffic costs is not measured.",
    chapters: ["flash-prefill"],
  },
];
