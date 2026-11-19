"use client";

import Link from "next/link";
import { budget } from "@/content/kernels/memory-budget";
import { arithmetic, bf16Bits, bf16Floor, e4m3, fp8Floor, kvPool, kvToken, mathUs, matrices, normalReach, q4Floor, qFmt, qGb, qMs, readWeights, ridge, rowsBig, zeroReach } from "@/content/kernels/quantization";
import { ceilings } from "@/lib/data";
import { StickyStory, type StoryStep } from "../sticky-story";
import { FloorFigure, LadderFigure, PlanFigure, PoolFigure, RidgeFigure, RoundFigure } from "./figures";

const fmt = qFmt;
const peak = ceilings();
const chapter = (slug: string, text: string) => (
  <Link href={`/kernels/${slug}`} className="link-underline text-accent">
    {text}
  </Link>
);
const scaleCount = matrices.reduce((s, m) => s + m.rows * m.count, 0);
const scalePct = ((scaleCount * 4) / readWeights) * 100;
const topBig = arithmetic[arithmetic.length - 1];

const steps: StoryStep[] = [
  {
    id: "floor",
    title: "the floor moves with the bytes",
    body: (
      <>
        <p>
          The floor is a division: the bytes a token reads over the bandwidth that delivers them. A token reads {fmt(readWeights / 1e9, 2)} billion weights, the four matrices of every layer and the LM head (the embedding gives one row). In BF16 that is {qGb(bf16Floor.gb * 1e9)}, and at the measured {peak.gbs} GB/s the floor is <span className="text-accent">{qMs(bf16Floor.ms)}</span>.
        </p>
        <p>
          Store the same weights in FP8 and a token reads {qGb(fp8Floor.gb * 1e9)}: the floor falls to <span className="text-accent">{qMs(fp8Floor.ms)}</span>. At 4 bits it is {qGb(q4Floor.gb * 1e9)} and {qMs(q4Floor.ms)}. Nothing in the division is about kernels. Half the bytes per weight is half the floor.
        </p>
        <p>
          That is the floor, not the token. A token today takes more than its floor, and the parts that are not weights, attention reading the cache, the norms, the gaps between kernels, do not shrink with them. So FP8 weights would gain less than half. How much less is not measured: the kernels are stubs.
        </p>
      </>
    ),
  },
  {
    id: "arithmetic",
    title: "the arithmetic stays free",
    body: (
      <>
        <p>
          Reading fewer bytes would not help if the GPU then waited on the math. At batch 1 each weight does one multiply and one add, 2 FLOP. Per byte read that is 1 FLOP in BF16, 2 in FP8 and 4 at 4 bits. {chapter("foundations", "Foundations")} put this GPU&apos;s ridge, where the math starts to set the time, at <span className="text-accent">{fmt(ridge)} FLOP per byte</span>.
        </p>
        <p>
          Every width is far to the left of it. Even at {rowsBig} rows, the engine&apos;s largest batch, 4-bit weights do {fmt(topBig.intensity)} FLOP per byte, {fmt(topBig.pct)}% of the ridge. All the math of a batch-1 token takes {fmt(mathUs(1))} µs at the measured peak, against {qMs(fp8Floor.ms)} of reading in FP8.
        </p>
        <p>
          So turning a narrow weight back into a number as it is used costs time only if it adds a trip to memory. A product that reads the byte and converts it as it multiplies, without writing a wider copy anywhere, stays bound by the read. One caution: the ridge uses the tensor cores&apos; measured peak, and the conversion runs on the ordinary cores, whose rate is not measured here.
        </p>
      </>
    ),
  },
  {
    id: "scale",
    title: "a scale per row fits the weights into eight bits",
    body: (
      <>
        <p>
          E4M3 holds {e4m3.perPower} values in every power of two from 2^{e4m3.minNormalExp} up to 256, then 256 to {e4m3.max} in steps of {e4m3.topStep}. <span className="text-accent">{e4m3.max} is the largest</span>: there is no infinity, and anything larger is held at {e4m3.max}. Below 2^{e4m3.minNormalExp}, seven more values sit 2^{e4m3.minSubExp} apart, and anything under 2^{e4m3.minSubExp - 1} becomes 0.
        </p>
        <p>
          A scale says what the byte means. The stored byte is e4m3(w ÷ s), and the weight read back is the byte&apos;s value × s. The plan gives every output row of a weight matrix its own scale, s = the row&apos;s largest |w| ÷ {e4m3.max}, so the largest weight of each row lands exactly on {e4m3.max} and none saturates. It costs one FP32 number per row: {fmt(scaleCount)} for TinyLlama, {fmt(scalePct, 2)}% of the FP8 weights&apos; bytes.
        </p>
        <p>
          Dividing by s slides a whole row along the ladder at once. A weight more than {fmt(normalReach)} times smaller than its row&apos;s largest lands among the coarse values at the bottom, and one about {fmt(zeroReach)} times smaller becomes 0. How many of TinyLlama&apos;s weights land there is not measured here.
        </p>
      </>
    ),
  },
  {
    id: "rounding",
    title: "rounding costs at most half a step",
    body: (
      <>
        <p>
          Between 1 and 2, E4M3 has {e4m3.perPower} values: 1, 1.125, 1.25 and so on to 1.875. A weight between two of them rounds to the nearer one, so it moves by at most half a step, 0.0625: <span className="text-accent">1/{2 * e4m3.perPower} of the power of two</span> it sits in. Every power of two has the same {e4m3.perPower} steps, so the bound holds across the normal range.
        </p>
        <p>
          BF16 has {bf16Bits.perPower} steps in the same place, so it moves a weight by at most 1/{2 * bf16Bits.perPower}. Going from BF16 to FP8 makes each weight&apos;s possible error {bf16Bits.perPower / e4m3.perPower} times larger. The rounding happens once, when the weights are written, and costs nothing per token.
        </p>
        <p>
          What it does to the model&apos;s answers is another question, and this chapter cannot answer it. Small errors in many weights partly cancel and partly add up through {budget.layers} layers. The spec leaves it to the wiring: the logit error against the reference, before and after. Until then the effect on TinyLlama&apos;s output is not measured.
        </p>
      </>
    ),
  },
  {
    id: "kv",
    title: "an FP8 cache holds twice the tokens",
    body: (
      <>
        <p>
          The weights are not the only bytes. {chapter("memory-budget", "The memory budget")} counted {fmt(kvToken.bf16)} bytes for each cached token in BF16. The cache can be stored as E4M3 the same way, and then a token costs <span className="text-accent">{fmt(kvToken.fp8)} bytes</span>.
        </p>
        <p>
          The pool keeps its size, {qGb(kvPool.bytes)}, so it holds twice the blocks: {fmt(kvPool.fp8)} tokens where it held {fmt(kvPool.bf16)}, {fmt(kvPool.contextsFp8)} full contexts of {fmt(budget.fullContext)} tokens where {fmt(kvPool.contextsBf16)} fit. For TinyLlama memory was not the limit, so on this GPU that is room nobody needs yet. For a larger model whose pool is small, it is a longer context or a larger batch.
        </p>
        <p>
          The spec is plain that it will not make attention faster yet: attention is far from its bandwidth roof at batch {budget.maxBatch}, so this is about room first. It uses one scale for all keys and one for all values, and the wiring starts from a scale of 1, which reaches {e4m3.max}, far above what TinyLlama&apos;s keys and values reach.
        </p>
      </>
    ),
  },
  {
    id: "plan",
    title: "the plan: a dtype and three kernels",
    body: (
      <>
        <p>
          Three commits put the plan in the tree, for Joe to implement. DType::F8E4M3 is one byte per element, named F8_E4M3 as safetensors spells it, and it is the one piece that is done. The rest are stubs that throw, each with a spec and tests written first.
        </p>
        <p>
          gemm_fp8 is the weight product: y = row_scale × Σ x · w, with x in BF16, w in FP8 and one scale per output row, the sum in FP32 and the result rounded to BF16 once. write_kv_fp8 rounds each step&apos;s new keys and values into an FP8 cache, to the nearest value and held at ±{e4m3.max}. paged_attention_decode_fp8 reads them back as byte × scale, exact in FP32, and is decode attention from there, held to the tolerance of the other paged kernels. Each rejects a BF16 tensor where FP8 is due.
        </p>
        <p>
          None of it is wired. A later branch writes each row&apos;s scale once at load, gives the cache a dtype, puts the FP8 path behind a flag and measures bytes per step, step time and logit error. Real engines already do this: llama.cpp&apos;s GGUF files store weights at 2 to 8 bits in small blocks, each block with its own scale, and vLLM serves FP8 weights, an FP8 KV cache, and 4-bit AWQ and GPTQ checkpoints.
        </p>
      </>
    ),
  },
];

const captions = [
  `calculated: ${fmt(readWeights)} weights read per token, from TinyLlama's config.json, at 2, 1 and 0.5 bytes, ÷ the measured ${peak.gbs} GB/s (results.md, Roofline); not measured: the kernels are stubs`,
  `calculated: 2 FLOP per weight per row ÷ bytes per weight, counting the weights' bytes only; ridge = the measured ${peak.tflops} TFLOP/s ÷ ${peak.gbs} GB/s (results.md, Roofline); not measured: the kernels are stubs`,
  "calculated from the E4M3 definition in the spec of write_kv_fp8 and the per-row scale the spec of gemm_fp8 plans; one cell per power of two, not to scale in value; not measured: the kernels are stubs",
  "calculated from the formats: 3 mantissa bits in E4M3, 7 in BF16; the effect on TinyLlama's output is not measured",
  `calculated: the ${fmt(budget.poolBlocks)} blocks the engine allocated (measured, the memory budget) at ${fmt(kvToken.bf16)} and ${fmt(kvToken.fp8)} bytes per token, one block kept back as the engine does; not measured: the kernels are stubs`,
  "from the specs in src/kernels/gemm_fp8.cu and src/kernels/fp8_kv.cu and the dtype in include/engine/core/tensor.h; block widths illustrative; nothing here runs yet",
];

export const quantSteps = steps;
export const quantCaptions = captions;

const figures = [FloorFigure, RidgeFigure, LadderFigure, RoundFigure, PoolFigure, PlanFigure];

export function QuantStory() {
  return (
    <StickyStory
      steps={steps}
      caption={(i) => captions[i]}
      figure={(i) => {
        const Figure = figures[i] ?? PlanFigure;
        return <Figure />;
      }}
    />
  );
}
