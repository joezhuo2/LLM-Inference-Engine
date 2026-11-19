"use client";

import Link from "next/link";
import { throughput, valueAt } from "@/content/results";
import { flashShape } from "@/content/kernels/flash-prefill";
import { eagerStage, firstStage, load, loadTitle, loadVerdict, next, noisy, nowStage, prefill, smallOf, stageById, stepStages, timeFmt, token } from "@/content/kernels/where-the-time-went";
import { published } from "@/content/published";
import { bigSide, engineSide, scaleSources } from "@/content/scale";
import { modelShape } from "@/lib/data";
import { about, num } from "@/lib/round";
import { ScaleFigure } from "../scale-figure";
import { StickyStory, type StoryStep } from "../sticky-story";
import { FloorFigure, LoadFigure, NextFigure, PrefillFigure, StagesFigure } from "./figures";

const fmt = timeFmt;
const layers = modelShape().layers;
const first = firstStage;
const now = nowStage;
const fast = stageById("fast");
const v2 = stageById("v2");
const tuned = stageById("tuned");
const batchRows = 16;
const batchGain = valueAt(throughput, "engine_graphs", batchRows) / valueAt(throughput, "engine_graphs", 1);
const firstFloorPct = token.firstMs === null ? 0 : (token.floorMs / token.firstMs) * 100;
const sentence =(text: string) => text.charAt(0).toUpperCase() + text.slice(1);
const upTo = (rate: number | undefined) => (rate === undefined ? "at no rate offered" : `up to an offered ${rate}`);
const whole = load.gaps.find((g) => g.id === "engine");
const chunkedGap = load.gaps.find((g) => g.id === "chunked");
const h100 = published.h100;
const l3 = published.llama3;
const aboveFloor = token.ms - token.floorMs;
const bigShare = (100 * bigSide.readFloorMs) / (bigSide.readFloorMs + aboveFloor);
const flashClause = load.hasFlash === null ? "" : load.hasFlash ? ", with the tiled prefill kernel of the FlashAttention prefill chapter" : ", before the tiled prefill kernel became the default, so its prompts still ran through the decode attention";

const steps: StoryStep[] = [
  {
    id: "decode",
    title: "every kernel shrank its slice, and the GEMMs kept the step",
    body: (
      <>
        <p>
          The first profile of a batch-1 decode step, on {first.date}, had {first.kernels} kernels in a span of {about(first.span)} µs. The projections and the LM head, all cuBLAS, took <span className="text-accent">{about(first.groups.gemm)} µs</span> of kernel time, attention {about(first.groups.attention)} µs, and the norms, adds, RoPE, SwiGLU and the cache write {about(smallOf(first))} µs together.
        </p>
        <p>
          Then the chapters landed. {fast && `Fusing each add with the norm after it took ${first.kernels - fast.kernels} launches out of the step. `}
          {fast && v2 && `Attention v2 took attention from ${about(fast.groups.attention)} to ${about(v2.groups.attention)} µs, the largest single cut. `}
          {v2 && tuned && `Tuning add + RMSNorm took the small kernels from ${about(smallOf(v2))} to ${about(smallOf(tuned))} µs. `}
          CUDA Graphs removed the gaps: {about(eagerStage.idle)} µs of idle time in a step became <span className="text-accent">{about(now.idle)} µs</span>.
        </p>
        <p>
          The GEMMs did not move, because no chapter changed them: {about(first.groups.gemm)} µs then and {about(now.groups.gemm)} µs now, a difference the clocks explain. Their share of the kernel time rose from {fmt(token.firstGemmShare)}% to {fmt(token.gemmShare)}% only because everything around them shrank. Sampling is outside the span: one greedy sample kernel after each step, {now.sampleUs} µs.
        </p>
        {noisy.length > 0 && (
          <p>
            One capture reads oddly: on {noisy[0].date} the gaps were {about(noisy[0].idlePct)}% of the span, where today&apos;s are {about(eagerStage.idlePct)}%. Gaps depend on what else the machine is doing. Kernel time is the steadier number.
          </p>
        )}
      </>
    ),
  },
  {
    id: "floor",
    title: "most of a token is the read of the weights",
    body: (
      <>
        <p>
          Foundations gave the floor. A decoded token reads every weight once, {about(token.weightGb)} GB, and this GPU&apos;s memory delivers {about(token.peakGbs)} GB/s at best, so a token cannot take less than <span className="text-accent">{about(token.floorMs)} ms</span>. No kernel changes that number.
        </p>
        <p>
          Today&apos;s token takes {about(token.ms)} ms with the step replayed from a graph. <span className="text-accent">{fmt(token.floorPct)}%</span> of it is that read at the peak rate. {token.aboveMs > 0 ? `The kernels spend ${about(token.aboveMs)} ms above the floor` : `The kernels are busy for ${about(-token.aboveMs, 2, "")} ms less than the floor, so they read a little faster than the measured peak`}, the launch gaps {about(token.gapsMs)} ms, and {about(token.hostMs)} ms, {fmt(token.hostPct)}%, pass between the end of one step and the start of the next.
        </p>
        <p>
          So for one request at a time, a faster kernel has {token.aboveMs > 0 ? `${about(token.aboveMs)} ms` : "nothing"} left to win. The larger gains in this book came from sharing the read: with {batchRows} rows in a step the same {about(token.weightGb)} GB serve {batchRows} tokens, and the engine makes <span className="text-accent">{fmt(batchGain, 1)} times</span> the tokens per second it makes alone.
        </p>
      </>
    ),
  },
  {
    id: "prefill",
    title: "the first token of a long prompt was the slow part",
    body: (
      <>
        <p>
          Decode started close to its floor{token.firstMs !== null && `: the first profiled token already spent ${fmt(firstFloorPct)}% of its time on the read of the weights`}. Prefill was far from it. One {prefill.long}-token prompt took <span className="text-warn">{about(prefill.paged)} ms</span> to its first token, against {about(prefill.llama)} ms for llama.cpp, because every prompt token ran through the decode attention kernel and walked its whole context alone. The cost per prompt token climbed from {about(prefill.pagedPerTokenShort)} to {about(prefill.pagedPerTokenLong, 2, "")} ms as the prompt grew.
        </p>
        <p>
          The first FlashAttention kernel read the keys from a nearer place and kept the walk: {about(prefill.first)} ms, {fmt(prefill.firstLoss)}% slower. The tiled kernel gives one warp {flashShape.tile} query rows and passes the keys by them {flashShape.tile} at a time on the tensor cores: <span className="text-accent">{about(prefill.tiled)} ms</span>, {fmt(prefill.ratio, 1)} times less and {fmt(prefill.ahead)}% under llama.cpp in the same session. Its cost per token stays between {prefill.perTokenLow} and {prefill.perTokenHigh} ms at every length measured.
        </p>
        <p>
          What a prefill is made of changed with it. In a profile of the same prompt, attention is {about(prefill.attentionMs)} of {about(prefill.spanMs)} ms, {fmt(prefill.attentionPct)}%, and the {4 * layers} GEMMs are {fmt(prefill.gemmPct)}%. Prefill now looks like decode: mostly cuBLAS. The two systems are not timed in quite the same way, since llama.cpp&apos;s number has no sampler in it.
        </p>
      </>
    ),
  },
  {
    id: "load",
    title: loadTitle,
    body: (
      <>
        <p>
          Under load, requests arrive at random at a set rate, and what counts is goodput: the requests per second that got their first token within {fmt(load.slo.ttft_ms)} ms and every later token within {fmt(load.slo.tpot_ms)} ms on average. The sweep shown here was measured on {load.date} at commit {load.commit}
          {flashClause}.
        </p>
        <p>
          The engine&apos;s best goodput is <span className="text-accent">{about(load.best.value)}</span> requests per second, at an offered {load.best.qps}
          {load.engineStillRising ? ", the highest rate offered" : ""}. vLLM&apos;s is <span className="text-warn">{about(load.vllm.value)}</span> at an offered {load.vllm.qps}
          {load.vllmStillRising ? ", the highest rate offered, so its real peak was not found" : ""}. {sentence(loadVerdict)}.
        </p>
        <p>
          The tail shows it first. At an offered {load.tailRate} the engine&apos;s TTFT p99 is {about(load.tailEngine)} ms and vLLM&apos;s {about(load.tailVllm)} ms. The engine keeps that p99 inside the SLO {upTo(load.engineHolds)}, vLLM {upTo(load.vllmHolds)}.
          {whole && chunkedGap && ` Inside an answer, the longest wait between two tokens at that rate is ${about(whole.largest)} ms with whole prompts and ${about(chunkedGap.largest)} ms with a token budget; vLLM's gaps were not recorded.`}
        </p>
        <p>
          The comparison is uneven in both directions. vLLM&apos;s times include its HTTP server and the engine&apos;s do not. Both were held to {next.loadBatch} sequences at once, which is not vLLM&apos;s default. The lengths are uniform and the tokens random, not a real chat workload.
        </p>
      </>
    ),
  },
  {
    id: "next",
    title: "three things to do next",
    body: (
      <>
        <p>
          First, serve more than {next.loadBatch} sequences at once. With every request submitted together the engine goes from {about(next.throughputAtLoad)} to <span className="text-accent">{about(next.throughputAtKnee)}</span> tokens per second between {next.loadBatch} and {next.kneeBatch} rows, and the KV cache is not what stops it. That run has not been made under load, or against vLLM at its own default.
        </p>
        <p>
          Second, read K and V once per KV head. At {next.kneeBatch} rows attention is <span className="text-accent">{fmt(next.attentionShare)}%</span> of the step and grows in proportion to the rows: each row reads its own keys and values, and the {next.group} query heads that share a KV head each read them again. The prefill kernel does the same: it asks the caches for {about(next.requestedMb)} MB per layer to use {about(next.neededMb)} MB, with its tensor cores at {about(next.tensorPct)}% of their peak.
        </p>
        <p>
          Third, prepare the next step while this one runs. {about(next.hostMs)} ms of every batch-1 token, <span className="text-accent">{fmt(next.hostPct)}%</span>, pass between two steps. Without them a token would sit at {fmt(next.roofWithout)}% of the bandwidth roof where it sits at {fmt(next.roofNow)}% (calculated).
        </p>
        <p>
          Two smaller ones were left out: a token budget of {next.chunkBudget} still costs a lone long prompt {next.chunkCostPct}%, and float Gumbel noise would cut a sampled draw from {about(next.gumbelUs)} µs to under {num(Math.ceil(next.gumbelFloatHigh))} µs but changes the tokens of a fixed seed. None of the three reads fewer bytes. That is the other way past the floor, and{" "}
          <Link href="/kernels/beyond" className="link-underline text-accent">
            beyond this engine
          </Link>{" "}
          is about it.
        </p>
      </>
    ),
  },
  {
    id: "bigger",
    title: `on an ${bigSide.gpu}, ${bigSide.model}'s floor is ${fmt(bigSide.readFloorMs, 2)} ms a token`,
    body: (
      <>
        <p>
          Carry the floor to a model and a GPU people serve: {bigSide.model} in BF16 on an {bigSide.gpu}. A decode step reads every weight but the embedding table, {about(bigSide.readBytes / 1e9)} GB, and the {bigSide.gpu}&apos;s memory is published at {h100.bandwidthTBs} TB/s, so a token cannot take less than <span className="text-accent">{about(bigSide.readFloorMs)} ms</span>. Here the floor is {about(engineSide.readFloorMs)} ms: a model that reads {about(bigSide.readBytes / engineSide.readBytes)} times the bytes, on memory {about(bigSide.bandwidth / engineSide.bandwidth)} times as fast, still waits on the same division.
        </p>
        <p>
          What the rest of a token would cost there is not known. Here everything above the floor, the kernels, the launch gaps and the time between steps, adds up to {about(aboveFloor)} ms. If the kernels on the {bigSide.gpu} also reached the floor and that part stayed the same, the floor would be <span className="text-accent">{about(bigShare)}%</span> of a token against {fmt(token.floorPct)}% here: a fixed cost weighs more as the read gets shorter. That is an assumption, not a measurement.
        </p>
        <p>
          These numbers are calculated from published specs. The {bigSide.gpu} has no measured step to compare: nothing in this book ran on one or with {bigSide.model}. What carries over is the order of the work: divide the bytes by the bandwidth first, then measure what is left.
        </p>
      </>
    ),
  },
];

const captions = [
  `results/decode_step.md as committed after each change (${stepStages.filter((s) => !s.replay).map((s) => s.commit).join(", ")}), batch 1, GPU µs per decode step; time per op is a mean over the steps and the gaps are the idle part of the median span, so a bar runs a few percent past its span; the last profile takes its gaps from the graph replay`,
  `calculated: ${fmt(token.weightGb, 2)} GB of weights from the model's shapes, divided by the measured ${token.peakGbs} GB/s; measured in one run, results/decode_step_graphs.md: median time per token ${fmt(token.ms, 2)} ms, span ${fmt(now.span)} µs, busy ${fmt(now.busy)} µs; ${fmt(batchGain, 1)} times is max batch ${batchRows} against 1 with graphs, results.md`,
  `results.md, FlashAttention prefill kernels for the chart (one session on 2026-11-10, median of 5 runs, llama.cpp with -ub 512; the first kernel measured at its own commit) and Prefill against prompt length for the text (one request per run, median of 3 runs, another session); attention and GEMM shares from profiling.md, FlashAttention prefill in chunks, a later session, so compare them with each other only; in the text: ${fmt(prefill.paged, 1)} ms one walk per row, ${fmt(prefill.first, 1)} ms the first flash kernel, ${fmt(prefill.tiled, 1)} ms tiled, ${fmt(prefill.llama, 1)} ms llama.cpp; attention ${fmt(prefill.attentionMs, 1)} of ${fmt(prefill.spanMs, 1)} ms`,
  `results.md, Under load: engine runs measured ${load.date} at commit ${load.commit} on the ${load.gpu}; Poisson arrivals at each rate, medians of the runs; SLO: TTFT at most ${fmt(load.slo.ttft_ms)} ms and TPOT at most ${fmt(load.slo.tpot_ms)} ms; past saturation a run ends before the queue is steady, so read those points as overloaded, not as stable values; how vLLM was run: baselines.md; at an offered ${load.tailRate}: TTFT p99 ${fmt(load.tailEngine)} ms for the engine and ${fmt(load.tailVllm)} ms for vLLM${whole && chunkedGap ? `, largest gap ${fmt(whole.largest)} ms with whole prompts and ${fmt(chunkedGap.largest)} ms with a budget` : ""}`,
  `the bars are not one unit: ${fmt(next.throughputGain, 2)}x is tokens per second at ${next.kneeBatch} rows over ${next.loadBatch}, offline, and ${fmt(next.attentionShare)}% attention's share of an eager ${next.kneeBatch}-row step (results.md, Batch size past 16); ${fmt(next.hostPct)}% is time per token minus span; none of the three is built, so each is an upper bound read from a measurement; ${fmt(next.throughputAtLoad)} tokens/s at ${next.loadBatch} rows and ${fmt(next.throughputAtKnee)} at ${next.kneeBatch}; the prefill kernel asks for ${fmt(next.requestedMb, 1)} MB per layer to use ${fmt(next.neededMb, 1)} MB, tensor cores at ${next.tensorPct}%; open issues #94 (attention at batch 16 and more) and #122 (overlap host work with GPU work)`,
  `calculated, not measured: ${scaleSources.big}; read per token (${fmt(l3.params)} − ${fmt(l3.vocab)} × ${fmt(l3.hidden)}) × 2 = ${fmt(bigSide.readBytes)} bytes, ÷ ${h100.bandwidthTBs} TB/s = ${fmt(bigSide.readFloorMs, 2)} ms; above the floor here ${fmt(token.ms, 2)} − ${fmt(token.floorMs, 2)} = ${fmt(aboveFloor, 2)} ms (measured minus calculated); ${fmt(bigSide.readFloorMs, 2)} ÷ (${fmt(bigSide.readFloorMs, 2)} + ${fmt(aboveFloor, 2)}) = ${fmt(bigShare, 1)}%, if that part stayed the same`,
];

export const timeSteps = steps;
export const timeCaptions = captions;

export function TimeScaleFigure() {
  return <ScaleFigure title={`the floor for ${bigSide.model} on an ${bigSide.gpu}`} quantities={["readFloor"]} marks={{ readFloor: { value: token.ms, label: `one token measured here, ${fmt(token.ms, 2)} ms` } }} />;
}

const figures = [StagesFigure, FloorFigure, PrefillFigure, LoadFigure, NextFigure, TimeScaleFigure];

export function TimeStory() {
  return (
    <StickyStory
      steps={steps}
      caption={(i) => captions[i]}
      figure={(i) => {
        const Figure = figures[i] ?? NextFigure;
        return <Figure />;
      }}
    />
  );
}
