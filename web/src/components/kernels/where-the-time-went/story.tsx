"use client";

import Link from "next/link";
import { throughput, valueAt } from "@/content/results";
import { flashShape } from "@/content/kernels/flash-prefill";
import { eagerStage, firstStage, load, loadTitle, loadVerdict, next, noisy, nowStage, prefill, smallOf, stageById, stepStages, timeFmt, token } from "@/content/kernels/where-the-time-went";
import { modelShape } from "@/lib/data";
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
const flashClause = load.hasFlash === null ? "" : load.hasFlash ? ", with the tiled prefill kernel of the FlashAttention prefill chapter" : ", before the tiled prefill kernel became the default, so its prompts still ran through the decode attention";

const steps: StoryStep[] = [
  {
    id: "decode",
    title: "every kernel shrank its slice, and the GEMMs kept the step",
    body: (
      <>
        <p>
          The first profile of a batch-1 decode step, on {first.date}, had {first.kernels} kernels in a span of {fmt(first.span)} µs. The projections and the LM head, all cuBLAS, took <span className="text-accent">{fmt(first.groups.gemm)} µs</span> of kernel time, attention {fmt(first.groups.attention)} µs, and the norms, adds, RoPE, SwiGLU and the cache write {fmt(smallOf(first))} µs together.
        </p>
        <p>
          Then the chapters landed. {fast && `Fusing each add with the norm after it took ${first.kernels - fast.kernels} launches out of the step. `}
          {fast && v2 && `Attention v2 took attention from ${fmt(fast.groups.attention)} to ${fmt(v2.groups.attention)} µs, the largest single cut. `}
          {v2 && tuned && `Tuning add + RMSNorm took the small kernels from ${fmt(smallOf(v2))} to ${fmt(smallOf(tuned))} µs. `}
          CUDA Graphs removed the gaps: {fmt(eagerStage.idle)} µs of idle time in a step became <span className="text-accent">{fmt(now.idle)} µs</span>.
        </p>
        <p>
          The GEMMs did not move, because no chapter changed them: {fmt(first.groups.gemm)} µs then and {fmt(now.groups.gemm)} µs now, a difference the clocks explain. Their share of the kernel time rose from {fmt(token.firstGemmShare)}% to {fmt(token.gemmShare)}% only because everything around them shrank. Sampling is outside the span: one greedy sample kernel after each step, {now.sampleUs} µs.
        </p>
        {noisy.length > 0 && (
          <p>
            One capture reads oddly: on {noisy[0].date} the gaps were {noisy[0].idlePct}% of the span, where today&apos;s are {eagerStage.idlePct}%. Gaps depend on what else the machine is doing. Kernel time is the steadier number.
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
          Foundations gave the floor. A decoded token reads every weight once, {fmt(token.weightGb, 2)} GB, and this GPU&apos;s memory delivers {token.peakGbs} GB/s at best, so a token cannot take less than <span className="text-accent">{fmt(token.floorMs, 2)} ms</span>. No kernel changes that number.
        </p>
        <p>
          Today&apos;s token takes {fmt(token.ms, 2)} ms with the step replayed from a graph. <span className="text-accent">{fmt(token.floorPct)}%</span> of it is that read at the peak rate. The kernels spend {fmt(token.aboveMs, 2)} ms above the floor, the launch gaps {fmt(token.gapsMs, 2)} ms, and {fmt(token.hostMs, 2)} ms, {fmt(token.hostPct)}%, pass between the end of one step and the start of the next.
        </p>
        <p>
          So for one request at a time, a faster kernel has {fmt(token.aboveMs, 2)} ms left to win. The larger gains in this book came from sharing the read: with {batchRows} rows in a step the same {fmt(token.weightGb, 2)} GB serve {batchRows} tokens, and the engine makes <span className="text-accent">{fmt(batchGain, 1)} times</span> the tokens per second it makes alone.
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
          Decode started close to its floor{token.firstMs !== null && `: the first profiled token already spent ${fmt(firstFloorPct)}% of its time on the read of the weights`}. Prefill was far from it. One {prefill.long}-token prompt took <span className="text-warn">{fmt(prefill.paged, 1)} ms</span> to its first token, against {fmt(prefill.llama, 1)} ms for llama.cpp, because every prompt token ran through the decode attention kernel and walked its whole context alone. The cost per prompt token climbed from {prefill.pagedPerTokenShort} to {prefill.pagedPerTokenLong} ms as the prompt grew.
        </p>
        <p>
          The first FlashAttention kernel read the keys from a nearer place and kept the walk: {fmt(prefill.first, 1)} ms, {fmt(prefill.firstLoss)}% slower. The tiled kernel gives one warp {flashShape.tile} query rows and passes the keys by them {flashShape.tile} at a time on the tensor cores: <span className="text-accent">{fmt(prefill.tiled, 1)} ms</span>, {fmt(prefill.ratio, 1)} times less and {fmt(prefill.ahead)}% under llama.cpp in the same session. Its cost per token stays between {prefill.perTokenLow} and {prefill.perTokenHigh} ms at every length measured.
        </p>
        <p>
          What a prefill is made of changed with it. In a profile of the same prompt, attention is {fmt(prefill.attentionMs, 1)} of {fmt(prefill.spanMs, 1)} ms, {fmt(prefill.attentionPct)}%, and the {4 * layers} GEMMs are {fmt(prefill.gemmPct)}%. Prefill now looks like decode: mostly cuBLAS. The two systems are not timed in quite the same way, since llama.cpp&apos;s number has no sampler in it.
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
          The engine&apos;s best goodput is <span className="text-accent">{fmt(load.best.value, 2)}</span> requests per second, at an offered {load.best.qps}
          {load.engineStillRising ? ", the highest rate offered" : ""}. vLLM&apos;s is <span className="text-warn">{fmt(load.vllm.value, 2)}</span> at an offered {load.vllm.qps}
          {load.vllmStillRising ? ", the highest rate offered, so its real peak was not found" : ""}. {sentence(loadVerdict)}.
        </p>
        <p>
          The tail shows it first. At an offered {load.tailRate} the engine&apos;s TTFT p99 is {fmt(load.tailEngine)} ms and vLLM&apos;s {fmt(load.tailVllm)} ms. The engine keeps that p99 inside the SLO {upTo(load.engineHolds)}, vLLM {upTo(load.vllmHolds)}.
          {whole && chunkedGap && ` Inside an answer, the longest wait between two tokens at that rate is ${fmt(whole.largest)} ms with whole prompts and ${fmt(chunkedGap.largest)} ms with a token budget; vLLM's gaps were not recorded.`}
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
          First, serve more than {next.loadBatch} sequences at once. With every request submitted together the engine goes from {fmt(next.throughputAtLoad)} to <span className="text-accent">{fmt(next.throughputAtKnee)}</span> tokens per second between {next.loadBatch} and {next.kneeBatch} rows, and the KV cache is not what stops it. That run has not been made under load, or against vLLM at its own default.
        </p>
        <p>
          Second, read K and V once per KV head. At {next.kneeBatch} rows attention is <span className="text-accent">{fmt(next.attentionShare)}%</span> of the step and grows in proportion to the rows: each row reads its own keys and values, and the {next.group} query heads that share a KV head each read them again. The prefill kernel does the same: it asks the caches for {fmt(next.requestedMb)} MB per layer to use {fmt(next.neededMb, 1)} MB, with its tensor cores at {next.tensorPct}% of their peak.
        </p>
        <p>
          Third, prepare the next step while this one runs. {fmt(next.hostMs, 2)} ms of every batch-1 token, <span className="text-accent">{fmt(next.hostPct)}%</span>, pass between two steps. Without them a token would sit at {fmt(next.roofWithout)}% of the bandwidth roof where it sits at {fmt(next.roofNow)}% (calculated).
        </p>
        <p>
          Two smaller ones were left out: a token budget of {next.chunkBudget} still costs a lone long prompt {next.chunkCostPct}%, and float Gumbel noise would cut a sampled draw from {fmt(next.gumbelUs)} µs to under {fmt(next.gumbelFloatHigh)} µs but changes the tokens of a fixed seed. None of the three reads fewer bytes. That is the other way past the floor, and{" "}
          <Link href="/kernels/beyond" className="link-underline text-accent">
            beyond this engine
          </Link>{" "}
          is about it.
        </p>
      </>
    ),
  },
];

const captions = [
  `results/decode_step.md as committed after each change (${stepStages.filter((s) => !s.replay).map((s) => s.commit).join(", ")}), batch 1, GPU µs per decode step; time per op is a mean over the steps and the gaps are the idle part of the median span, so a bar runs a few percent past its span; the last profile takes its gaps from the graph replay`,
  `calculated: ${fmt(token.weightGb, 2)} GB of weights from the model's shapes, divided by the measured ${token.peakGbs} GB/s; measured in one run, results/decode_step_graphs.md: median time per token ${fmt(token.ms, 2)} ms, span ${fmt(now.span)} µs, busy ${fmt(now.busy)} µs; ${fmt(batchGain, 1)} times is max batch ${batchRows} against 1 with graphs, docs/results.md`,
  `docs/results.md, Prefill against prompt length and FlashAttention prefill kernels: one request per run, median of 5 runs, same session, llama.cpp with -ub 512; the first kernel was measured at its own commit; attention and GEMM shares from docs/profiling.md, FlashAttention prefill in chunks, a later session, so compare them with each other only`,
  `docs/results.md, Under load: engine runs measured ${load.date} at commit ${load.commit} on the ${load.gpu}; Poisson arrivals at each rate, medians of the runs; SLO: TTFT at most ${fmt(load.slo.ttft_ms)} ms and TPOT at most ${fmt(load.slo.tpot_ms)} ms; past saturation a run ends before the queue is steady, so read those points as overloaded, not as stable values; how vLLM was run: docs/baselines.md`,
  `the bars are not one unit: ${fmt(next.throughputGain, 2)}x is tokens per second at ${next.kneeBatch} rows over ${next.loadBatch}, offline, and ${fmt(next.attentionShare)}% attention's share of an eager ${next.kneeBatch}-row step (docs/results.md, Batch size past 16); ${fmt(next.hostPct)}% is time per token minus span; none of the three is built, so each is an upper bound read from a measurement; open issues #94 (attention at batch 16 and more) and #122 (overlap host work with GPU work)`,
];

export const timeSteps = steps;
export const timeCaptions = captions;

const figures = [StagesFigure, FloorFigure, PrefillFigure, LoadFigure, NextFigure];

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
