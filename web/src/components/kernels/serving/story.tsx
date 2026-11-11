"use client";

import { chapterNumber } from "@/content/book-order";
import { blockBytes, blockTokens, budgetSystem, chatFit, engineBest, loadPredatesFlash, queueBound, vllmPreemptions, chatLengths, chatMean, engine, firstOverPool, fmt, fullBlocks, fullContext, heldBy, highRate, kvValues, loadConfigs, loadPreemptions, loadRates, lowRate, maxBatch, peak, poolBlocks, poolFraction, poolGb, poolTokens, reservedFit, slo, sloMet, sweepLast, sweepRows, tokenBytes, tpot, trace, ttftTail, usableBlocks, vllm, worstCaseBlocks, worstCasePct } from "@/content/kernels/serving";
import { measured } from "@/content/kernels/types";
import { loadRun, modelShape } from "@/lib/data";
import { StickyStory, type StoryStep } from "../sticky-story";
import { BatchingFigure, BatchLimitFigure, batchGrids, BudgetFigure, FitFigure, LoadFigure, QueueFigure, tightBlocks } from "./figures";
import { firstBlocked, firstEviction, queueBlocks, queueConfig, queueEvictions, queueRequests, queueRun, selfEviction } from "./queue";

const layers = modelShape().layers;
const run = loadRun();
const gs = batchGrids.static;
const gr = batchGrids.roomy;
const gt = batchGrids.tight;
const tightPreemptions = gt.evictions.reduce((n, e) => n + e, 0);
const tput = (batch: string) => measured("throughput-and-latency", { system: "engine + graphs", "max batch": batch }, "output tok/s", 0, "results");
const best = { engine: peak(engine), budget: peak(budgetSystem), vllm: peak(vllm) };
const allMet = loadRates.filter((q) => sloMet(engine, q) === 100);
const held100 = heldBy(reservedFit);
const victim = queueRequests[queueRun[firstEviction].evicted[0]].name;
const needy = queueRequests[queueRun[firstEviction].needy ?? 0].name;
const blocked = queueRun[firstBlocked].wait;
const self = selfEviction >= 0 ? queueRequests[queueRun[selfEviction].evicted[0]].name : "";

export const servingRanges = {
  admission: { from: 0, to: 3, still: 1 },
  preempt: { from: Math.max(0, firstEviction - 2), to: Math.min(queueRun.length - 1, Math.max(firstBlocked, selfEviction, firstEviction + 1)), still: firstEviction },
};

export const servingSteps: StoryStep[] = [
  {
    id: "budget",
    title: `every token costs ${fmt(tokenBytes)} bytes of cache`,
    body: (
      <>
        <p>
          The KV cache keeps a key and a value for every token a request has seen, in every layer (chapter {chapterNumber("paged-kv")}). For TinyLlama that is 2 × {layers} layers × {kvValues} numbers × 2 bytes,{" "}
          <span className="text-accent">{fmt(tokenBytes)} bytes</span> a token. The cache is handed out in blocks of {blockTokens} tokens, {fmt(blockBytes)} bytes each.
        </p>
        <p>
          At startup the engine takes {poolFraction}% of the memory left after the weights as one pool: <span className="text-text">{fmt(poolBlocks)}</span> blocks on this GPU, {fmt(poolGb, 2)} GB, room for {fmt(poolTokens)} tokens. That is the budget. Nothing adds to it while the server runs.
        </p>
        <p>
          A request of length L holds ceil(L / {blockTokens}) blocks: enough for its prompt when it starts, then one more each time its length reaches a multiple of {blockTokens}. So a request has no fixed price. It starts small, grows by a block every {blockTokens} decode steps, and gives everything back when it finishes.
        </p>
      </>
    ),
  },
  {
    id: "fit",
    title: "divide the pool by what requests hold",
    body: (
      <>
        <p>
          How many requests fit is the pool divided by what one request holds, and the question is what to count. Nobody knows on arrival how long a request will get. Its context can run to {fmt(fullContext)} tokens, {fullBlocks} blocks. Reserve that for each and <span className="text-warn">{reservedFit}</span> requests fit, whatever they go on to do.
        </p>
        <p>
          The engine counts what is there: a request holds only the blocks its tokens fill. The requests of a recorded chat run ended at {Math.min(...chatLengths)} to {Math.max(...chatLengths)} tokens. {reservedFit} of them hold <span className="text-accent">{fmt(held100)}</span> blocks, {fmt((100 * held100) / poolBlocks, 1)}% of the pool, and the pool has room for about {fmt(chatFit)} such requests. Drag the slider on the right.
        </p>
        <p>
          The price is that the answer is not known in advance. Requests keep growing after they were let in, so a pool that had room for everyone at admission can still run out later. Preemption, further on, is what the engine does then.
        </p>
      </>
    ),
  },
  {
    id: "max-batch",
    title: "the row limit filled first, never the pool",
    body: (
      <>
        <p>
          There is a second limit, max_batch: the most requests the scheduler runs at once, {maxBatch} by default. It is there because of time, not memory. Every running request adds a row to every step, and a step with more rows takes longer: {sweepRows.map((r) => `${r.tpot} ms at ${r.batch}`).join(", ")}, measured. Each running request waits one whole step for each of its tokens, so more rows mean more tokens per second in total and a slower answer for everyone.
        </p>
        <p>
          {maxBatch} requests that all reach the full {fmt(fullContext)} tokens hold <span className="text-accent">{fmt(worstCaseBlocks)}</span> blocks, {fmt(worstCasePct)}% of the pool{worstCaseBlocks <= usableBlocks ? `, so with max_batch ${maxBatch} the pool cannot run out` : ", more than the pool"}. The load runs later in this chapter recorded <span className="text-accent">{loadPreemptions}</span> preemptions over all {loadConfigs} engine configurations and {loadRates.length} rates.{worstCaseBlocks <= usableBlocks ? " A request that waited there waited for a row, never for memory." : ""}
        </p>
        <p>
          Raise max_batch and memory comes back into play{firstOverPool ? `: from ${firstOverPool.batch} on, the worst case is more than the pool` : ""}. The run at max_batch {sweepLast.batch} peaked at only {fmt(sweepLast.peak)} blocks because its requests were short. A larger model shifts the balance the other way: each token costs more bytes and the weights leave less room for the pool.
        </p>
      </>
    ),
  },
  {
    id: "admission",
    title: "first come, first served, nothing skips ahead",
    body: (
      <>
        <p>
          Before every step, Scheduler::schedule() looks at the front of the waiting queue. While fewer than max_batch requests are running, it asks the block manager one question: are ceil(L / {blockTokens}) blocks free, with L the request&apos;s whole length so far? If so the request takes those blocks and joins the back of the running list. If not, or if its prompt would push the prompts admitted in this step past max_prefill_tokens, admission stops for this step.
        </p>
        <p>
          Two things follow. The queue is strict: a short request behind a long one waits even when it would fit, so a long request is never starved by a stream of short ones. And only what exists is reserved: blocks for the prompt, none for the answer. The answer&apos;s blocks are asked for one at a time, as it grows.
        </p>
        <p>
          A step that admits anyone is the prefill of the requests it admitted, and only of them. Each of them gets its first token from it, and the requests that were already running sit that step out. Chapter {chapterNumber("chunked-prefill")} is about that wait.
        </p>
      </>
    ),
  },
  {
    id: "preempt",
    title: "a full pool evicts the newest request",
    body: (
      <>
        <p>
          In a decode step every running request appends one slot. Most of the time it lands in a block the request already holds. When its length is a multiple of {blockTokens} it needs a new block, and if none is free the scheduler makes room: the request at the <span className="text-warn">back of the running list</span>, the one admitted last, is the victim.
        </p>
        <p>
          The victim loses every block, so its whole cache is gone. It keeps its prompt and the tokens it has generated, and goes to the front of the waiting queue, ahead of requests that never ran. One eviction frees at least one block, which is all a growing request needs; another request later in the same step can cause another. A request that is itself at the back of the list evicts itself.
        </p>
        <p>
          Once blocks for its whole length are free it is admitted like any other request, and its prefill recomputes the keys and values of the prompt plus the generated tokens, starting at position <span className="text-accent">0</span>. Nothing the client received is taken back. The same tokens through the same weights rebuild the same cache, so a greedy request continues as if nothing had happened: a test squeezes 16 requests into 56 blocks and compares every output with a roomy run. With sampling it asks for 14 of the 16, since the recomputed values come from a pass of another shape and can differ in their last bits.
        </p>
        <p>
          Evicting from the back means the oldest request is never the victim while a newer one runs, and the scheduler refuses a pool smaller than one full context. So the oldest request can always finish, and the queue always drains.
        </p>
      </>
    ),
  },
  {
    id: "static",
    title: "a static batch waits for its longest request",
    body: (
      <>
        <p>
          Why decide again before every step? The alternative is a static batch: take the first {gs.slots} requests, run them together, and take the next group only when every one of them is done. A request that stops early keeps its slot until the longest request in its batch finishes.
        </p>
        <p>
          Here one request runs far longer than the rest of its batch, which is done early. The dashed cells are slots that go through every forward pass and produce nothing: the requests waste <span className="text-warn">{gs.wasted}</span> slot-steps.
        </p>
      </>
    ),
  },
  {
    id: "continuous",
    title: "continuous batching refills a slot every step",
    body: (
      <>
        <p>
          Continuous batching, also called iteration-level scheduling, runs the scheduler before every step instead of before every batch. A finished request leaves at once, and the request at the front of the queue takes its slot on the next step.
        </p>
        <p>
          Same queue, room for every request: the requests waste only <span className="text-accent">{gr.wasted}</span> slot-steps, all of them at the end, when the queue is empty and the last requests drain.
        </p>
      </>
    ),
  },
  {
    id: "rules",
    title: "a tight pool costs recomputed steps",
    body: (
      <>
        <p>
          Now give the same queue a pool that is too small: a few blocks, each a couple of steps long where the engine&apos;s hold {blockTokens} tokens. Admission and preemption both show in the grid. <span className="text-warn">{tightPreemptions}</span> preemptions follow: the crossed cells are evictions and the shaded cells are steps recomputed after one.
        </p>
        <p>
          Every shaded cell is work done twice, which is what preemption costs. The run still wastes {gt.wasted} slot-steps against the {gs.wasted} of static batching. The first workbench at the end of this chapter runs the same two rules on a real block pool.
        </p>
      </>
    ),
  },
  {
    id: "load",
    group: "measured",
    title: `goodput peaks at an offered ${engineBest.qps} requests a second`,
    body: (
      <>
        <p>
          The test: 120 requests arrive at random times, a Poisson process at {lowRate} to {highRate} per second, with prompts of 16 to 512 tokens and answers of 16 to 256, and at most {maxBatch} run together. Goodput counts the requests per second that got their first token within {slo.ttft_ms} ms and each later token within {slo.tpot_ms} ms.
        </p>
        <p>
          {allMet.length > 0 && `Up to ${allMet[allMet.length - 1]} per second every request meets both. `}With whole-prompt steps the engine&apos;s best is <span className="text-accent">{best.engine.value.toFixed(2)}</span> at an offered {best.engine.qps}, and at {highRate} only {sloMet(engine, highRate)}% of the requests are in time. Its 99th percentile wait for the first token is {fmt(ttftTail(engine, highRate))} ms at {highRate} per second, while the median time per token is {tpot(engine, highRate)} ms.{queueBound(engine, highRate) ? ` The first token misses its limit and the later ones do not: a running request is fine, and the late ones are in the queue, waiting for one of the ${maxBatch} rows.` : ""}
        </p>
        <p>
          What running together buys is the other number: offline, a batch of {maxBatch} gives {fmt(tput(String(maxBatch)))} tokens per second against {fmt(tput("1"))} one request at a time. {loadPredatesFlash ? `These load runs are from ${run.date} and predate the FlashAttention prefill kernel of chapter ${chapterNumber("flash-prefill")}; they have not been rerun with it.` : `These load runs are from ${run.date}, with the FlashAttention prefill kernel of chapter ${chapterNumber("flash-prefill")}.`}
        </p>
      </>
    ),
  },
  {
    id: "vllm",
    group: "measured",
    title: best.vllm.value > engineBest.value ? "vLLM won on kernel speed, not scheduling" : "vLLM and the engine, on the same arrivals",
    body: (
      <>
        <p>
          vLLM served the same token ids at the same arrival times, with the same weights and the same limit of {maxBatch} requests. Its goodput reaches <span className="text-warn">{best.vllm.value.toFixed(2)}</span> at an offered {best.vllm.qps}, {(best.vllm.value / engineBest.value).toFixed(1)}x the engine&apos;s best of {engineBest.value.toFixed(2)}.
        </p>
        <p>
          The scheduler is not the difference. Both batch continuously and both stop at {maxBatch} rows{loadPreemptions + vllmPreemptions === 0 ? ", and neither ran short of memory: no run of either preempted a request" : ""}. vLLM mixes pieces of prefill into its decode steps; the engine does the same with a token budget (chapter {chapterNumber("chunked-prefill")}) and its best goodput moves from {best.engine.value.toFixed(2)} to {best.budget.value.toFixed(2)}.
        </p>
        <p>
          The difference is how long the work takes. At {lowRate} per second the engine&apos;s median time per token is {tpot(engine, lowRate)} ms and vLLM&apos;s {tpot(vllm, lowRate)}; at {highRate} it is {tpot(engine, highRate)} against {tpot(vllm, highRate)}.
          {trace ? ` In the engine's run at ${trace.qps} per second, ${fmt(trace.prefill_steps)} prefill steps took ${trace.prefill_s} of the ${trace.run_s} seconds, and its ${fmt(trace.decode_steps)} decode steps averaged ${trace.decode_ms} ms each, against vLLM's whole time per token of ${tpot(vllm, String(trace.qps))} ms with its prefill included.` : ""} {loadPredatesFlash ? " In these runs the engine's prompt rows went through its decode attention kernel one row at a time, a cost that grows with the square of the prompt, while vLLM runs FlashAttention." : ` These runs already use the FlashAttention prefill kernel of chapter ${chapterNumber("flash-prefill")}, so prefill is the smaller part of the run: the rest of the gap is the decode step at a batch of ${maxBatch}.`}
        </p>
        <p>
          The comparison even favours the engine: vLLM&apos;s times include its HTTP server and a Python client, and the engine&apos;s are taken inside the process. Its limits: uniform lengths and random tokens, 120 requests a run, and a cap of {maxBatch} that is not vLLM&apos;s default.
        </p>
      </>
    ),
  },
];

export const servingCaptions = [
  `TinyLlama: ${layers} layers, ${modelShape().kv_heads} KV heads of ${modelShape().hidden / modelShape().heads} numbers in BF16; bytes per block from kv_block_bytes in src/kv/cache_size.cpp; the pool is ${poolFraction}% of the free device memory at startup (kKvCacheFraction), ${fmt(poolBlocks)} blocks in the recorded run; ${fmt(usableBlocks)} can be handed out, block 0 is the null block`,
  `calculated: ${fmt(usableBlocks)} blocks / ${fullBlocks} = ${reservedFit} full contexts; the lengths are the ${chatLengths.length} final lengths of the recorded chat run (mean ${fmt(chatMean)} tokens), repeated; this counts memory only, not max_batch`,
  `step times, tokens per second and peak blocks from docs/results.md (Batch size past 16): twice max_batch requests at once, 64 prompt and 64 output tokens, CUDA Graphs; the worst case is calculated, max_batch × ${fullBlocks} blocks; preemptions from the load table`,
  `illustrative: ${queueRequests.length} requests, a pool of ${queueBlocks} blocks of ${queueConfig.blockSize} tokens and max_batch ${queueConfig.maxBatch}, run through a TypeScript port of Scheduler::schedule(); the engine's blocks hold ${blockTokens} tokens and max_prefill_tokens defaults to ${fmt(fullContext)}`,
  `the same example run: ${needy} needs a block at step ${queueRun[firstEviction].step} and ${victim} is evicted${blocked ? `; at step ${queueRun[firstBlocked].step} ${queueRequests[blocked.id].name} needs ${blocked.need} blocks and waits at the front` : ""}${self ? `; at step ${queueRun[selfEviction].step} ${self} evicts itself` : ""}; ${queueEvictions} evictions in all; the test is ForcedPreemptionKeepsTheOutputsOfEveryRequest in tests/gpu/test_serve.cpp`,
  `illustrative request lengths in steps, ${gs.slots} slots; the same 10 requests in every figure; request 1 runs 9 steps, requests 0, 2 and 3 end by step 3; ${gs.steps} steps in all`,
  `illustrative lengths; a pool that never runs out; a prefill and a decode step are drawn as one cell; ${gr.steps} steps against static batching's ${gs.steps}`,
  `illustrative lengths; ${tightBlocks} blocks of 2 steps instead of ${blockTokens} tokens; ${gt.steps} steps and ${gt.wasted} wasted slot-steps against static batching's ${gs.wasted}; rules from src/sched/scheduler.cpp: first come first served up to max_batch, preempt from the back of the running list`,
  `docs/results.md (Under load), measured ${run.date} at commit ${run.commit} on the ${run.gpu.replace("NVIDIA GeForce ", "")}: engine bench --max-batch ${maxBatch} --graphs, median of 3 runs, greedy; with 120 requests an overloaded run ends before its queue settles, so read the points past the peak as overloaded, not as stable values`,
  `the same sweep; vLLM 0.32.0 run as in docs/baselines.md with --max-num-seqs ${maxBatch}, its times taken in the client over HTTP${trace ? `; the step counts are from the trace of one engine run at ${trace.qps} per second` : ""}; vLLM's attention backend there is FlashAttention 2`,
];

export function ServingStory() {
  return (
    <StickyStory
      steps={servingSteps}
      caption={(i) => servingCaptions[i]}
      figure={(i) => {
        switch (servingSteps[i].id) {
          case "budget":
            return <BudgetFigure />;
          case "fit":
            return <FitFigure />;
          case "max-batch":
            return <BatchLimitFigure />;
          case "admission":
            return <QueueFigure key="admission" {...servingRanges.admission} />;
          case "preempt":
            return <QueueFigure key="preempt" {...servingRanges.preempt} />;
          case "static":
            return <BatchingFigure kind="static" />;
          case "continuous":
            return <BatchingFigure kind="roomy" />;
          case "rules":
            return <BatchingFigure kind="tight" />;
          case "load":
            return <LoadFigure kind="goodput" />;
          default:
            return <LoadFigure kind="tpot" />;
        }
      }}
    />
  );
}
