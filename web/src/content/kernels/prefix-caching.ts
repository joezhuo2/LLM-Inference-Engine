import { cachedOf, shareRun } from "@/components/kernels/prefix-caching/share";
import { kvReplay, modelShape, type PrefixRow, prefixRuns, profilingTable, resultsDoc } from "@/lib/data";
import type { KernelStory } from "./types";

export const pcFmt = (v: number, d = 0) => v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });

const fail = (what: string) => {
  throw new Error(`results.md Prefix caching: ${what}`);
};
const section = "prefix-caching";
const runs = prefixRuns();
const table = profilingTable(section, 0, "results");
const column = (name: string) => {
  const i = table.header.indexOf(name);
  if (i < 0) fail(`no column "${name}"`);
  return i;
};
const tableRow = (r: PrefixRow) => table.rows.find((t) => t[column("shared prefix")] === String(r.shared) && t[column("own tokens")] === String(r.own) && t[column("arrivals")] === r.arrivals && t[column("cache")] === r.cache) ?? fail(`no row for ${r.shared} shared, ${r.arrivals}, cache ${r.cache}`);
const same = (cell: string, v: number, d: number) => cell.replace("%", "") === v.toFixed(d);

export type PrefixSetting = PrefixRow & { prefill_ms: number };

export const settings: PrefixSetting[] = runs.rows.map((r) => {
  const t = tableRow(r);
  if (!same(t[column("first request TTFT ms")], r.first_ttft_ms, 1) || !same(t[column("later requests TTFT p50 ms")], r.later_ttft_ms, 1) || !same(t[column("hit rate")], 100 * r.hit_rate, 1) || !same(t[column("output tok/s")], r.output_tps, 1)) fail(`the row for ${r.shared} shared, ${r.arrivals}, cache ${r.cache} differs from results/prefix/; rerun scripts/prefix_cache_report.py`);
  const prefill = Number.parseFloat(t[column("prefill step p50 ms")]);
  if (Number.isNaN(prefill)) fail(`no prefill step for ${r.shared} shared, ${r.arrivals}, cache ${r.cache}`);
  return { ...r, prefill_ms: prefill };
});

const at = (shared: number, rate: number, cache: "on" | "off") => settings.find((s) => s.shared === shared && s.rate === rate && s.cache === cache) ?? fail(`no setting for ${shared} shared at rate ${rate}, cache ${cache}`);
const quietRate = Math.max(...settings.map((s) => s.rate));
export const sharedLengths = [...new Set(settings.map((s) => s.shared))].sort((a, b) => a - b);
export const pairs = sharedLengths.map((p) => ({ shared: p, quiet: { on: at(p, quietRate, "on"), off: at(p, quietRate, "off") }, burst: { on: at(p, 0, "on"), off: at(p, 0, "off") } }));

const first = settings[0];
const block = kvReplay().meta.block_size;
const sameOut = resultsDoc.prefix_same ?? fail('the "Same output" paragraph no longer gives the identical outputs, the near ties and the layers');
const budget = resultsDoc.prefix_budget ?? fail("the paragraph on all-at-once arrivals no longer gives the prefill token budget");
if (sameOut.layers !== modelShape().layers) fail("the layers of the bit-identical check are not TinyLlama's");

for (const s of settings.filter((r) => r.cache === "on")) {
  const port = shareRun({ shared: s.shared, own: s.own, requests: s.requests, atOnce: s.rate === 0, budget, maxBatch: s.max_batch, block });
  if (port.cachedTotal !== s.cached_tokens || port.total !== s.prefill_tokens) fail(`the block arithmetic of the sandbox gives ${port.cachedTotal} cached tokens for ${s.shared} shared, ${s.arrivals}, and the runs ${s.cached_tokens}; fix share.ts or the chapter`);
}
const largest = pairs[pairs.length - 1];
if (pairs.some((p) => p.quiet.on.later_ttft_ms >= p.quiet.off.later_ttft_ms || p.burst.on.output_tps <= p.burst.off.output_tps)) fail("the chapter says a hit lowers the later requests' TTFT and raises the tokens per second all at once at every prefix length; rewrite it");
if (largest.quiet.on.first_ttft_ms >= 0.9 * largest.quiet.off.first_ttft_ms) fail("the chapter records the cold first request as faster with the cache on, unexplained; rewrite that sentence");

const quietPrefill = pairs.map((p) => p.quiet.on.prefill_ms);

export const prefix = {
  date: runs.date.slice(0, 10),
  gpu: runs.gpu.replace(/^NVIDIA (GeForce )?/, ""),
  commit: runs.commit,
  block,
  pool: kvReplay().meta.kv_blocks,
  budget,
  own: first.own,
  requests: first.requests,
  maxBatch: first.max_batch,
  quietArrivals: at(sharedLengths[0], quietRate, "on").arrivals,
  largest,
  smallest: pairs[0],
  prefillLow: Math.min(...quietPrefill),
  prefillHigh: Math.max(...quietPrefill),
  sameOut,
  perHit: (shared: number) => cachedOf(shared, first.own, block),
  speedup: (p: (typeof pairs)[number]) => p.burst.on.output_tps / p.burst.off.output_tps,
  firstStep: (shared: number) => shareRun({ shared, own: first.own, requests: first.requests, atOnce: true, budget, maxBatch: first.max_batch, block }).firstStep,
};

const largestOff = Math.round(largest.quiet.off.later_ttft_ms);
const largestOn = Math.round(largest.quiet.on.later_ttft_ms);

export const prefixCaching: KernelStory = {
  slug: "prefix-caching",
  index: 19,
  book: true,
  about: true,
  title: "prefix caching",
  short: `Many requests start with the same tokens. Keep the keys and values of every full block a step writes, and a later request that starts the same way skips their prefill: ${largestOff} ms to the first token became ${largestOn} at a ${pcFmt(largest.shared)}-token shared prefix (measured).`,
  formula: "tokens to prefill = prompt tokens − the whole blocks already cached",
  headline: {
    value: `${largestOff} → ${largestOn} ms`,
    label: `time to the first token of a request that shares a ${pcFmt(largest.shared)}-token prefix, cache off and on (measured, requests arriving ${prefix.quietArrivals.replace("/s", " a second")})`,
  },
  sandbox: {
    title: "share a prefix yourself",
    intro: `Set how long the shared start is, how many tokens each request adds and whether the requests arrive one at a time or all at once. The block arithmetic is a TypeScript port of the rules in src/kv/block_manager.cpp and src/sched/scheduler.cpp: only whole ${block}-token blocks are shared, the block of the last prompt token is always computed, and requests admitted in the same step find nothing. At the measured settings it gives exactly the cached tokens the runs counted.`,
  },
  sources: [
    { path: "src/kv/block_manager.cpp", label: "the block keys, the match, the shared blocks and their counts, eviction" },
    { path: "src/sched/scheduler.cpp", label: "admission with the cache, the prefill start, what a finished step publishes" },
    { path: "src/sched/paged_batch.cpp", label: "a prefill whose rows start after the cached tokens" },
    { path: "include/engine/sched/scheduler.h", label: "SchedulerConfig::prefix_cache, Step::cached, PrefixCounts" },
  ],
  tests: [
    { path: "tests/unit/test_prefix_cache.cpp", label: "the block manager's match, sharing, publishing, eviction and truncate" },
    { path: "tests/unit/test_scheduler_prefix.cpp", label: "the scheduler with a fake model that checks every key a row reads" },
    { path: "tests/gpu/test_serve_prefix_cache.cpp", label: "bit-identical keys and values, and the greedy outputs, on the GPU" },
  ],
  testing: `The 29 block manager tests run on the CPU: nothing published matches nothing, only full blocks match and the last token never does, a match follows the whole chain, shared blocks need no free blocks and are never written into, released blocks stay cached and go last, eviction takes the end of a chain first, a failed call changes nothing, and truncate takes a cut block out of the cache. The 9 scheduler tests use a fake model that checks every key a row reads was written for that row's own tokens, including random traffic. On the GPU a hit reads keys and values bit-identical to the ones its own prefill writes in all ${sameOut.layers} layers, and greedy outputs with the cache on and off are compared over shared system prompts, concurrent requests, eviction, speculation, chunked prefill and a second chat turn: ${sameOut.same} of ${sameOut.outputs} are identical and the other ${sameOut.split} split at a near tie within ${sameOut.ulps} BF16 ulps.`,
  excerpts: [
    {
      file: "src/kv/block_manager.cpp",
      from: "int BlockManager::match_blocks(std::span<const int32_t> tokens,",
      to: "    return matched;",
      caption: "The match: a block's key is every token from the start of the sequence to the end of that block, hashed and kept to compare, so a block matches only when everything before it matches too",
    },
    {
      file: "src/kv/block_manager.cpp",
      from: "    if (available < needed) {",
      to: "    sequences_[seq] = std::move(seq_data);",
      caption: "allocate_tokens: a matched block that sat released leaves the eviction list, every matched block gets one more holder and starts the new block table, and only the rest come from the pool",
    },
    {
      file: "src/sched/scheduler.cpp",
      from: "        const int written = int(req.length()) - 1;",
      to: "            block_manager_.publish(s_id, std::span(tokens).first(size_t(written)));",
      caption: "What a finished step publishes: every token whose keys and values are written, after a prefill or when a block fills",
    },
  ],
  measurements: [{ doc: "results", section, table: 0, caption: `${first.requests} requests, each a shared prefix plus ${first.own} tokens of its own, with the cache on and off: one at a time (${prefix.quietArrivals}) and all at once; median of ${first.runs} runs`, command: "scripts/prefix_cache_sweep.sh, then scripts/prefix_cache_report.py" }],
  timeline: [
    {
      date: "Nov 10",
      prs: [203],
      branch: "kv/prefix-cache-stub",
      version: "spec",
      author: "claude",
      title: "a spec and its tests, with no body",
      text: "The stub on BlockManager: match_prefix, can_allocate_tokens, allocate_tokens, publish, ref_count and num_cached throw, and a spec and 27 tests say what they must do: key a block by every token up to its end, share matched blocks with a count, never match the block of the last token, keep released blocks cached and evict the least recently released.",
    },
    {
      date: "Nov 21",
      prs: [226],
      branch: "kv/prefix-cache-impl",
      version: "impl",
      author: "joe",
      title: "Joe's prefix cache in the block manager",
      text: "Each block gets a count of the sequences that hold it, whether it is published, and its key. A map from keys to blocks finds a prompt's cached blocks, and a list of released blocks, oldest first, picks what to evict when no blank block is left. Claude's review found two bugs, a room check against the wrong count and a publish that refused short sequences; Joe fixed both.",
    },
    {
      date: "Nov 21",
      prs: [226],
      branch: "kv/prefix-cache-impl",
      version: "wired",
      author: "claude",
      title: "the scheduler shares blocks, and the engine uses it by default",
      text: "Written by Claude at Joe's request: admission matches the prompt, a prefill starts after its cached tokens, only uncached rows count against the step's token budget, and a finished step publishes what it wrote, generated tokens included. truncate now takes a block it cuts into out of the cache. serve and bench turn the cache on; --no-prefix-cache turns it off.",
      metric: { kind: "value", value: `${largestOff} → ${largestOn} ms`, label: `time to the first token at a ${pcFmt(largest.shared)}-token shared prefix (measured)` },
    },
  ],
};
