"use client";

import NumberFlow from "@number-flow/react";
import { useState } from "react";
import { flashFmt, flashKeyRows, flashLengths, flashLong, flashNumbers, flashPath, flashPaths, flashPerToken, flashProfile, flashShape, flashShort, flashSystems, flashTokens, flashTtft } from "@/content/kernels/flash-prefill";
import { modelShape } from "@/lib/data";
import { EasedBar } from "../eased-bar";
import { expectAnswer, type Prediction, Predict } from "../predict";
import { StickyStory, type StoryStep } from "../sticky-story";
import { Toggle } from "../toggle";
import { FirstKernelFigure, FragmentFigure, MmaFigure, OpenFigure, RegistersFigure, type SpecGroup, SpecFigure, SquareFigure, TileSweepFigure, TtftFigure, type TtftSeries } from "./figures";

const n = flashNumbers;
const pf = flashProfile;
const s = flashShape;
const layers = modelShape().layers;
const longTokens = flashTokens(flashLong);
const keyRows = flashKeyRows;
const calls = s.headDim / s.tile;
const qWords = calls * 4;
const accs = (s.headDim / 8) * 4;
const budget = /budget (\d+)/.exec(flashSystems.chunked)?.[1] ?? "";
const chunks = longTokens / Number(budget);
const pagedPathLong = flashPath(flashPaths.paged, flashLong);

const specGroups: SpecGroup[] = [
  { title: "matches the naive attention", lines: ["chunk lengths around the tile size", "prompts up to 8,192 tokens", "cached prefixes, 16 chunks in one call", `${s.layouts} head and block layouts`, `Hugging Face on all ${layers} layers`] },
  { title: "bit for bit the same", lines: ["alone or next to other chunks", "whatever rows come after a row", "on every repeat, and from a graph", "the first position is exactly V₀"] },
  { title: "must not read", lines: ["NaN in the cache slots of the", "chunk's own positions", "NaN outside the context", "−1 in unused block table entries"] },
  { title: "numerics", lines: ["scores up to 128: no overflow", "a maximum that rises, falls, spikes", "a fully masked first tile:", "no 0 / 0, no −∞ − −∞"] },
];

const series: TtftSeries[] = [
  { label: "first kernel", values: flashLengths.map((p) => flashPath(flashPaths.first, p)), tone: "warn" },
  { label: "decode kernels, per row", values: flashLengths.map((p) => flashTtft(flashSystems.paged, p)), tone: "baseline" },
  { label: "llama.cpp", values: flashLengths.map((p) => flashTtft(flashSystems.llama, p)), tone: "muted" },
  { label: "tiled kernel", values: flashLengths.map((p) => flashTtft(flashSystems.flash, p)), tone: "accent" },
];

const openItems = [
  `K and V are not shared between the ${s.group} query heads of a KV head\neach one reads the same keys and values again`,
  `a chunk just over ${pf.launchAfter.blocksAtOnce} blocks still waits\nbudget ${pf.wide.budget}: ${pf.wide.blocks} blocks, ${pf.wide.attention} ms of attention`,
  "the load sweeps were not rerun\nmeasured before this kernel, with per-row prefill",
];

const shortTiled = flashTtft(flashSystems.flash, flashShort);
const shortPaged = flashTtft(flashSystems.paged, flashShort);
const shortGain = shortPaged / shortTiled;
const prediction: Prediction = expectAnswer({
  question: `A ${flashShort}-token prompt is only ${flashTokens(flashShort) / s.tile} tiles long. Is the tiled kernel still faster there than one walk per row?`,
  choices: ["slower", "within 10%", "faster"],
  answer: shortGain > 1.1 ? "faster" : shortGain < 1 / 1.1 ? "slower" : "within 10%",
  why: `${shortTiled} ms against ${shortPaged} ms, ${shortGain.toFixed(1)} times less. The square is small at this length, so the walks were a smaller part of the time, but removing them still shows. Pick ${flashShort} to check.`,
}, "faster");

const pickerRows = (prompt: string) => [
  { label: "first kernel", value: flashPath(flashPaths.first, prompt), tone: "text-warn", bar: "bg-warn" },
  { label: "decode kernels", value: flashTtft(flashSystems.paged, prompt), tone: "text-muted", bar: "bg-baseline/70" },
  { label: "llama.cpp", value: flashTtft(flashSystems.llama, prompt), tone: "text-text", bar: "bg-text/60" },
  { label: "tiled kernel", value: flashTtft(flashSystems.flash, prompt), tone: "text-accent", bar: "bg-accent" },
];

export function LengthPicker() {
  const [prompt, setPrompt] = useState(flashLong);
  const rows = pickerRows(prompt);
  const max = Math.max(...rows.map((r) => r.value));
  const tiled = rows[3].value;
  return (
    <div className="flex h-full flex-col gap-3 tabular-nums text-xs">
      <Predict {...prediction} />
      <Toggle label="prompt tokens" choices={flashLengths} value={prompt} onChange={setPrompt} />
      <div className="flex-1 space-y-3 pt-2">
        {rows.map((r) => (
          <div key={r.label} className="grid grid-cols-[6.5rem_1fr_5.5rem] items-center gap-2">
            <span className={r.tone}>{r.label}</span>
            <div className="h-3 min-w-0 rounded-full bg-border">
              <EasedBar fraction={r.value / max} className={r.bar} />
            </div>
            <span className={`text-right tabular-nums ${r.tone}`}>
              <NumberFlow value={r.value} suffix=" ms" format={{ minimumFractionDigits: 1, maximumFractionDigits: 1 }} />
            </span>
          </div>
        ))}
      </div>
      <div className="space-y-1 border-t border-border pt-2 leading-relaxed">
        <div>
          <span className="text-muted">tiled kernel, per prompt token </span>
          <span className="text-text">{`${flashPerToken(flashSystems.flash, prompt)} ms`}</span>
        </div>
        <div>
          <span className="text-muted">decode kernels / tiled kernel </span>
          <span className="text-text">{`${(rows[1].value / tiled).toFixed(1)}x`}</span>
        </div>
      </div>
    </div>
  );
}

const steps: StoryStep[] = [
  {
    id: "square",
    title: "the square",
    body: (
      <>
        <p>
          Prefill used the decode attention: one row per prompt token, one thread block per row and query head, and row t walks its whole context of t + 1 keys and values alone. Twice the prompt is twice the rows, each walking twice as far, so the reads grow with the square of the length. None of that work uses the tensor cores.
        </p>
        <p>
          It showed in the time to first token. At {flashLong} tokens the engine took <span className="text-warn">{n.behind.toFixed(1)} times</span> as long as llama.cpp, calculated from the two times measured in one session.
        </p>
      </>
    ),
  },
  {
    id: "spec",
    title: "a spec and tests first",
    body: (
      <>
        <p>
          Before any kernel there was a written spec and <span className="text-accent">{s.tests} tests</span>. The spec fixes the interface: one call handles every prefill chunk of a step. chunk_starts says which rows belong to which chunk, cached_lens how many tokens of that sequence are already in the KV cache, and there is one block table row per chunk instead of one per token.
        </p>
        <p>
          The tests pin what a faster kernel could break without anyone noticing. A row must not depend on the rows after it or on what else is in the call, bit for bit. And the kernel must not read cache slots outside its context, which the tests fill with NaN.
        </p>
      </>
    ),
  },
  {
    id: "first",
    title: "the first kernel",
    body: (
      <>
        <p>
          The first kernel used what the interface offered: it read the chunk&apos;s own keys and values straight from the QKV buffer instead of going through the cache, and it kept an online softmax. It also kept the shape of the decode kernels: one thread block per row and head, one key at a time, two barriers per key. Every test passed.
        </p>
        <p>
          Wired into the runner and measured, it was slower than the path it replaced: <span className="text-warn">{flashFmt(n.first)} ms</span> at {flashLong} tokens. Moving where the bytes come from saves nothing when every row still walks every key alone.
        </p>
      </>
    ),
  },
  {
    id: "tiles",
    title: "sixteen rows share a pass",
    body: (
      <>
        <p>
          The tiled kernel changes who walks. One warp owns <span className="text-accent">{s.tile}</span> query rows of one head and loads them once. The keys of the sequence then stream past in tiles of the same size, and every row of the query tile uses each key tile.
        </p>
        <p>
          A row must still see only its own past. Keys after a query are masked, and keys past the last row of the tile load as zeros, so no row depends on what follows it. Keys the sequence already has in the KV cache come through the block table, and the chunk&apos;s own come from the QKV buffer.
        </p>
      </>
    ),
  },
  {
    id: "mma",
    title: "one mma call",
    body: (
      <>
        <p>
          The scores of a tile are a small matrix multiply, and the tensor cores have an instruction for exactly that, mma.sync. One call takes {s.tile} query rows by {s.tile} dimensions and {s.tile} dimensions by {s.half} keys, and adds their product to <span className="text-accent">{s.tile} × {s.half}</span> scores kept in FP32.
        </p>
        <p>
          The operands are fragments: every lane of the warp holds a fixed few elements of each matrix in its own registers. A key tile goes through as two halves of {s.half} keys, and a head of {s.headDim} dimensions takes <span className="text-accent">{calls}</span> calls per half.
        </p>
      </>
    ),
  },
  {
    id: "fragment",
    title: "nothing moves between lanes",
    body: (
      <>
        <p>
          After the two halves, a lane holds the scores of its two query rows against four keys. That is exactly what the next product, weights times values, wants from that lane as its first operand. So the scores turn into weights where they are, and nothing is passed between lanes to set it up.
        </p>
        <p>
          In between sits the online softmax, once per row. Each score is rounded as in the other attention kernels and masked, then the row&apos;s maximum and sum are reduced over the <span className="text-accent">{s.lanesPerRow}</span> lanes that share the row, with two shuffles. The weights are rounded to BF16 and multiplied with the values through the same instruction.
        </p>
      </>
    ),
  },
  {
    id: "registers",
    title: "all of it in registers",
    body: (
      <>
        <p>
          The weighted values add up in FP32 and are scaled down when a row&apos;s maximum rises. A lane keeps <span className="text-accent">{accs}</span> of those accumulators in registers, next to its piece of the query tile. There is no shared memory and no barrier, and warps share nothing.
        </p>
        <p>
          Two details keep it that way. The value columns are interleaved, so the values a lane needs from one key sit side by side and load as whole words. And the kernel is a template on the head size, so its loops unroll and its arrays stay in registers.
        </p>
      </>
    ),
  },
  {
    id: "result",
    title: "the result",
    body: (
      <>
        <p>
          As the default, the tiled kernel takes one {flashLong}-token prompt to its first token in <span className="text-accent">{flashFmt(n.tiled)} ms</span>. The figure puts the per-row path, the first kernel and llama.cpp from the same session beside it.
        </p>
        <p>
          The work still grows with the square of the prompt: the kernel computes the same number of score and value products. But each key tile is read once per {s.tile} rows and the products run on the tensor cores, and the cost per prompt token stays flat, at most <span className="text-accent">{n.perTokenHigh} ms</span>.
        </p>
      </>
    ),
  },
  {
    id: "lengths",
    title: "every length",
    body: (
      <>
        <p>
          The gain is largest where the square was largest, <span className="text-accent">{n.ratio.toFixed(1)} times</span> less time at {flashLong} tokens, and it shrinks with the prompt. Pick a length on the right to see the four measured paths.
        </p>
        <p>
          llama.cpp&apos;s number is the time of its prompt decode calls with no sampler, and the engine&apos;s includes sampling the first token, so the two are not timed in quite the same way. The engine is ahead at every length measured.
        </p>
      </>
    ),
  },
  {
    id: "open",
    title: "what is still open",
    body: (
      <>
        <p>
          Nsight Compute says the tensor cores run at <span className="text-accent">{pf.step.tensor}%</span> of their peak rate and {pf.step.occupancy}% of the warp places are filled, nearly all that {pf.launchAfter.registers} registers per thread allow. The loads ask the caches for {flashFmt(Math.round(pf.step.requestedMb))} MB per layer to use {pf.neededMb.toFixed(1)} MB of keys and values; {pf.step.l1Hit}% of it is in L1 and DRAM is nearly idle. Staging the keys once in shared memory for the <span className="text-accent">{s.group}</span> query heads that share a KV head was not tried.
        </p>
        <p>
          Chunking a lone prompt with a budget of {budget} cost <span className="text-warn">{pf.plainBefore.cost}%</span>, most of it in the attention kernel: with {pf.launchBefore.registers} registers only {pf.launchBefore.blocksAtOnce} thread blocks fit on the GPU at once, and a {budget}-row chunk launches {pf.after.blocks}. Two divisions done in 32 bits brought it to {pf.launchAfter.registers} registers and {pf.launchAfter.blocksAtOnce} blocks, and the cost to <span className="text-accent">{pf.plainAfter.cost}%</span>. The load sweeps of the chunked prefill chapter have not been rerun.
        </p>
      </>
    ),
  },
];

const captions = [
  `key rows read per head for one ${flashLong}-token prompt, calculated from the prompt length and the tile size: ${flashFmt(keyRows.perRow)} with one walk per row, ${flashFmt(keyRows.tiled)} with one pass per ${s.tile} rows; measured, docs/results.md: ${flashFmt(n.paged)} ms to the first token against llama.cpp's ${flashFmt(n.llama)} ms`,
  `tests/gpu/test_flash_prefill.cpp: ${s.tests} tests, counting one test over ${s.layouts} head and block layouts ${s.layouts} times; the same file passes against both kernels`,
  `docs/results.md, FlashAttention prefill kernels: one request per run, median of 5, same session; ${flashFmt(n.first)} against ${flashFmt(pagedPathLong)} ms at ${flashLong} tokens, and ${flashPath(flashPaths.first, flashShort)} against ${flashPath(flashPaths.paged, flashShort)} ms at ${flashShort}; the row of keys is a sketch`,
  `a ${8 * s.tile}-token prompt drawn as 8 by 8 tiles of ${s.tile}; the grid of thread blocks is tiles of ${s.tile} rows by groups of ${s.warps} heads, from src/kernels/flash_prefill.cu`,
  `the fragment layout of mma.sync.aligned.m16n8k16 as the kernel uses it (src/kernels/flash_prefill.cu); 4 of the 32 lanes shown in turn; ${calls} calls per half calculated from the ${s.headDim}-dimension heads in config.json`,
  `from src/kernels/flash_prefill.cu: 4 of the 32 lanes shown in turn above, the ${s.lanesPerRow} lanes of query row 0 below; the weights are not yet divided by the sum, that happens once at the end`,
  `calculated from the kernel's fragment sizes at head_dim ${s.headDim}: ${qWords} words of Q and ${accs} accumulators per lane; a head of up to ${s.maxHeadDim} dimensions in steps of ${s.tile} is accepted`,
  `docs/results.md, Prefill against prompt length: one request per run, median of 5 runs, RTX 5060 Laptop GPU; per prompt token the tiled kernel stays between ${n.perTokenLow} and ${n.perTokenHigh} ms while the per-row path climbs from ${n.pagedPerTokenShort} to ${n.pagedPerTokenLong} ms; llama.cpp with -ub 512`,
  `the same medians; ${flashFmt(n.paged)} to ${flashFmt(n.tiled)} ms at ${flashLong} tokens is ${n.ratio.toFixed(1)} times less, and ${n.ahead.toFixed(0)}% less than llama.cpp's ${flashFmt(n.llama)} ms (calculated); the first kernel was measured at its own commit in the same session`,
  `docs/profiling.md, FlashAttention prefill in chunks: kernel time from Nsight Systems, median of 3 captures, with the kernel before and after the 32-bit divisions in one session; counters from Nsight Compute on one launch; ${pf.plainBefore.cost}% and ${pf.plainAfter.cost}% are ${pf.plainBefore.chunked} against ${pf.plainBefore.whole} ms and ${pf.plainAfter.chunked} against ${pf.plainAfter.whole} ms with no profiler, median of 5; ${pf.neededMb.toFixed(1)} MB and blocks at once are calculated`,
];

export const flashPrefillSteps = steps;
export const flashPrefillCaptions = captions;

export function FlashSquare() {
  return <SquareFigure tile={s.tile} perRow={keyRows.perRow} tiled={keyRows.tiled} tokens={flashLong} ms={n.paged} />;
}

export function FlashSpec() {
  return <SpecFigure tests={s.tests} groups={specGroups} />;
}

export function FlashFirst() {
  return <FirstKernelFigure paged={pagedPathLong} first={n.first} tokens={flashLong} loss={n.firstLoss} />;
}

export function FlashTiles() {
  return <TileSweepFigure mode="kernel" tile={s.tile} warps={s.warps} />;
}

export function FlashMma() {
  return <MmaFigure tile={s.tile} half={s.half} calls={calls} headDim={s.headDim} />;
}

export function FlashFragment() {
  return <FragmentFigure tile={s.tile} lanesPerRow={s.lanesPerRow} />;
}

export function FlashRegisters() {
  return <RegistersFigure headDim={s.headDim} qWords={qWords} accs={accs} group={s.group} />;
}

export function FlashTtft() {
  return <TtftFigure lengths={flashLengths.map(flashTokens)} series={series} />;
}

export function FlashOpen() {
  return <OpenFigure whole={pf.whole} before={pf.before} after={pf.after} atOnce={[pf.launchBefore.blocksAtOnce, pf.launchAfter.blocksAtOnce]} step={pf.step} last={[pf.lastBefore.occupancy, pf.lastAfter.occupancy]} budget={budget} chunks={chunks} tokens={flashLong} items={openItems} />;
}

const figures = [FlashSquare, FlashSpec, FlashFirst, FlashTiles, FlashMma, FlashFragment, FlashRegisters, FlashTtft, LengthPicker, FlashOpen];

export function FlashPrefillStory() {
  return (
    <StickyStory
      steps={steps}
      caption={(i) => captions[i]}
      figure={(i) => {
        const Figure = figures[i] ?? FlashOpen;
        return <Figure />;
      }}
    />
  );
}
