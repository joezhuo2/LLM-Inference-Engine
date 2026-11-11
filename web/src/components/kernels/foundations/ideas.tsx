"use client";

import { motion } from "motion/react";
import { ceilings, hardware, kvReplay, medianOf, modelShape } from "@/lib/data";
import { Frame, Label } from "../frame";
import { useLoop } from "../loop";
import { StickyStory, type StoryStep } from "../sticky-story";

const m = modelShape();
const peak = ceilings();
const hw = hardware();
const pool = kvReplay().meta;

const headDim = m.hidden / m.heads;
const layerParams = m.hidden * (m.hidden + 2 * m.kv_heads * headDim) + m.hidden * m.hidden + m.hidden * 2 * m.intermediate + m.intermediate * m.hidden;
const readParams = m.layers * layerParams + m.vocab * m.hidden;
const readBytes = readParams * 2;
const storedBytes = (readParams + m.vocab * m.hidden) * 2;
const readMs = (readBytes / (peak.gbs * 1e9)) * 1e3;
const mathMs = ((2 * readParams) / (peak.tflops * 1e12)) * 1e3;
const kvBytesPerToken = m.layers * 2 * m.kv_heads * headDim * 2;
const poolTokens = pool.kv_blocks * pool.block_size;
const poolBytes = poolTokens * kvBytesPerToken;
const chatTokens = 200;
const chatBlocks = Math.ceil(chatTokens / pool.block_size);
const chatsFit = poolTokens / (chatBlocks * pool.block_size);
const reserveTokens = 2048;
const cacheSteps = 8;
const uncachedWork = (cacheSteps * (cacheSteps + 1)) / 2;

const need = (v: number | undefined, what: string) => {
  if (v === undefined) throw new Error(`results/summaries has no ${what}`);
  return v;
};
const batchRows = [1, 4, 16].map((batch) => ({ batch, tps: need(medianOf("engine_graphs", batch, "output_throughput"), `engine_graphs output_throughput at batch ${batch}`), tpot: need(medianOf("engine_graphs", batch, "tpot_s_p50"), `engine_graphs tpot at batch ${batch}`) * 1e3 }));
const b1 = batchRows[0];
const b16 = batchRows[2];

const gb = (bytes: number) => (bytes / 1e9).toFixed(2);
const ms = (v: number, d = 1) => v.toFixed(d);
const int = (v: number) => Math.round(v).toLocaleString("en-US");

export const ideaNumbers = { readBytes, readMs, mathMs, measuredMs: b1.tpot, kvBytesPerToken, poolTokens, poolBytes, storedBytes, vram: hw.vram_gb, b1, b16, batchRows };

const ideaSteps: StoryStep[] = [
  {
    id: "idea-memory",
    title: "decode waits on memory",
    body: (
      <>
        <p>
          To produce one token, the GPU has to read every weight of the model once: all the layers, then the LM head. For TinyLlama that is <span className="text-accent">{gb(readBytes)} GB</span>. At the bandwidth this GPU&apos;s memory can deliver, the reading alone takes several milliseconds.
        </p>
        <p>
          The arithmetic on those weights would take a small fraction of a millisecond if the numbers were already there. So the measured step, {ms(b1.tpot)} ms, is almost all waiting for bytes. Making the maths faster would change nothing; moving fewer bytes, or using each byte for more work, is what helps.
        </p>
        <p>The rest of this chapter builds this into the roofline, and every later chapter is judged against it.</p>
      </>
    ),
  },
  {
    id: "idea-cache",
    title: "the KV cache remembers",
    body: (
      <>
        <p>
          Attention lets the newest token look back at every earlier one. Each token gets a query, what it is looking for, a key, what it offers, and a value, what it passes on. The new token compares its query with the key of every earlier token and mixes their values by how well they match.
        </p>
        <p>
          The keys and values of earlier tokens never change, so the engine keeps them: the KV cache. Each step then computes K and V only for the new token and reads the rest, instead of recomputing all of them. The figure counts the computations over eight steps, and the gap grows with every token.
        </p>
        <p>
          The price is memory: <span className="text-accent">{int(kvBytesPerToken)} bytes</span> for every token of every conversation. The paged KV cache and attention chapters are about storing and reading this cache well.
        </p>
      </>
    ),
  },
  {
    id: "idea-batch",
    title: "batching shares the read",
    body: (
      <>
        <p>
          If reading the weights is the cost, read them once and use them for many requests. A batch of 16 conversations runs as one step: the weights stream in once, and each row does its own arithmetic on them.
        </p>
        <p>
          The step gets only a little longer, because the arithmetic was nearly free. The output rises to <span className="text-accent">{int(b16.tps)} tokens per second</span>, many times what one request gets.
        </p>
        <p>This is why serving engines batch everything they can, and why the scheduler in the paged KV cache chapter exists.</p>
      </>
    ),
  },
  {
    id: "idea-budget",
    title: "memory is the budget",
    body: (
      <>
        <p>
          The weights take a share of the GPU&apos;s memory, and most of the rest becomes the KV cache: a pool of <span className="text-accent">{int(poolTokens)} token slots</span>. How many conversations fit at once is decided by that pool, not by the arithmetic.
        </p>
        <p>
          Reserving the longest possible answer for each request would fit far fewer of them, most of each reservation empty. Handing out small blocks only as a conversation grows fits about <span className="text-accent">{int(chatsFit)} chats</span> of a short length. That is the paged KV cache.
        </p>
      </>
    ),
  },
  {
    id: "idea-latency",
    title: "batching trades latency for throughput",
    body: (
      <>
        <p>
          Bigger batches give more tokens per second in total, but each conversation waits a little longer for its next token, {ms(b16.tpot)} ms in a full batch. An engine is always trading the speed one user feels against how many users it can serve.
        </p>
        <p>
          A long new prompt makes the trade worse. Its prefill is one big step, and every conversation already decoding waits for it to finish. Chunked prefill, in Part IV, cuts the prompt into pieces that ride along with the decode steps, so <span className="text-accent">nobody stalls</span>. CUDA Graphs shave the time between kernels so every step is shorter.
        </p>
      </>
    ),
  },
];

export const ideaCaptions = [
  `bytes and FLOPs from config.json; ${peak.gbs} GB/s and ${peak.tflops} TFLOP/s measured on the ${hw.gpu.replace("NVIDIA GeForce ", "")}; reading ${ms(readMs)} ms and arithmetic ${ms(mathMs, 2)} ms calculated; the step is the median time per output token of engine + CUDA Graphs at batch 1`,
  `illustrative: ${cacheSteps} tokens, one head; K and V computed over ${cacheSteps} steps: ${cacheSteps} with a cache, ${uncachedWork} without; ${int(kvBytesPerToken)} bytes = ${m.layers} layers × K and V × ${m.kv_heads} KV heads × ${headDim} values × 2 bytes`,
  `measured medians of engine + CUDA Graphs runs from docs/results.md: ${int(b1.tps)} tokens per second and ${ms(b1.tpot)} ms per output token at batch 1, ${int(b16.tps)} and ${ms(b16.tpot)} ms at batch 16, ${(b16.tps / b1.tps).toFixed(0)} times the tokens per second`,
  `${hw.vram_gb.toFixed(1)} GB from docs/hardware.md, weights ${gb(storedBytes)} GB; the pool is the ${int(pool.kv_blocks)} blocks of ${pool.block_size} tokens the engine allocated in the recorded chat run; calculated: ${int(poolTokens / reserveTokens)} reservations of ${int(reserveTokens)} tokens, or about ${int(chatsFit)} chats of ${chatTokens} tokens; reservations drawn to scale within each row`,
  `illustrative: three conversations decoding while a long prompt arrives, step lengths not to scale; measured medians: ${ms(b1.tpot)} ms per token alone, ${ms(b16.tpot)} ms at batch 16`,
];

export const ideaInstructions = [
  "Watch the read bar fill while the arithmetic finishes almost at once.",
  "Count the cells computed in each triangle.",
  "Watch one weight read feed one row, then 16.",
  "Compare the empty space in the two rows.",
  "Watch the three conversations stall, then keep going.",
];

const ink = "fill-text";
const W = 560;

function MemoryFigure() {
  const { on, grow } = useLoop(12);
  const x0 = 40;
  const scale = 470 / 8;
  const read = readMs * scale;
  const math = Math.max(3, mathMs * scale);
  const step = b1.tpot * scale;
  const cell = 17;
  return (
    <Frame label={`Reading ${gb(readBytes)} GB of weights takes ${ms(readMs)} ms; the arithmetic takes ${ms(mathMs, 2)} ms; the measured step takes ${ms(b1.tpot)} ms`}>
      <Label x={20} y={30} className={ink}>
        one token at batch 1: every layer&apos;s weights stream in from DRAM
      </Label>
      {Array.from({ length: m.layers }, (_, i) => (
        <g key={i}>
          <rect x={x0 + i * (cell + 4)} y={50} width={cell} height={cell} rx={3} className="fill-baseline/40" />
          <motion.rect x={x0 + i * (cell + 4)} y={50} width={cell} height={cell} rx={3} className="fill-accent" {...on(0.05 + (i / m.layers) * 0.5, 0.94)} />
        </g>
      ))}
      <Label x={x0} y={88}>
        layer 1
      </Label>
      <Label x={x0 + (m.layers - 1) * (cell + 4) + cell} y={88} anchor="end">
        layer 22, then the LM head
      </Label>
      <Label x={x0} y={130} className={ink}>{`reading the weights: ${gb(readBytes)} GB ÷ ${peak.gbs} GB/s`}</Label>
      <rect x={x0} y={140} width={read} height={26} rx={4} className="fill-baseline/20" />
      <motion.rect x={x0} y={140} width={read} height={26} rx={4} className="fill-accent" {...grow(0.05, 0.55)} />
      <Label x={x0 + read + 8} y={158} className={ink}>{`${ms(readMs)} ms`}</Label>
      <Label x={x0} y={200} className={ink}>{`the arithmetic: ${(2 * readParams / 1e9).toFixed(1)} GFLOP ÷ ${peak.tflops} TFLOP/s`}</Label>
      <rect x={x0} y={210} width={math} height={26} rx={2} className="fill-text" />
      <Label x={x0 + math + 8} y={228} className={ink}>{`${ms(mathMs, 2)} ms`}</Label>
      <Label x={x0} y={270} className={ink}>
        the measured decode step
      </Label>
      <motion.rect x={x0} y={280} width={step} height={26} rx={4} className="fill-baseline/60" {...grow(0.58, 0.72)} />
      <motion.g {...on(0.72, 0.94)}>
        <Label x={x0 + step - 8} y={298} anchor="end" className="fill-bg">{`${ms(b1.tpot)} ms`}</Label>
      </motion.g>
      {[0, 2, 4, 6, 8].map((t) => (
        <g key={t}>
          <line x1={x0 + t * scale} x2={x0 + t * scale} y1={318} y2={324} className="stroke-muted" />
          <Label x={x0 + t * scale} y={338} anchor="middle">{`${t} ms`}</Label>
        </g>
      ))}
      <motion.g {...on(0.6, 0.94)}>
        <Label x={20} y={378} className={ink}>{`The maths units sit idle for about ${Math.round((1 - mathMs / b1.tpot) * 100)}% of the step, waiting for bytes.`}</Label>
      </motion.g>
    </Frame>
  );
}

function Triangle({ x, cached, on }: { x: number; cached: boolean; on: ReturnType<typeof useLoop>["on"] }) {
  const n = 8;
  const c = 22;
  const g = 4;
  const y0 = 80;
  return (
    <g>
      <Label x={x} y={60} className={ink}>
        {cached ? "with a KV cache" : "without a cache"}
      </Label>
      {Array.from({ length: n }, (_, s) => (
        <motion.g key={s} {...on(0.04 + (s / n) * 0.6, 0.94)}>
          <Label x={x - 8} y={y0 + s * (c + g) + 15} anchor="end">{`step ${s + 1}`}</Label>
          {Array.from({ length: s + 1 }, (_, t) => {
            const fresh = !cached || t === s;
            return <rect key={t} x={x + t * (c + g)} y={y0 + s * (c + g)} width={c} height={c} rx={3} className={fresh ? "fill-accent" : "fill-baseline/40"} />;
          })}
        </motion.g>
      ))}
      <motion.g {...on(0.68, 0.94)}>
        <Label x={x} y={y0 + n * (c + g) + 22} className={ink}>{`K and V computed: ${cached ? n : (n * (n + 1)) / 2}`}</Label>
      </motion.g>
    </g>
  );
}

function CacheFigure() {
  const { on } = useLoop(13);
  return (
    <Frame label="Without a cache every step recomputes K and V for all earlier tokens, 36 in 8 steps; with a cache each step computes only the new token's, 8 in 8 steps">
      <Label x={20} y={28} className={ink}>
        columns are tokens; each row is one decode step
      </Label>
      <Triangle x={60} cached={false} on={on} />
      <Triangle x={330} cached on={on} />
      <rect x={60} y={340} width={14} height={14} rx={3} className="fill-accent" />
      <Label x={80} y={351}>
        K and V computed this step
      </Label>
      <rect x={260} y={340} width={14} height={14} rx={3} className="fill-baseline/40" />
      <Label x={280} y={351}>
        read from the cache
      </Label>
      <Label x={20} y={385} className={ink}>{`the cache costs ${int(kvBytesPerToken)} bytes per token`}</Label>
    </Frame>
  );
}

function BatchFigure() {
  const { on, grow } = useLoop(13);
  const rowsY = 60;
  const maxTps = b16.tps;
  return (
    <Frame label={`One read of the weights serves 1 row or 16; tokens per second ${batchRows.map((r) => `${int(r.tps)} at batch ${r.batch}`).join(", ")}`}>
      <Label x={20} y={30} className={ink}>{`one step reads the ${gb(readBytes)} GB of weights once, for every row in the batch`}</Label>
      <rect x={40} y={rowsY} width={70} height={150} rx={6} className="fill-accent/80" />
      <Label x={75} y={rowsY + 80} anchor="middle" className="fill-bg">
        weights
      </Label>
      <motion.g {...on(0.02, 0.4)}>
        <line x1={110} y1={rowsY + 75} x2={200} y2={rowsY + 75} className="stroke-accent" strokeWidth={2} />
        <rect x={200} y={rowsY + 66} width={160} height={18} rx={3} className="fill-text" />
        <Label x={370} y={rowsY + 79} className={ink}>
          1 request
        </Label>
      </motion.g>
      <motion.g {...on(0.44, 0.94)}>
        {Array.from({ length: 16 }, (_, i) => (
          <g key={i}>
            <line x1={110} y1={rowsY + 75} x2={200} y2={rowsY + 4 + i * 9} className="stroke-accent/50" strokeWidth={1} />
            <rect x={200} y={rowsY + i * 9} width={160} height={7} rx={2} className="fill-text" />
          </g>
        ))}
        <Label x={370} y={rowsY + 79} className={ink}>
          16 requests, same read
        </Label>
      </motion.g>
      <Label x={20} y={250} className={ink}>
        output tokens per second, measured
      </Label>
      {batchRows.map((r, i) => {
        const w = (r.tps / maxTps) * 280;
        const y = 266 + i * 38;
        return (
          <g key={r.batch}>
            <Label x={90} y={y + 18} anchor="end">{`batch ${r.batch}`}</Label>
            <rect x={100} y={y} width={280} height={24} rx={4} className="fill-baseline/15" />
            <motion.rect x={100} y={y} width={w} height={24} rx={4} className={r.batch === 16 ? "fill-accent" : "fill-baseline/70"} {...grow(0.1 + i * 0.25, 0.3 + i * 0.25)} />
            <motion.g {...on(0.3 + i * 0.25, 0.94)}>
              <Label x={100 + w + 8} y={y + 17} className={ink}>{`${int(r.tps)} tok/s · ${ms(r.tpot)} ms a step`}</Label>
            </motion.g>
          </g>
        );
      })}
    </Frame>
  );
}

const lens = [180, 90, 260];

function pack(x0: number, full: number) {
  const cells: { x: number; k: number; req: number }[] = [];
  let k = 0;
  let x = x0;
  let whole = 0;
  for (let req = 0; x + 9 <= x0 + full; req++) {
    const blocks = Math.ceil(lens[req % 3] / 16);
    let b = 0;
    for (; b < blocks && x + 9 <= x0 + full; b++) {
      cells.push({ x, k: k++, req });
      x += 10;
    }
    if (b === blocks) whole++;
  }
  return { cells, whole };
}

function BudgetFigure() {
  const { on, grow } = useLoop(14);
  const x0 = 40;
  const full = 480;
  const vram = hw.vram_gb * 1e9;
  const wW = (storedBytes / vram) * full;
  const kW = (poolBytes / vram) * full;
  const reserve = 150;
  const paged = pack(x0, full);
  return (
    <Frame label={`${hw.vram_gb.toFixed(1)} GB: ${gb(storedBytes)} GB of weights and ${gb(poolBytes)} GB of KV cache; reserving 2,048 tokens per request leaves most of each reservation empty, 16-token blocks pack requests tightly`}>
      <Label x={20} y={30} className={ink}>{`the GPU's ${hw.vram_gb.toFixed(1)} GB`}</Label>
      <rect x={x0} y={42} width={full} height={34} rx={5} className="fill-border" />
      <rect x={x0} y={42} width={wW} height={34} rx={5} className="fill-text" />
      <rect x={x0 + wW} y={42} width={kW} height={34} className="fill-accent/70" />
      <Label x={x0 + 8} y={64} className="fill-bg">{`weights ${gb(storedBytes)} GB`}</Label>
      <Label x={x0 + wW + 8} y={64} className="fill-bg">{`KV cache ${gb(poolBytes)} GB: ${int(poolTokens)} tokens`}</Label>
      <Label x={x0 + full} y={94} anchor="end">
        the rest: activations, workspace, the driver
      </Label>
      <Label x={20} y={140} className={ink}>
        reserve the longest answer, 2,048 tokens, per request
      </Label>
      {lens.map((len, i) => {
        const x = x0 + i * (reserve + 10);
        return (
          <g key={i}>
            <rect x={x} y={152} width={reserve} height={30} rx={4} className="fill-warn/15 stroke-warn/60" strokeWidth={1} strokeDasharray="4 3" />
            <motion.rect x={x} y={152} width={(len / 2048) * reserve} height={30} rx={4} className="fill-accent" {...grow(0.05, 0.35)} />
            <Label x={x + 4} y={200}>{`${len} used`}</Label>
          </g>
        );
      })}
      <Label x={x0} y={226}>
        three requests fill the row; the dashed space is reserved and never used
      </Label>
      <Label x={20} y={270} className={ink}>
        paged: a 16-token block only when a request needs one
      </Label>
      {paged.cells.map((c) => (
        <motion.rect key={c.k} x={c.x} y={282} width={9} height={30} rx={2} className={c.req % 2 ? "fill-accent/60" : "fill-accent"} {...on(0.38 + (c.k / paged.cells.length) * 0.4, 0.94)} />
      ))}
      <motion.g {...on(0.8, 0.94)}>
        <Label x={x0} y={334}>
          {`the same row holds ${paged.whole} requests, packed block by block`}
        </Label>
      </motion.g>
      <Label x={20} y={378} className={ink}>{`whole pool: ${int(poolTokens / 2048)} reservations of 2,048, or about ${int(poolTokens / (chatBlocks * pool.block_size))} chats of ${chatTokens} tokens`}</Label>
    </Frame>
  );
}

function LatencyFigure() {
  const { on } = useLoop(14);
  const x0 = 110;
  const stepW = 34;
  const lanes = ["chat A", "chat B", "chat C"];
  const laneY = (i: number) => 70 + i * 34;
  const row = (y0: number, chunked: boolean, start: number) => {
    const marks: { lane: number; step: number }[] = [];
    const prefill = chunked ? [] : [2, 3, 4, 5];
    for (let s = 0; s < 12; s++) if (!prefill.includes(s)) for (let l = 0; l < 3; l++) marks.push({ lane: l, step: s });
    return (
      <g transform={`translate(0 ${y0})`}>
        {lanes.map((name, i) => (
          <Label key={name} x={x0 - 12} y={laneY(i) + 14} anchor="end">
            {name}
          </Label>
        ))}
        <Label x={x0 - 12} y={laneY(3) + 14} anchor="end">
          new prompt
        </Label>
        {marks.map((mk) => (
          <motion.rect key={`${mk.lane}-${mk.step}`} x={x0 + mk.step * stepW + 4} y={laneY(mk.lane) + 2} width={stepW - 8} height={16} rx={3} className="fill-text" {...on(start + (mk.step / 12) * 0.36, start + 0.44)} />
        ))}
        {chunked
          ? [2, 3, 4, 5].map((s) => <motion.rect key={s} x={x0 + s * stepW + 4} y={laneY(3) + 2} width={stepW - 8} height={16} rx={3} className="fill-accent" {...on(start + (s / 12) * 0.36, start + 0.44)} />)
          : (
            <>
              <motion.rect x={x0 + 2 * stepW + 4} y={laneY(3) + 2} width={4 * stepW - 8} height={16} rx={3} className="fill-accent" {...on(start + (2 / 12) * 0.36, start + 0.44)} />
              <motion.rect x={x0 + 2 * stepW} y={laneY(0) - 4} width={4 * stepW} height={3 * 34} rx={4} className="fill-warn/15 stroke-warn/60" strokeDasharray="4 3" {...on(start + 0.14, start + 0.44)} />
              <motion.g {...on(start + 0.16, start + 0.44)}>
                <Label x={x0 + 4 * stepW} y={laneY(1) + 14} anchor="middle" className="fill-warn">
                  everyone waits
                </Label>
              </motion.g>
            </>
          )}
      </g>
    );
  };
  return (
    <Frame label={`A long prompt's prefill stalls three conversations for four steps; chunked prefill spreads it over the steps so nobody stalls. Batch 16 gives ${(b16.tps / b1.tps).toFixed(0)} times the tokens per second, each token ${ms(b16.tpot)} ms instead of ${ms(b1.tpot)} ms`}>
      <Label x={20} y={30} className={ink}>{`batch 16: ${(b16.tps / b1.tps).toFixed(0)}× the tokens per second, but each token takes ${ms(b16.tpot)} ms, not ${ms(b1.tpot)}`}</Label>
      <motion.g {...on(0.02, 0.48)}>
        <Label x={20} y={52} className={ink}>
          a long prompt as one prefill step
        </Label>
        {row(0, false, 0.02)}
      </motion.g>
      <motion.g {...on(0.5, 0.97)}>
        <Label x={20} y={52} className={ink}>
          chunked prefill: the prompt rides along in pieces
        </Label>
        {row(0, true, 0.5)}
      </motion.g>
      <Label x={x0} y={laneY(4) + 26}>
        each column is one step; every bar is one token or one piece of the prompt
      </Label>
      <line x1={x0} x2={x0 + 12 * stepW} y1={laneY(4) + 6} y2={laneY(4) + 6} className="stroke-muted" />
      <Label x={x0 + 12 * stepW} y={laneY(4) + 44} anchor="end">
        time →
      </Label>
      <Label x={20} y={385} className={ink}>
        bigger batches and fair prefill trade one user&apos;s speed against how many are served
      </Label>
    </Frame>
  );
}

const figures = [MemoryFigure, CacheFigure, BatchFigure, BudgetFigure, LatencyFigure];

export function IdeaFigure({ index }: { index: number }) {
  const Figure = figures[index] ?? MemoryFigure;
  return <Figure />;
}

export const foundationsIdeas = ideaSteps;

export function FoundationsIdeasStory() {
  return <StickyStory steps={ideaSteps} kicker="big idea" caption={(i) => ideaCaptions[i]} figure={(i) => <IdeaFigure index={i} />} />;
}

export const ideasWidth = W;
