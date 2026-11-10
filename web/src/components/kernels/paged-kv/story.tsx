"use client";

import { useEffect, useMemo, useState } from "react";
import { measured } from "@/content/kernels/types";
import { kvReplay, modelShape } from "@/lib/data";
import { SmoothRange } from "@/components/smooth-range";
import { StickyStory, type StoryStep } from "../sticky-story";
import { useLoop } from "../loop";
import { chapterNumber } from "@/content/book-order";
import { AppendFigure, first, fullContext, heldAt, lookup, reqs, ReuseFigure, TableFigure, WhyBlocksFigure, WriteKvFigure, writeShare, writeUs } from "./figures";

const replay = kvReplay();
const B = replay.meta.block_size;
const layers = modelShape().layers;
const blockBytes = 360448;
const poolGb = (replay.meta.kv_blocks * blockBytes) / 1e9;
const tput = (batch: string) => measured("throughput-and-latency", { system: "engine + graphs", "max batch": batch }, "output tok/s", 0, "results");

const fmt = (v: number, d = 0) =>
  v.toLocaleString("en-US", {
    minimumFractionDigits: d,
    maximumFractionDigits: d,
  });

const cols = 12;
const shown = Math.ceil((replay.max_block + 1) / cols) * cols;
const W = 560;
const H = 266;
const cw = 520 / cols;
const ch = 22;
const chartTop = 200;
const chartH = 48;
const tEnd = replay.duration_s;
const maxHeld = replay.peak.blocks;
const tx = (t: number) => 20 + (t / tEnd) * 520;
const hy = (h: number) => chartTop + chartH - (h / maxHeld) * chartH;
const path = replay.steps.map((s, i) => `${i ? "L" : "M"}${tx(s[0]).toFixed(1)},${hy(s[2]).toFixed(1)}`).join("");
const tint = ["fill-accent", "fill-accent/55", "fill-baseline"];
const speed = 0.5;

export function stepAt(t: number) {
  let lo = 0;
  let hi = replay.steps.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (replay.steps[mid][0] <= t) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

function ReplayPlayer() {
  const { reduced } = useLoop(1);
  const [picked, setPicked] = useState<number | null>(null);
  const [now, setNow] = useState(0);
  const [paused, setPaused] = useState(false);
  const playing = !paused && !reduced;
  const [startAt, setStartAt] = useState(0);
  const step = picked ?? (reduced ? replay.peak.step : now);

  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    let t0 = performance.now();
    let start = startAt;
    const tick = (time: number) => {
      const t = start + ((time - t0) / 1000) * speed;
      if (t > tEnd + 1) {
        t0 = time;
        start = 0;
      }
      setNow(stepAt(Math.min(t > tEnd + 1 ? 0 : t, tEnd)));
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, startAt]);

  const held = useMemo(() => heldAt(step), [step]);
  const byId = new Map(held.map((b) => [b.id, b]));
  const [t, , blocks] = replay.steps[step];
  const running = new Set(held.map((b) => b.req)).size;
  const tokens = Math.round(held.reduce((n, b) => n + b.fill * B, 0));
  const stats = [
    { label: "requests running", value: running, cls: "text-text" },
    { label: "blocks held", value: blocks, cls: "text-accent" },
    {
      label: "slots filled",
      value: blocks ? (100 * tokens) / (blocks * B) : 0,
      cls: "text-accent",
      suffix: "%",
    },
    {
      label: `reserving ${fmt(fullContext)} each`,
      value: running * (fullContext / B),
      cls: "text-warn",
    },
  ];
  return (
    <div className="flex h-full flex-col gap-2 tabular-nums text-xs">
      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={() => {
            if (picked !== null) setNow(picked);
            setStartAt(replay.steps[step][0]);
            setPicked(null);
            setPaused(playing);
          }}
          className="pressable rounded border border-border px-2.5 py-1 text-accent"
          aria-pressed={playing}
        >
          {playing ? "pause" : "play"}
        </button>
        <SmoothRange
          min={0}
          max={replay.steps.length - 1}
          value={step}
          onValue={(v) => {
            setPaused(true);
            setPicked(v);
          }}
          aria-label="step of the recorded run"
          className="w-full accent-[var(--accent)]"
        />
        <span className="w-24 shrink-0 text-right tabular-nums text-muted">{`${t.toFixed(2)} s`}</span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={`The KV pool at step ${step} of the recorded run, ${t.toFixed(2)} s in: ${running} requests hold ${blocks} blocks`}>
        <text x={20} y={14} className="fill-muted text-[11px]">{`blocks 0 to ${shown - 1} of ${fmt(replay.meta.kv_blocks)}; the rest stay free all run`}</text>
        {Array.from({ length: shown }, (_, id) => {
          const b = byId.get(id);
          const x = 20 + (id % cols) * cw;
          const y = 24 + Math.floor(id / cols) * (ch + 3);
          return (
            <g key={id}>
              <rect x={x} y={y} width={cw - 3} height={ch} rx={2} className={id === 0 ? "fill-warn/20 stroke-warn/50" : "fill-bg stroke-border"} strokeWidth={0.75} />
              {b && <rect x={x} y={y} width={(cw - 3) * b.fill} height={ch} rx={2} className={tint[b.req % 3]} />}
              {b && (
                <text x={x + 4} y={y + 15} className="pointer-events-none fill-bg text-[10px]">
                  {b.req}
                </text>
              )}
            </g>
          );
        })}
        <text x={20} y={chartTop - 6} className="fill-muted text-[11px]">{`blocks held over the ${tEnd.toFixed(1)} s run (peak ${maxHeld})`}</text>
        <path d={path} className="fill-none stroke-accent" strokeWidth={1.25} />
        <line x1={20} x2={540} y1={chartTop + chartH} y2={chartTop + chartH} className="stroke-border" />
        <line x1={tx(t)} x2={tx(t)} y1={chartTop - 2} y2={chartTop + chartH} className="stroke-text" strokeWidth={1} />
      </svg>
      <div className="grid grid-cols-2 gap-x-4 gap-y-2 border-t border-border pt-2 tabular-nums sm:grid-cols-4">
        {stats.map((r) => (
          <div key={r.label}>
            <div className="text-muted">{r.label}</div>
            <div className={r.cls}>
              {`${fmt(r.value)}${r.suffix ?? ""}`}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

const reqBlocks = first.blocks.length;

export const pagedKvSteps: StoryStep[] = [
  {
    id: "why",
    title: "nobody knows a request's length, so hold only the blocks it fills",
    body: (
      <>
        <p>
          Every token a request has seen leaves a key and a value in every layer, and attention reads all of them again on every step. When a request arrives nobody knows how long it will
          get: a chat answer can stop after a few tokens or run to {fmt(fullContext)}. A cache with one contiguous row per request has to reserve the worst case up front.
        </p>
        <p>
          The first four requests of a recorded chat run would have reserved far more that way. Taken one block at a time, as
          they grew, they needed <span className="text-accent">{reqs.slice(0, 4).reduce((n, q) => n + q.blocks.length, 0)}</span> blocks. The idea is the one behind virtual memory: fixed-size
          pages, handed out on demand, wherever there is room.
        </p>
      </>
    ),
  },
  {
    id: "pool",
    title: "one pool, a table per request",
    body: (
      <>
        <p>
          At startup the engine takes a large share of the GPU in one allocation: <span className="text-text">{fmt(replay.meta.kv_blocks)}</span> blocks, each a block of tokens&apos; keys and
          values for every layer. The first block is never handed out, so a table entry that points at it can mean &quot;nothing here&quot;.
        </p>
        <p>
          The block manager keeps, per request, its length and its block table, the list of blocks it owns in order. A position of the first request falls in one of its logical blocks, the table names the physical block behind it, and so it lives in slot <span className="text-accent">{lookup.slot}</span>. The table comes straight from the
          run: the request&apos;s blocks are scattered across the pool between those of the requests running beside it.
        </p>
      </>
    ),
  },
  {
    id: "append",
    title: `a new block only every ${B} tokens`,
    body: (
      <>
        <p>
          A prefill takes blocks for the whole prompt at once: the first request&apos;s {first.prompt} prompt tokens fill its first few blocks. After that, every decode step appends one slot per running request. Most of the time the slot is in a block the request already holds; only when its length reaches a multiple of the block size does
          append_slot take a new one.
        </p>
        <p>
          The new block is the lowest free one. Over its decode steps the request asked for a block <span className="text-accent">{reqBlocks - Math.ceil(first.prompt / B)}</span> times, and the
          most a request ever leaves empty is the rest of its last block, one slot short of a full block.
        </p>
      </>
    ),
  },
  {
    id: "reuse",
    title: "freed blocks go out again first",
    body: (
      <>
        <p>
          When a request finishes, every block in its table goes back to the pool, and because the lowest free id goes out first, the next request that needs a block takes those. In the recorded run
          one of the first request&apos;s blocks had already served another request. Every request of the run fit in the first{" "}
          <span className="text-accent">{replay.max_block}</span> blocks after the null block.
        </p>
        <p>
          If the pool does run out, the scheduler takes a running request&apos;s blocks back and recomputes its tokens later. Which request it picks, what that request loses and why its output does not change is chapter {chapterNumber("serving")}.
        </p>
      </>
    ),
  },
  {
    id: "write",
    group: "how it got built",
    title: "write_kv is a pure copy, one thread per value",
    body: (
      <>
        <p>
          write_kv is the kernel that fills the slots. After the QKV projection and RoPE, each token&apos;s row holds its query heads and then its key heads and value heads. write_kv runs one
          thread per (token, KV head, dimension), looks up the token&apos;s slot, and copies one key and one value bit for bit into the block and the row inside it that the slot names.
        </p>
        <p>
          The cache is laid out by block, then KV head, then token, then dimension, so one head&apos;s keys in a block are one contiguous tile, which is what the attention kernel reads. Padding rows get
          an empty slot and write nothing. It is a pure copy: <span className="text-accent">{`${fmt(writeUs, 1)} µs`}</span> per single-request decode step over one launch per layer, a small share of the kernel
          time.
        </p>
        <p>
          The layout and the kernel&apos;s tests came before the kernel, and the tests were checked against nine deliberately broken kernels first. Joe&apos;s kernel then passed them, and one of them caught two slips in the spec: a QKV row with no query columns was accepted, and an empty batch returned before its arguments were checked.
        </p>
      </>
    ),
  },
  {
    id: "replay",
    group: "measured",
    title: `a busy run held ${replay.peak.blocks} of ${fmt(replay.meta.kv_blocks)} blocks`,
    body: (
      <>
        <p>
          On the right is the whole recorded run, replayed through the same allocation rules at half speed: chat requests arriving at a steady rate,
          up to {replay.max_running} at once. Each block shows the request that holds it and how full it is. Drag the slider to stop at any step.
        </p>
        <p>
          At its busiest the run held <span className="text-accent">{replay.peak.blocks}</span> blocks, a tiny part of the pool. The pool is simply most of the memory left after
          the weights, so a run like this leaves almost all of it free, room for many more or much longer requests. Serving them side by side from one pool is what batching needs: a full batch serves
          many times the tokens per second of a single request (CUDA Graphs on, both). How the scheduler decides which requests share a step, and how many fit, is chapter {chapterNumber("serving")}.
        </p>
      </>
    ),
  },
];

export const pagedKvCaptions = [
  `the four requests' real lengths from ${replay.file}; blocks are ${B} tokens; ${layers} layers; an answer can stop after 9 tokens; reserving ${fmt(fullContext)} tokens each, the four would hold ${4 * (fullContext / B)} blocks`,
  `block table of request ${first.id} and the pool at that step, both from the build-time replay of ${replay.file}: position ${lookup.pos} is logical block ${lookup.logical}, physical block ${lookup.phys}, among the request's ${reqBlocks} blocks; the pool is ${fmt(poolGb, 1)} GB, ${fmt(blockBytes)} bytes and ${B} tokens a block, block 0 the null block`,
  `request ${first.id}'s real prompt and blocks: its ${first.prompt} prompt tokens fill blocks ${first.blocks.slice(0, Math.ceil(first.prompt / B)).map((b) => b.id).join(", ")}, then ${first.final_length - first.prompt} decode steps; at most ${B - 1} slots stay empty; the slots filled over time are drawn, not timed`,
  `block reuse from the replay: request ${first.id}'s fourth block had served another request, and all ${replay.meta.num_requests} requests stayed within blocks 1 to ${replay.max_block}; the preemption test, ForcedPreemptionKeepsTheOutputsOfEveryRequest in tests/gpu/test_serve.cpp, puts 16 requests in 56 blocks and forces 6 preemptions`,
  `TinyLlama: 32 query heads and 4 KV heads of 64 dims, laid out [block, KV head, ${B}, 64] per layer, a 2 KB tile per head and block; the slot picks row slot mod ${B} of block slot / ${B}, padding rows get slot -1; time from results/decode_step.md (GPU time per op), ${layers} launches, ${writeShare}% of the kernel time`,
  `${replay.file}, recorded ${replay.meta.date.slice(0, 10)} on the ${replay.meta.gpu.replace("NVIDIA GeForce ", "")}: ${replay.meta.num_requests} requests at ${replay.meta.request_rate} per second, peak ${replay.peak.blocks} of ${fmt(replay.meta.kv_blocks)} blocks, the pool 90% of the memory left after the weights; measured ${fmt(tput("16"))} tokens/s at batch 16 against ${fmt(tput("1"))} at batch 1; every step's free block count checked against the trace at build time`,
];

export function PagedKvStory() {
  return (
    <StickyStory
      steps={pagedKvSteps}
      caption={(i) => pagedKvCaptions[i]}
      figure={(i) => {
        switch (pagedKvSteps[i].id) {
          case "why":
            return <WhyBlocksFigure />;
          case "pool":
            return <TableFigure />;
          case "write":
            return <WriteKvFigure launchUs={writeUs} share={writeShare} />;
          case "append":
            return <AppendFigure />;
          case "reuse":
            return <ReuseFigure />;
          default:
            return <ReplayPlayer />;
        }
      }}
    />
  );
}
