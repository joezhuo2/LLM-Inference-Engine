"use client";

import { motion } from "motion/react";
import { useEffect, useMemo, useState } from "react";
import { measured } from "@/content/kernels/types";
import { kvReplay } from "@/lib/data";
import { StickyStory, type StoryStep } from "../sticky-story";
import { useLoop } from "../loop";
import { AppendFigure, first, fullContext, heldAt, lookup, reqs, ReuseFigure, TableFigure, WhyBlocksFigure, WriteKvFigure } from "./figures";

const replay = kvReplay();
const B = replay.meta.block_size;
const blockBytes = 360448;
const poolGb = (replay.meta.kv_blocks * blockBytes) / 1e9;
const writeUs = measured("gpu-time-per-op-in-a-decode-step", { op: "write_kv" }, "GPU us/step", 0, "results");
const writeShare = measured("gpu-time-per-op-in-a-decode-step", { op: "write_kv" }, "% of kernel time", 0, "results");
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

function stepAt(t: number) {
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
  const ease = { duration: reduced ? 0 : 0.35, ease: "easeInOut" as const };
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
    <div className="flex h-full flex-col gap-2 font-mono text-xs">
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
        <input
          type="range"
          min={0}
          max={replay.steps.length - 1}
          value={step}
          onChange={(e) => {
            setPaused(true);
            setPicked(Number(e.target.value));
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
              <motion.rect
                x={x}
                y={y}
                height={ch}
                rx={2}
                className={b ? tint[b.req % 3] : "fill-accent"}
                initial={false}
                animate={{
                  width: b ? (cw - 3) * b.fill : 0,
                  opacity: b ? 1 : 0,
                }}
                transition={ease}
              />
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
        <motion.line y1={chartTop - 2} y2={chartTop + chartH} className="stroke-text" strokeWidth={1} initial={false} animate={{ x1: tx(t), x2: tx(t) }} transition={{ duration: 0 }} />
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

const steps: StoryStep[] = [
  {
    id: "why",
    title: "memory for a context nobody knows yet",
    body: (
      <>
        <p>
          Every token a request has seen leaves a key and a value in each of the 22 layers, and attention reads all of them again on every step. When a request arrives nobody knows how long it will
          get: a chat answer can stop after 9 tokens or run to {fmt(fullContext)}. A cache with one contiguous row per request has to reserve the worst case up front.
        </p>
        <p>
          The first four requests of a recorded chat run would have reserved <span className="font-mono text-warn">{`${4 * (fullContext / B)} blocks`}</span> that way. Taken 16 tokens at a time, as
          they grew, they needed <span className="font-mono text-accent">{reqs.slice(0, 4).reduce((n, q) => n + q.blocks.length, 0)}</span>. The idea is the one behind virtual memory: fixed-size
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
          At startup the engine takes {fmt(poolGb, 1)} GB of the GPU in one allocation: <span className="font-mono text-text">{fmt(replay.meta.kv_blocks)}</span> blocks, each 16 tokens of keys and
          values for every layer, {fmt(blockBytes)} bytes. Block 0 is never handed out, so a table entry of 0 can mean &quot;nothing here&quot;.
        </p>
        <p>
          The block manager keeps, per request, its length and its block table, the list of blocks it owns in order. Position {lookup.pos} of request {first.id} is in its logical block{" "}
          {lookup.logical}, the table says that is physical block {lookup.phys}, so it lives in slot <span className="font-mono text-accent">{lookup.slot}</span>. The table comes straight from the
          run: request {first.id}&apos;s {reqBlocks} blocks are scattered across the pool between those of the requests running beside it.
        </p>
      </>
    ),
  },
  {
    id: "write",
    title: "write_kv",
    body: (
      <>
        <p>
          write_kv is the kernel that fills the slots. After the QKV projection and RoPE, each token&apos;s row holds its query heads and then its 4 key heads and 4 value heads. write_kv runs one
          thread per (token, KV head, dimension), looks up the token&apos;s slot, and copies one key and one value bit for bit into row slot mod 16 of block slot / 16.
        </p>
        <p>
          The cache is laid out [block, KV head, 16, 64] per layer, so one head&apos;s 16 keys in a block are one contiguous 2 KB tile, which is what the attention kernel reads. Padding rows get slot
          -1 and write nothing. It is a pure copy: <span className="font-mono text-text">{`${fmt(writeUs, 1)} µs`}</span> per batch-1 decode step over its 22 launches, {writeShare}% of the kernel
          time.
        </p>
      </>
    ),
  },
  {
    id: "append",
    title: "one token at a time",
    body: (
      <>
        <p>
          A prefill takes blocks for the whole prompt at once: request {first.id}&apos;s {first.prompt} tokens fill blocks{" "}
          {first.blocks
            .slice(0, Math.ceil(first.prompt / B))
            .map((b) => b.id)
            .join(", ")}
          . After that, every decode step appends one slot per running request. Most of the time the slot is in a block the request already holds; only when its length reaches a multiple of 16 does
          append_slot take a new one.
        </p>
        <p>
          The new block is the lowest free one. Over its {first.final_length - first.prompt} decode steps request {first.id} asked for a block {reqBlocks - Math.ceil(first.prompt / B)} times, and the
          most a request ever leaves empty is the rest of its last block, 15 slots.
        </p>
      </>
    ),
  },
  {
    id: "reuse",
    title: "giving blocks back",
    body: (
      <>
        <p>
          When a request finishes, every block in its table goes back to the pool, and because the lowest free id goes out first, the next request that needs a block takes those. In the recorded run
          request {first.id}
          &apos;s fourth block had already served another request. All {replay.meta.num_requests} requests fit in blocks 1 to {replay.max_block}.
        </p>
        <p>
          If the pool runs out, the scheduler preempts the newest request: its blocks are freed, it goes back to the front of the queue, and its prompt plus what it has written so far is recomputed
          when there is room. A test squeezes 16 requests into 56 blocks, forcing 6 preemptions, and every output matches a run with room to spare.
        </p>
      </>
    ),
  },
  {
    id: "replay",
    title: "the recorded run",
    body: (
      <>
        <p>
          On the right is the whole recorded run, replayed through the same allocation rules at half speed: {replay.meta.num_requests} chat requests arriving at {replay.meta.request_rate} per second,
          up to {replay.max_running} at once. Each block shows the request that holds it and how full it is. Drag the slider to stop at any step.
        </p>
        <p>
          At its busiest the run held <span className="font-mono text-accent">{replay.peak.blocks}</span> of the {fmt(replay.meta.kv_blocks)} blocks. The pool is simply 90% of the memory left after
          the weights, so a run like this leaves almost all of it free, room for many more or much longer requests. Serving them side by side from one pool is what batching needs: batch 16 serves{" "}
          <span className="font-mono text-accent">{`${fmt(tput("16"))} tokens/s`}</span> against {fmt(tput("1"))} at batch 1 (CUDA Graphs on, both).
        </p>
      </>
    ),
  },
];

const captions = [
  `the four requests' real lengths from ${replay.file}; blocks are ${B} tokens`,
  `block table of request ${first.id} and the pool at that step, both from the build-time replay of ${replay.file}`,
  "TinyLlama: 32 query heads and 4 KV heads of 64 dims; time from results/decode_step.md (GPU time per op)",
  `request ${first.id}'s real prompt and blocks; the slots filled over time are drawn, not timed`,
  "block reuse from the replay; the preemption numbers from ForcedPreemptionKeepsTheOutputsOfEveryRequest in tests/gpu/test_serve.cpp",
  `${replay.file}, recorded ${replay.meta.date.slice(0, 10)} on the ${replay.meta.gpu.replace("NVIDIA GeForce ", "")}; every step's free block count checked against the trace at build time`,
];

export function PagedKvStory() {
  return (
    <StickyStory
      steps={steps}
      caption={(i) => captions[i]}
      figure={(i) => {
        switch (i) {
          case 0:
            return <WhyBlocksFigure />;
          case 1:
            return <TableFigure />;
          case 2:
            return <WriteKvFigure launchUs={writeUs} share={writeShare} />;
          case 3:
            return <AppendFigure />;
          case 4:
            return <ReuseFigure />;
          default:
            return <ReplayPlayer />;
        }
      }}
    />
  );
}
