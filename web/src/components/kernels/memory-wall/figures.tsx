"use client";

import { motion, type MotionProps } from "motion/react";
import { Frame, Label } from "../frame";
import { useLoop } from "../loop";

type Box = { x: number; y: number; w: number; h: number };

const cellFill = "fill-surface stroke-border";

function useZoom(duration: number) {
  const { reduced } = useLoop(duration);
  return (start: number, end: number, from: Box, to: Box): MotionProps => {
    if (reduced) return { initial: false, animate: { opacity: 1 } };
    const sx = from.w / to.w;
    const sy = from.h / to.h;
    const small = `translate(${from.x - to.x * sx}px, ${from.y - to.y * sy}px) scale(${sx}, ${sy})`;
    const full = "translate(0px, 0px) scale(1, 1)";
    const loop = { duration, repeat: Infinity };
    return {
      style: { transformOrigin: "0px 0px" },
      initial: { opacity: 0, transform: small },
      animate: { opacity: [0, 0, 1, 1, 0], transform: [small, small, full, full, full] },
      transition: {
        opacity: { ...loop, ease: "linear", times: [0, start, start + 0.03, 0.95, 1] },
        transform: { ...loop, ease: ["linear", "easeInOut", "linear", "linear"], times: [0, start, end, 0.95, 1] },
      },
    };
  };
}

const gridCols = 11;
const gridCell = { w: 22, h: 22, pitch: 26 };
const gridBlock = 5;
const warps = 8;
const warpCell = { w: 30, h: 30, pitch: 34 };
const warpPick = 2;
const threadCell = { w: 14, h: 14, pitch: 17 };

const gridBox = (i: number): Box => ({ x: 20 + (i % gridCols) * gridCell.pitch, y: 50 + Math.floor(i / gridCols) * gridCell.pitch, w: gridCell.w, h: gridCell.h });
const blockBox: Box = { x: 20, y: 150, w: warps * warpCell.pitch - 4, h: warpCell.h };
const warpBox = (i: number): Box => ({ x: 20 + i * warpCell.pitch, y: 150, w: warpCell.w, h: warpCell.h });
const threadsBox: Box = { x: 20, y: 236, w: 16 * threadCell.pitch - 3, h: 2 * threadCell.pitch - 3 };

export function HierarchyFigure({ sms, blocks, threadsPerBlock }: { sms: number; blocks: number; threadsPerBlock: number }) {
  const { on } = useLoop(14);
  const zoom = useZoom(14);
  const picked = gridBox(gridBlock);
  return (
    <Frame label="A kernel launch is a grid of blocks; one block is split into warps of 32 threads, and the 32 threads of a warp run the same instruction together">
      <Label x={20} y={22}>{"one kernel launch, drilled down: grid, block, warp, thread"}</Label>
      {Array.from({ length: blocks }, (_, i) => {
        const b = gridBox(i);
        return <rect key={i} x={b.x} y={b.y} width={b.w} height={b.h} rx={2} className={i === gridBlock ? "fill-accent/30 stroke-accent" : cellFill} strokeWidth={0.75} />;
      })}
      <Label x={20} y={118} className="fill-text">{`grid: ${blocks} blocks, ${threadsPerBlock} threads each`}</Label>
      <motion.g {...zoom(0.2, 0.34, picked, blockBox)}>
        {Array.from({ length: warps }, (_, i) => {
          const w = warpBox(i);
          return <rect key={i} x={w.x} y={w.y} width={w.w} height={w.h} rx={3} className={i === warpPick ? "fill-accent/30 stroke-accent" : cellFill} strokeWidth={0.75} />;
        })}
      </motion.g>
      <motion.g {...on(0.34, 0.95)}>
        <Label x={20} y={200} className="fill-text">{`block: ${threadsPerBlock} threads = ${warps} warps, on one SM`}</Label>
      </motion.g>
      <motion.g {...zoom(0.46, 0.6, warpBox(warpPick), threadsBox)}>
        {Array.from({ length: 32 }, (_, i) => (
          <rect key={i} x={20 + (i % 16) * threadCell.pitch} y={236 + Math.floor(i / 16) * threadCell.pitch} width={threadCell.w} height={threadCell.h} rx={2} className={cellFill} strokeWidth={0.75} />
        ))}
      </motion.g>
      <motion.g {...on(0.6, 0.95)}>
        <Label x={20} y={290} className="fill-text">warp: 32 threads, one instruction at a time</Label>
      </motion.g>
      <motion.g {...on(0.68, 0.9, true)}>
        {Array.from({ length: 32 }, (_, i) => (
          <rect key={i} x={20 + (i % 16) * threadCell.pitch} y={236 + Math.floor(i / 16) * threadCell.pitch} width={threadCell.w} height={threadCell.h} rx={2} className="fill-accent" />
        ))}
      </motion.g>
      <Label x={20} y={332}>{`this GPU has ${sms} SMs; a block runs on one of them and stays there`}</Label>
      <Label x={20} y={348}>the 32 threads of a warp run in lockstep: one instruction, 32 values</Label>
      <Label x={20} y={364}>{"a thread finds its value from blockIdx × blockDim + threadIdx"}</Label>
    </Frame>
  );
}

export type Tier = { name: string; size: string; note: string; detail: string; box: Box };

export function MemoryFigure({ tiers }: { tiers: Tier[] }) {
  const { move, on } = useLoop(14);
  const centre = 170;
  const hops = [
    { from: 0, to: 1, start: 0.05, end: 0.2 },
    { from: 1, to: 2, start: 0.22, end: 0.37 },
    { from: 2, to: 3, start: 0.39, end: 0.54 },
  ];
  const mid = (t: Tier) => t.box.y + t.box.h / 2 - 6;
  const left = (t: Tier) => t.box.x + 10;
  const right = (t: Tier) => t.box.x + t.box.w - 26;
  return (
    <Frame label="The GPU memory hierarchy: global memory in DRAM is the largest and slowest, then the L2 cache, then shared memory in each SM, then registers private to each thread; data is loaded down the tiers and the result written back up">
      <Label x={20} y={22}>{"a tile of data goes down the tiers, the result comes back up"}</Label>
      {tiers.map((t, i) => (
        <g key={t.name}>
          <rect x={t.box.x} y={t.box.y} width={t.box.w} height={t.box.h} rx={4} className={i === 0 ? "fill-baseline/20 stroke-baseline" : i === 3 ? "fill-accent/20 stroke-accent" : "fill-surface stroke-border"} strokeWidth={0.75} />
          <Label x={centre} y={t.box.y + 19} anchor="middle" className="fill-text">{t.name}</Label>
          <Label x={centre} y={t.box.y + 35} anchor="middle" className={i === 0 ? "fill-baseline" : "fill-accent"}>{t.size}</Label>
          <Label x={340} y={t.box.y + 19} className="fill-text">{t.note}</Label>
          <Label x={340} y={t.box.y + 35}>{t.detail}</Label>
        </g>
      ))}
      {hops.map((h) => (
        <motion.rect key={h.from} x={left(tiers[h.from])} y={mid(tiers[h.from])} width={16} height={12} rx={2} className="fill-accent" {...move(h.start, h.end, left(tiers[h.to]) - left(tiers[h.from]), mid(tiers[h.to]) - mid(tiers[h.from]), false)} />
      ))}
      <motion.circle cx={right(tiers[3]) + 8} cy={mid(tiers[3]) + 6} r={5} className="fill-warn" {...on(0.56, 0.68, false)} />
      <motion.rect x={right(tiers[3])} y={mid(tiers[3])} width={16} height={12} rx={2} className="fill-text" {...move(0.7, 0.9, right(tiers[0]) - right(tiers[3]), mid(tiers[0]) - mid(tiers[3]), true)} />
      <Label x={20} y={372}>closer to the thread: smaller and faster</Label>
      <Label x={20} y={388}>only the DRAM bandwidth is measured on this GPU</Label>
    </Frame>
  );
}

export type Lane = { title: string; rows: number; flopPerByte: number; memUs: number; computeUs: number; note: string };

export function ContrastFigure({ lanes, gpuLine }: { lanes: Lane[]; gpuLine: string[] }) {
  const { grow, on } = useLoop(14);
  const x = 150;
  const full = 290;
  const maxUs = Math.max(...lanes.flatMap((l) => [l.memUs, l.computeUs]));
  const rate = 0.5 / maxUs;
  const width = (us: number) => Math.max(3, (full * us) / maxUs);
  return (
    <Frame label="Time to read the weights from memory against time to do the arithmetic: for one row the memory read takes about a hundred times longer than the arithmetic, for 512 rows the arithmetic takes five times longer than the read; the longer bar sets the time of the operation">
      <Label x={20} y={22}>{"one layer's gate/up weights: bytes ÷ bandwidth, FLOPs ÷ compute"}</Label>
      {lanes.map((l, li) => {
        const top = 52 + li * 130;
        const memLonger = l.memUs >= l.computeUs;
        const bars = [
          { label: "memory", us: l.memUs, gates: memLonger },
          { label: "compute", us: l.computeUs, gates: !memLonger },
        ];
        return (
          <g key={l.title}>
            <Label x={20} y={top} className="fill-text">{l.title}</Label>
            <Label x={20} y={top + 16}>{`${l.rows} row${l.rows > 1 ? "s" : ""}, ${l.flopPerByte} FLOP per byte`}</Label>
            {bars.map((b, bi) => {
              const y = top + 30 + bi * 26;
              const start = 0.04 + li * 0.04;
              return (
                <g key={b.label}>
                  <Label x={20} y={y + 10} className={b.gates ? "fill-accent" : "fill-muted"}>{b.label}</Label>
                  <rect x={x} y={y} width={full} height={12} rx={6} className="fill-border" />
                  <motion.rect x={x} y={y} width={width(b.us)} height={12} rx={6} className={b.gates ? "fill-accent" : "fill-baseline"} {...grow(start, start + b.us * rate)} />
                  <Label x={x + full + 8} y={y + 10} className={b.gates ? "fill-accent" : "fill-muted"}>{`${b.us.toFixed(b.us < 10 ? 1 : 0)} µs`}</Label>
                </g>
              );
            })}
            <motion.g {...on(0.04 + li * 0.04 + 0.5 + 0.02, 0.95)}>
              <Label x={20} y={top + 100} className="fill-text">{l.note}</Label>
            </motion.g>
          </g>
        );
      })}
      {gpuLine.map((t, i) => (
        <Label key={t} x={20} y={332 + i * 16}>{t}</Label>
      ))}
    </Frame>
  );
}

export type RoofPoint = { x: number; y: number; label: string };

const W = 560;
const pad = { l: 46, r: 20, t: 50, b: 90 };
const H = 400;

export function RooflineFigure({ gbs, tflops, rowsPath, measured }: { gbs: number; tflops: number; rowsPath: { rows: number; flopPerByte: number }[]; measured: RoofPoint[] }) {
  const { on, reduced } = useLoop(16);
  const xr = [0.5, 1000];
  const yr = [0.05, 100];
  const lx = (v: number) => pad.l + ((Math.log10(v) - Math.log10(xr[0])) / (Math.log10(xr[1]) - Math.log10(xr[0]))) * (W - pad.l - pad.r);
  const ly = (v: number) => H - pad.b - ((Math.log10(v) - Math.log10(yr[0])) / (Math.log10(yr[1]) - Math.log10(yr[0]))) * (H - pad.t - pad.b);
  const ridge = (tflops * 1e12) / (gbs * 1e9);
  const roof = (f: number) => Math.min((f * gbs) / 1000, tflops);
  const path = rowsPath.map((p) => ({ x: lx(p.flopPerByte), y: ly(roof(p.flopPerByte)) }));
  const times = rowsPath.map((_, i) => 0.08 + (0.8 * i) / (rowsPath.length - 1));
  const crossAt = 0.08 + (0.8 * 2) / (rowsPath.length - 1) - 0.02;
  return (
    <Frame label="Roofline of this GPU: performance in TFLOP/s against FLOPs per byte of memory traffic. Left of the ridge at 107 FLOP/byte performance is capped by memory bandwidth, right of it by the compute peak. A ring slides up the slope as the batch grows and flattens at the ridge; measured points at 1 and 16 rows sit on the slope">
      <Label x={20} y={22}>{"more rows per weight read: the point moves right"}</Label>
      {[0.1, 1, 10, 100].map((v) => (
        <g key={v}>
          <line x1={pad.l} x2={W - pad.r} y1={ly(v)} y2={ly(v)} className="stroke-border" strokeWidth={0.5} />
          <Label x={pad.l - 6} y={ly(v) + 3} anchor="end">{String(v)}</Label>
        </g>
      ))}
      {[1, 10, 100, 1000].map((v) => (
        <Label key={v} x={lx(v)} y={H - pad.b + 16} anchor="middle">{String(v)}</Label>
      ))}
      <Label x={W - pad.r} y={H - pad.b + 34} anchor="end">FLOP per byte of memory traffic ≈ rows per weight read</Label>
      <Label x={pad.l} y={pad.t - 8}>TFLOP/s</Label>
      <line x1={lx(ridge)} x2={lx(ridge)} y1={pad.t} y2={H - pad.b} className="stroke-border" strokeDasharray="3 3" />
      <path d={`M ${lx(xr[0])} ${ly(roof(xr[0]))} L ${lx(ridge)} ${ly(tflops)} L ${lx(xr[1])} ${ly(tflops)}`} className="fill-none stroke-text" strokeWidth={1.25} />
      <Label x={lx(1.4)} y={ly(roof(1.4)) - 26} className="fill-text">{`${gbs} GB/s`}</Label>
      <Label x={W - pad.r} y={ly(tflops) + 14} anchor="end" className="fill-text">{`${tflops} TFLOP/s`}</Label>
      <Label x={lx(ridge)} y={H - pad.b + 52} anchor="middle" className="fill-text">{`ridge ${Math.round(ridge)}`}</Label>
      <motion.g {...on(0.02, crossAt, true)}>
        <Label x={pad.l + 8} y={H - pad.b + 72} className="fill-accent">memory bound: time = bytes ÷ bandwidth</Label>
      </motion.g>
      <motion.g {...on(crossAt, 0.95, false)}>
        <Label x={W - pad.r} y={H - pad.b + 72} anchor="end" className="fill-accent">compute bound: time = FLOPs ÷ peak</Label>
      </motion.g>
      {measured.map((m) => (
        <g key={m.label}>
          <circle cx={lx(m.x)} cy={ly(m.y)} r={4} className="fill-accent" opacity={0.85} />
          <Label x={lx(m.x) + 8} y={ly(m.y) + 14} className="fill-accent">{m.label}</Label>
        </g>
      ))}
      {reduced ? (
        <circle cx={path[2].x} cy={path[2].y} r={7} className="fill-none stroke-text" strokeWidth={1.5} />
      ) : (
        <motion.circle
          cx={0}
          cy={0}
          r={7}
          className="fill-none stroke-text"
          strokeWidth={1.5}
          initial={{ x: path[0].x, y: path[0].y }}
          animate={{ x: [path[0].x, path[0].x, ...path.map((p) => p.x), path[path.length - 1].x], y: [path[0].y, path[0].y, ...path.map((p) => p.y), path[path.length - 1].y] }}
          transition={{ duration: 16, repeat: Infinity, ease: "easeInOut", times: [0, 0.02, ...times, 1] }}
        />
      )}
      {rowsPath.map((p, i) => (
        <Label key={p.rows} x={lx(p.flopPerByte) + (i < 2 ? -10 : 0)} y={ly(roof(p.flopPerByte)) - (i < 2 ? 10 : 14)} anchor={i < 2 ? "end" : "middle"}>{`${p.rows} row${p.rows > 1 ? "s" : ""}`}</Label>
      ))}
    </Frame>
  );
}

export type FusionNumbers = { tokens: number; unfusedKb: number; fusedKb: number };

const trip = 40;

export function FusionFigure({ numbers }: { numbers: FusionNumbers }) {
  const { move, on, pulse } = useLoop(16);
  const pace = 0.14;
  const begin = 0.04;
  const at = (k: number) => [begin + k * pace, begin + k * pace + pace * 0.8] as const;
  const dram = (y: number, label: string) => (
    <g>
      <rect x={20} y={y} width={520} height={22} rx={3} className="fill-baseline/20 stroke-baseline" strokeWidth={0.75} />
      <Label x={30} y={y + 15} className="fill-text">{label}</Label>
    </g>
  );
  const kernel = (x: number, y: number, w: number, label: string, labelY = 24) => (
    <g>
      <rect x={x} y={y} width={w} height={40} rx={3} className="fill-surface stroke-accent" strokeWidth={0.75} />
      <Label x={x + w / 2} y={y + labelY} anchor="middle" className="fill-text">{label}</Label>
    </g>
  );
  const packet = (k: number, x: number, y: number, up: boolean, name: string, cls = "fill-accent") => {
    const [s, e] = at(k);
    return (
      <g key={`${name}-${k}`}>
        <motion.rect x={x} y={y} width={16} height={10} rx={2} className={cls} {...move(s, e, 0, up ? -trip : trip, false)} />
        <Label x={x + 20} y={y + (up ? -trip / 2 : trip / 2) + 8}>{name}</Label>
      </g>
    );
  };
  const ticks = (count: number, y: number) =>
    Array.from({ length: count }, (_, k) => <motion.rect key={k} x={20 + k * 20} y={y} width={14} height={8} rx={2} className="fill-accent" {...on(at(k)[1], 0.95, true)} />);
  const topDram = 120;
  const topKernel = 40;
  const lowDram = 292;
  const lowKernel = 212;
  return (
    <Frame label="Unfused, an add kernel and an RMSNorm kernel make six transfers to and from global memory, because the sum is written out and read back twice; fused, one kernel makes four transfers and keeps the sum in registers between the two steps">
      <Label x={20} y={22} className="fill-text">unfused: two kernels, the sum goes out to DRAM and back</Label>
      {kernel(150, topKernel, 100, "add")}
      {kernel(330, topKernel, 120, "RMSNorm")}
      {dram(topDram, "global memory")}
      {packet(0, 160, topDram - 4, true, "x")}
      {packet(1, 200, topDram - 4, true, "r")}
      {packet(2, 232, topKernel + 44, false, "sum")}
      {packet(3, 344, topDram - 4, true, "sum")}
      {packet(4, 384, topDram - 4, true, "sum")}
      {packet(5, 424, topKernel + 44, false, "norm")}
      {ticks(6, topDram + 30)}
      <Label x={150} y={topDram + 38}>6 transfers</Label>
      <Label x={20} y={196} className="fill-text">fused: one kernel, the sum never leaves the chip</Label>
      {kernel(175, lowKernel, 190, "add + RMSNorm", 16)}
      <motion.g {...pulse(0)}>
        <Label x={270} y={lowKernel + 32} anchor="middle" className="fill-accent">sum stays in registers</Label>
      </motion.g>
      {dram(lowDram, "global memory")}
      {packet(0, 200, lowDram - 4, true, "x")}
      {packet(1, 240, lowDram - 4, true, "r")}
      {packet(2, 280, lowKernel + 44, false, "sum")}
      {packet(3, 320, lowKernel + 44, false, "norm")}
      {ticks(4, lowDram + 30)}
      <Label x={110} y={lowDram + 38}>4 transfers</Label>
      <Label x={20} y={352}>{`${numbers.tokens} tokens, Nsight Compute: ${numbers.unfusedKb} KB of DRAM traffic became ${numbers.fusedKb} KB`}</Label>
      <Label x={20} y={368}>the naive pair reads the sum twice, once for the squares, once to scale</Label>
    </Frame>
  );
}
