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
