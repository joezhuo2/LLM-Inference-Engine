"use client";

import { motion } from "motion/react";
import { Frame, Label } from "@/components/kernels/frame";
import { useLoop } from "@/components/kernels/loop";

const left = 20;
const width = 520;
const pitch = 13;
const ticks = Math.floor(width / pitch);
const at = (x: number) => 0.03 + ((x - left) / width) * 0.8;

const stall = { from: 10, to: 30 };
const promptStart = left + stall.from * pitch;
const chunkPitch = 52;
const chunks = 6;

function Lane({ y, label, children }: { y: number; label: string; children: React.ReactNode }) {
  return (
    <g>
      <Label x={left} y={y - 6}>{label}</Label>
      <rect x={left} y={y} width={width} height={26} rx={3} className="fill-none stroke-border" />
      {children}
    </g>
  );
}

export function ChunkedFigure() {
  const { on, grow } = useLoop(10);
  const tick = (i: number, y: number) => {
    const x = left + i * pitch;
    return <motion.rect key={i} x={x + 2} y={y + 5} width={pitch - 4} height={16} rx={2} className="fill-accent" {...on(at(x))} />;
  };
  const before = 34;
  const after = 188;
  return (
    <Frame label="Two timelines of the same running decoders and one long prompt. Before, the prompt is one block and the decoders stop until it ends. After, the prompt is cut into chunks that share each step with the decode rows, so the decoders never stop. Widths are illustrative">
      <Lane y={before + 12} label="schedule(): the prompt goes first">
        {Array.from({ length: ticks }, (_, i) => (i >= stall.from && i < stall.to ? null : tick(i, before + 12)))}
        <motion.rect x={promptStart} y={before + 12} width={(stall.to - stall.from) * pitch} height={26} rx={3} className="fill-none stroke-warn" strokeDasharray="4 3" {...on(at(promptStart), 0.94)} />
      </Lane>
      <Lane y={before + 62} label="">
        <motion.rect x={promptStart} y={before + 67} width={(stall.to - stall.from) * pitch - 2} height={16} rx={2} className="fill-series-2" {...grow(at(promptStart), at(promptStart + (stall.to - stall.from) * pitch))} />
      </Lane>
      <Label x={left} y={before + 108}>decoders wait for the whole prompt</Label>
      <Lane y={after + 12} label="schedule_chunked(): decode rows first, the rest of the budget to the prompt">
        {Array.from({ length: ticks }, (_, i) => tick(i, after + 12))}
      </Lane>
      <Lane y={after + 62} label="">
        {Array.from({ length: chunks }, (_, k) => {
          const x = promptStart + k * chunkPitch;
          return <motion.rect key={k} x={x} y={after + 67} width={40} height={16} rx={2} className="fill-series-2" {...grow(at(x), at(x + 40))} />;
        })}
      </Lane>
      <Label x={left} y={after + 108}>the prompt takes more steps, the decoders never stop</Label>
      <rect x={left} y={344} width={10} height={10} rx={2} className="fill-accent" />
      <Label x={left + 16} y={353}>decode rows</Label>
      <rect x={left + 110} y={344} width={10} height={10} rx={2} className="fill-series-2" />
      <Label x={left + 126} y={353}>prefill tokens</Label>
      <Label x={left} y={384}>an illustration of the mechanism, not a measurement</Label>
    </Frame>
  );
}
