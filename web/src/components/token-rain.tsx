"use client";

import { useRef } from "react";
import { useCanvasLoop } from "@/lib/canvas";
import { createReplayer, loopTime, type Trace, traceDuration, useTrace } from "@/lib/trace";

type Drop = { col: number; spawn: number; chars: string; first: boolean };

const cell = 16;
const speed = 420;

function hash(a: number, b: number) {
  let h = (a * 2654435761) ^ (b * 1597334677);
  h = Math.imul(h ^ (h >>> 15), 2246822519);
  return (h ^ (h >>> 13)) >>> 0;
}

function createRain(trace: Trace, drops: { current: Drop[] }) {
  const counts = new Map<number, number>();
  return createReplayer(
    trace,
    (step) => {
      step.seqs.forEach((seq, j) => {
        const n = counts.get(seq) ?? 0;
        counts.set(seq, n + 1);
        const chars = step.tokens[j].text.replace(/\s+/g, "") || "·";
        drops.current.push({ col: hash(seq, n), spawn: step.end_s, chars, first: step.kind === "prefill" });
      });
    },
    () => {
      drops.current = [];
      counts.clear();
    },
  );
}

function RainCanvas({ trace }: { trace: Trace }) {
  const duration = traceDuration(trace);
  const drops = useRef<Drop[]>([]);
  const advance = useRef<((t: number) => void) | null>(null);
  const { canvasRef, playing, play } = useCanvasLoop(({ ctx, width, height, elapsedS, palette }) => {
    const t = loopTime(elapsedS, duration);
    advance.current ??= createRain(trace, drops);
    advance.current(t);
    ctx.clearRect(0, 0, width, height);
    const ncols = Math.max(1, Math.floor(width / cell));
    ctx.font = `14px ${palette.mono}`;
    ctx.textBaseline = "top";
    const life = (height + 12 * cell) / speed;
    drops.current = drops.current.filter((d) => t - d.spawn < life && t >= d.spawn);
    for (const d of drops.current) {
      const x = (d.col % ncols) * cell;
      const head = (t - d.spawn) * speed;
      for (let i = d.chars.length - 1, k = 0; i >= 0; i--, k++) {
        const y = head - k * cell;
        if (y < -cell || y > height) continue;
        ctx.globalAlpha = Math.max(0, 1 - k * 0.14);
        ctx.fillStyle = k === 0 ? (d.first ? palette.warn : palette.text) : palette.accent;
        ctx.fillText(d.chars[i], x, y);
      }
    }
    ctx.globalAlpha = 1;
  }, duration * 0.6);
  return (
    <>
      <canvas ref={canvasRef} aria-hidden="true" className="absolute inset-0 h-full w-full" />
      {!playing && (
        <button
          type="button"
          onClick={play}
          className="pressable absolute right-4 top-4 z-10 rounded border border-border bg-surface px-3 py-1 font-mono text-xs text-muted"
        >
          play the recorded run
        </button>
      )}
    </>
  );
}

export function TokenRain() {
  const trace = useTrace();
  return trace ? <RainCanvas trace={trace} /> : null;
}
