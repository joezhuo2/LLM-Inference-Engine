"use client";

import { useEffect, useRef } from "react";
import { useCanvasLoop } from "@/lib/canvas";
import { useIntroDone } from "@/lib/intro";
import { type Trace, traceDuration, useTrace } from "@/lib/trace";

type Drop = { col: number; spawn: number; chars: string; first: boolean };

const cell = 16;
const speed = 420;

function hash(a: number, b: number) {
  let h = (a * 2654435761) ^ (b * 1597334677);
  h = Math.imul(h ^ (h >>> 15), 2246822519);
  return (h ^ (h >>> 13)) >>> 0;
}

const overlapS = 0.4;
const span = 4;
const holdMs = 350;
const fadeMs = 300;
const rowSpan = 4096;
const glyphs = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789#$%&*";

type Cursor = { cycle: number; done: number; counts: Map<number, number> };

function createRain(trace: Trace, drops: { current: Drop[] }) {
  const period = Math.max(traceDuration(trace) - overlapS, 0.5);
  const cursors: Cursor[] = [];
  let nextCycle = 0;
  return (t: number) => {
    while (nextCycle * period <= t) cursors.push({ cycle: nextCycle++, done: 0, counts: new Map() });
    for (const cur of cursors) {
      const base = cur.cycle * period;
      while (cur.done < trace.steps.length && base + trace.steps[cur.done].end_s <= t) {
        const step = trace.steps[cur.done++];
        step.seqs.forEach((seq, j) => {
          const n = cur.counts.get(seq) ?? 0;
          cur.counts.set(seq, n + 1);
          const chars = step.tokens[j].text.replace(/\s+/g, "") || "·";
          drops.current.push({
            col: hash(seq + cur.cycle * 7919, n),
            spawn: base + step.end_s,
            chars,
            first: step.kind === "prefill",
          });
        });
      }
    }
    while (cursors.length && cursors[0].done >= trace.steps.length) cursors.shift();
  };
}

function RainCanvas({ trace }: { trace: Trace }) {
  const duration = traceDuration(trace);
  const drops = useRef<Drop[]>([]);
  const advance = useRef<((t: number) => void) | null>(null);
  const pointer = useRef<{ x: number; y: number } | null>(null);
  const lit = useRef(new Map<number, number>());
  const { canvasRef, playing, play } = useCanvasLoop(({ ctx, width, height, elapsedS, palette }) => {
    const t = elapsedS;
    advance.current ??= createRain(trace, drops);
    advance.current(t);
    ctx.clearRect(0, 0, width, height);
    const ncols = Math.max(1, Math.floor(width / cell));
    ctx.font = `14px ${palette.mono}`;
    ctx.textBaseline = "top";
    const life = (height + 12 * cell) / speed;
    const now = performance.now();
    const p = pointer.current;
    if (p) {
      const pc = Math.round(p.x / cell) - span / 2;
      const pr = Math.round(p.y / cell) - span / 2;
      for (let dx = 0; dx < span; dx++) {
        for (let dy = 0; dy < span; dy++) {
          if (pc + dx >= 0 && pr + dy >= 0) lit.current.set((pc + dx) * rowSpan + pr + dy, now);
        }
      }
    }
    const glow = (key: number) => {
      const at = lit.current.get(key);
      if (at === undefined) return 0;
      const age = now - at;
      return age < holdMs ? 1 : Math.max(0, 1 - (age - holdMs) / fadeMs);
    };
    const white = parseInt(palette.bg.slice(1, 3), 16) < 128 ? "#ffffff" : palette.text;
    const covered = new Set<number>();
    drops.current = drops.current.filter((d) => t - d.spawn < life && t >= d.spawn);
    for (const d of drops.current) {
      const col = d.col % ncols;
      const x = col * cell;
      const head = (t - d.spawn) * speed;
      for (let i = d.chars.length - 1, k = 0; i >= 0; i--, k++) {
        const y = head - k * cell;
        if (y < -cell || y > height) continue;
        ctx.globalAlpha = Math.max(0, 1 - k * 0.14);
        ctx.fillStyle = k === 0 ? (d.first ? palette.warn : palette.text) : palette.accent;
        ctx.fillText(d.chars[i], x, y);
        const key = col * rowSpan + Math.round(y / cell);
        const g = glow(key);
        if (g === 0) continue;
        covered.add(key);
        ctx.globalAlpha = g;
        ctx.fillStyle = white;
        ctx.fillText(d.chars[i], x, y);
      }
    }
    for (const key of lit.current.keys()) {
      const g = glow(key);
      if (g === 0) {
        lit.current.delete(key);
        continue;
      }
      if (covered.has(key)) continue;
      const col = Math.floor(key / rowSpan);
      const row = key % rowSpan;
      ctx.globalAlpha = g * 0.4;
      ctx.fillStyle = palette.accent;
      ctx.fillText(glyphs[hash(col, row) % glyphs.length], col * cell, row * cell);
    }
    ctx.globalAlpha = 1;
  }, duration * 0.6);

  useEffect(() => {
    const canvas = canvasRef.current;
    const area = canvas?.parentElement;
    if (!canvas || !area) return;
    const move = (e: PointerEvent) => {
      const r = canvas.getBoundingClientRect();
      pointer.current = { x: e.clientX - r.left, y: e.clientY - r.top };
    };
    const leave = () => (pointer.current = null);
    area.addEventListener("pointermove", move);
    area.addEventListener("pointerleave", leave);
    return () => {
      area.removeEventListener("pointermove", move);
      area.removeEventListener("pointerleave", leave);
    };
  }, [canvasRef]);
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
  const introDone = useIntroDone();
  return trace && introDone ? <RainCanvas trace={trace} /> : null;
}
