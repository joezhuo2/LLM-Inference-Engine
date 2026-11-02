"use client";

import { useEffect, useRef, useState } from "react";
import { usePrefersReducedMotion } from "@/lib/reduced-motion";
import type { TraceStep } from "@/lib/trace";
import { type Feed, slotColor } from "./feed";
import { type Paint, useFeedCanvas } from "./use-feed-canvas";

const PITCH = 6;
const BAR = 4;
const TOP = 18;
const AXIS_MS = 20;
const CELL = 5;
const LABEL_EVERY = 25;

type Hover = { index: number; pinned: boolean };

function gutter(lastIndex: number) {
  return Math.max(30, String(Math.max(0, lastIndex)).length * 6 + 10);
}

function stripWidth(maxBatch: number) {
  return maxBatch * CELL;
}

function piece(text: string) {
  return text === "" ? "∅" : text.replaceAll("\n", "⏎");
}

export function Timeline({ feed, maxBatch }: { feed: Feed; maxBatch: number }) {
  const reduced = usePrefersReducedMotion();
  const [hover, setHover] = useState<Hover | null>(null);
  const shown = useRef(0);
  const scroll = useRef(0);
  const tipRef = useRef<HTMLDivElement>(null);
  const hoverRef = useRef(hover);

  const target = () => feed.model.lastIndex + 1 + (feed.pending ? Math.min(1, feed.progress) : 0);

  const rowTop = (index: number, height: number) => height - (shown.current - scroll.current - index) * PITCH;

  const { canvasRef, kick } = useFeedCanvas(feed, ({ ctx, width, height, colors, dt, label }: Paint) => {
    const model = feed.model;
    const goal = target();
    const jump = reduced || feed.playing || Math.abs(goal - shown.current) > 60;
    shown.current = jump ? goal : shown.current + (goal - shown.current) * (1 - Math.exp(-dt / 0.14));
    if (Math.abs(goal - shown.current) < 0.01) shown.current = goal;
    const strip = stripWidth(maxBatch);
    const left = gutter(model.lastIndex);
    const barX = left;
    const barW = Math.max(40, width - left - strip - 14);
    const stripX = width - strip;
    const pxPerMs = barW / AXIS_MS;
    const focus = feed.focus();

    ctx.fillStyle = colors.border;
    for (let ms = 0; ms <= AXIS_MS; ms += 5) {
      const x = barX + ms * pxPerMs;
      label(ms === 0 ? "0 ms" : String(ms), x, 7, colors.muted, ms === 0 ? "left" : ms === AXIS_MS ? "right" : "center");
      ctx.fillRect(Math.round(x), TOP - 4, 1, height - TOP + 4);
    }
    label("batch", stripX, 7, colors.muted);

    ctx.save();
    ctx.beginPath();
    ctx.rect(0, TOP, width, height - TOP);
    ctx.clip();
    const first = Math.max(model.firstIndex, Math.floor(shown.current - scroll.current - (height - TOP) / PITCH) - 1);
    const last = Math.ceil(shown.current - scroll.current);
    const hovered = hoverRef.current?.index;
    const paths = new Map<string, Path2D>();
    const path = (color: string, dim: boolean) => {
      const key = `${dim ? 1 : 0}${color}`;
      let p = paths.get(key);
      if (!p) paths.set(key, (p = new Path2D()));
      return p;
    };
    const outlines = new Path2D();
    for (let k = first; k <= last; k++) {
      const y = rowTop(k, height);
      const done = model.step(k);
      const step = done ?? (k === model.lastIndex + 1 ? feed.pending ?? undefined : undefined);
      if (!step) {
        if (k <= model.lastIndex) path(colors.warn, true).rect(barX, y + BAR / 2, barW, 1);
        continue;
      }
      if (k % LABEL_EVERY === 0 && y > TOP + 2) label(String(k), left - 6, y + BAR / 2, colors.muted, "right");
      const ms = (step.end_s - step.start_s) * 1e3 * (done ? 1 : Math.min(1, feed.progress));
      const dim = focus !== null && !step.seqs.includes(focus);
      path(step.kind === "prefill" ? colors.prefill : colors.decode, dim).roundRect(barX, y, Math.max(1, Math.min(ms, AXIS_MS) * pxPerMs), BAR, [0, 2, 2, 0]);
      if (ms > AXIS_MS) path(colors.text, false).rect(barX + barW + 3, y, 3, BAR);
      const slots = done ? model.slotsOf(k) : step.seqs.map((id) => model.seqs.get(id)?.slot ?? -1);
      const n = Math.min(maxBatch, step.seqs.length);
      for (let i = 0; i < n; i++) {
        const slot = slots?.[i] ?? -1;
        path(slot >= 0 ? colors.slots[slot] : colors.baseline, dim).rect(stripX + i * CELL, y, CELL - 1, BAR);
        if (step.seqs[i] === focus) outlines.rect(stripX + i * CELL - 0.5, y - 1.5, CELL, BAR + 3);
      }
      if (k === hovered) outlines.rect(1.5, y - 1.5, width - 3, BAR + 3);
    }
    for (const [key, p] of paths) {
      ctx.globalAlpha = key[0] === "1" ? 0.3 : 1;
      ctx.fillStyle = key.slice(1);
      ctx.fill(p);
    }
    ctx.globalAlpha = 1;
    ctx.strokeStyle = colors.text;
    ctx.lineWidth = 1;
    ctx.stroke(outlines);
    for (const [k, missed] of model.gaps) {
      if (k < first || k > last + missed) continue;
      label(`${missed} missed`, barX + 4, rowTop(k - Math.ceil(missed / 2), height) + BAR / 2, colors.warn);
    }
    ctx.restore();

    const tip = tipRef.current;
    if (tip && hovered !== undefined) {
      const y = rowTop(hovered, height);
      tip.style.visibility = y < TOP - PITCH || y > height ? "hidden" : "visible";
      tip.style.transform = y > height / 2 ? `translateY(calc(${Math.round(y - 6)}px - 100%))` : `translateY(${Math.round(y + BAR + 6)}px)`;
    }
    return shown.current !== goal;
  }, 10);

  useEffect(() => {
    hoverRef.current = hover;
    kick();
  });

  const indexAt = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const y = e.clientY - rect.top;
    if (y < TOP) return null;
    const k = Math.floor(shown.current - scroll.current - (rect.height - y) / PITCH);
    return feed.model.step(k) ? k : null;
  };

  const point = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const k = indexAt(e);
    setHover(k === null ? null : { index: k, pinned: false });
  };

  const move = (delta: number | "first" | "last") => {
    const model = feed.model;
    if (model.lastIndex < 0) return;
    const from = hover?.index ?? model.lastIndex;
    const k = delta === "first" ? model.firstIndex : delta === "last" ? model.lastIndex : Math.max(model.firstIndex, Math.min(model.lastIndex, from + delta));
    const canvas = canvasRef.current;
    if (canvas && !feed.playing) {
      const rows = (canvas.clientHeight - TOP) / PITCH - 1;
      const bottom = shown.current - scroll.current;
      if (k >= bottom) scroll.current = Math.max(0, shown.current - k - 1);
      else if (k < bottom - rows) scroll.current = Math.max(0, shown.current - k - rows);
    }
    setHover({ index: k, pinned: true });
    kick();
  };

  const onKey = (e: React.KeyboardEvent) => {
    const keys: Record<string, number | "first" | "last"> = { ArrowUp: -1, ArrowDown: 1, PageUp: -20, PageDown: 20, Home: "first", End: "last" };
    if (e.key === "Escape") {
      setHover(null);
      scroll.current = 0;
      kick();
      return;
    }
    if (!(e.key in keys)) return;
    e.preventDefault();
    move(keys[e.key]);
  };

  const onWheel = (e: React.WheelEvent) => {
    if (feed.playing) return;
    const max = Math.max(0, shown.current - feed.model.firstIndex - 4);
    scroll.current = Math.max(0, Math.min(max, scroll.current - e.deltaY / PITCH / 2));
    kick();
  };

  const step = hover ? feed.model.step(hover.index) : undefined;

  return (
    <div className="relative h-full">
      <canvas
        ref={canvasRef}
        tabIndex={0}
        role="img"
        aria-label="Batch timeline: one row per scheduler step, newest at the bottom, bar length is the step time, prefill and decode colored, and a square per sequence in the batch. Arrow keys read a step."
        aria-describedby="timeline-tip"
        onKeyDown={onKey}
        onWheel={onWheel}
        onPointerMove={(e) => e.pointerType === "mouse" && point(e)}
        onPointerDown={point}
        onPointerLeave={(e) => e.pointerType === "mouse" && !hover?.pinned && setHover(null)}
        onBlur={() => setHover(null)}
        className="block h-full w-full touch-pan-y"
      />
      <div
        ref={tipRef}
        id="timeline-tip"
        role="status"
        className="pointer-events-none absolute left-2 right-2 top-0 z-10 rounded border border-border bg-surface p-2 shadow-sm"
        style={{ visibility: step ? "visible" : "hidden" }}
      >
        {step && <StepDetails step={step} feed={feed} />}
      </div>
    </div>
  );
}

function StepDetails({ step, feed }: { step: TraceStep; feed: Feed }) {
  const ms = (step.end_s - step.start_s) * 1e3;
  const total = feed.model.totalBlocks;
  return (
    <div className="space-y-1.5">
      <div className="text-text">
        {`step ${step.index} · ${step.kind} · ${ms.toFixed(1)} ms`}
        <span className="text-muted">{` · at ${step.start_s.toFixed(3)} s`}</span>
      </div>
      <div className="flex flex-wrap gap-1">
        {step.seqs.map((id, i) => {
          const slot = feed.model.slotsOf(step.index)?.[i] ?? -1;
          const done = step.finished.includes(id);
          return (
            <span key={id} className="flex items-center gap-1 rounded border border-border px-1 py-px">
              <span aria-hidden className="size-2 rounded-sm" style={{ background: slotColor(slot) }} />
              <span className="text-muted">{`#${id}`}</span>
              <span className="whitespace-pre text-text">{piece(step.tokens[i]?.text ?? "")}</span>
              {done && <span className="text-muted">done</span>}
            </span>
          );
        })}
      </div>
      <div className="text-muted">{`after it: ${step.running} running, ${step.waiting} waiting, ${(total - 1 - step.free_blocks).toLocaleString("en-US")} blocks held`}</div>
    </div>
  );
}
