"use client";

import clsx from "clsx";
import { useEffect, useRef, useState } from "react";
import { FREE, RESERVED, UNKNOWN, type DashModel } from "@/lib/dashboard/model";
import { usePrefersReducedMotion } from "@/lib/reduced-motion";
import { type Feed, slotColor } from "./feed";
import { type Paint, useFeedCanvas } from "./use-feed-canvas";

const GAP = 2;
const GUTTER = 28;
const BAR_H = 6;
const MAX_CELLS = 1024;
const fmt = (v: number) => v.toLocaleString("en-US");

export function poolWindow(highWater: number): number {
  let w = 128;
  while (w <= highWater && w < 1 << 20) w *= 2;
  return w;
}

type Geo = { blocks: number; per: number; cols: number; rows: number; size: number; gridH: number; width: number };

function geometry(window: number, width: number): Geo {
  const per = Math.max(1, Math.ceil(window / MAX_CELLS));
  const count = Math.ceil(window / per);
  const cols = count <= 256 ? 16 : 32;
  const rows = Math.ceil(count / cols);
  const size = Math.max(4, Math.floor((width - GUTTER) / cols));
  return { blocks: window, per, cols, rows, size, gridH: rows * size, width };
}

type Cell = { alpha: number; fill: number; slot: number; seq: number };

type Hover = { cell: number; x: number; y: number };

function target(model: DashModel, first: number, per: number) {
  let held = 0;
  let filled = 0;
  let seq = FREE;
  for (let b = first; b < Math.min(first + per, model.totalBlocks); b++) {
    const o = model.owner[b];
    if (o === FREE || o === RESERVED) continue;
    held++;
    if (seq === FREE) seq = o;
    if (o >= 0) {
      const s = model.seqs.get(o);
      const j = s ? s.blocks.indexOf(b) : -1;
      filled += s && j >= 0 ? model.filled(s, j) / model.blockSize : 1;
    } else {
      filled += 1;
    }
  }
  return { seq, alpha: held / per, fill: filled / per };
}

export function KvPool({ feed, minWindow }: { feed: Feed; minWindow: number }) {
  const reduced = usePrefersReducedMotion();
  const [hover, setHover] = useState<Hover | null>(null);
  const [, setVersion] = useState(0);
  const cells = useRef<Cell[]>([]);
  const windowRef = useRef(minWindow);
  const geoRef = useRef<Geo | null>(null);
  const [geo, setGeo] = useState<Geo | null>(null);
  const textCache = useRef("");
  const hoverRef = useRef(hover);
  const textRef = useRef<HTMLParagraphElement>(null);

  const { canvasRef, kick } = useFeedCanvas(feed, ({ ctx, width, colors, dt, label }: Paint) => {
    const model = feed.model;
    windowRef.current = Math.max(windowRef.current, minWindow, poolWindow(model.highWater));
    const g = geometry(windowRef.current, width);
    if (!geoRef.current || geoRef.current.size !== g.size || geoRef.current.blocks !== g.blocks || geoRef.current.width !== g.width) setGeo(g);
    geoRef.current = g;
    const count = Math.ceil(g.blocks / g.per);
    if (cells.current.length !== count) cells.current = Array.from({ length: count }, () => ({ alpha: 0, fill: 0, slot: -1, seq: FREE }));
    const k = reduced || dt === 0 ? 1 : 1 - Math.exp(-dt / 0.22);
    const focus = feed.focus();
    const hovered = hoverRef.current?.cell;
    let moving = false;

    for (let r = 0; r < g.rows; r++) {
      if (g.size >= 9 || r % 4 === 0) {
        label(fmt(r * g.cols * g.per), GUTTER - 5, r * g.size + g.size / 2, colors.muted, "right");
      }
    }

    for (let i = 0; i < count; i++) {
      const c = cells.current[i];
      const t = target(model, i * g.per, g.per);
      if (t.seq !== FREE) {
        if (t.seq !== c.seq && c.alpha > 0.02 && c.seq !== FREE) {
          t.alpha = 0;
          t.fill = 0;
        } else {
          c.seq = t.seq;
          c.slot = t.seq >= 0 ? (model.seqs.get(t.seq)?.slot ?? -1) : -1;
        }
      }
      c.alpha += (t.alpha - c.alpha) * k;
      c.fill += (t.fill - c.fill) * k;
      if (Math.abs(t.alpha - c.alpha) < 0.003) c.alpha = t.alpha;
      if (Math.abs(t.fill - c.fill) < 0.003) c.fill = t.fill;
      if (c.alpha !== t.alpha || c.fill !== t.fill) moving = true;
      if (c.alpha === 0) c.seq = FREE;

      const x = GUTTER + (i % g.cols) * g.size;
      const y = Math.floor(i / g.cols) * g.size;
      const s = g.size - GAP;
      ctx.lineWidth = 1;
      ctx.strokeStyle = i === 0 && g.per === 1 ? colors.warn : colors.border;
      ctx.beginPath();
      ctx.roundRect(x + 0.5, y + 0.5, s - 1, s - 1, 2);
      ctx.stroke();
      if (c.alpha > 0.003) {
        const color = c.seq === UNKNOWN ? colors.baseline : c.slot >= 0 ? colors.slots[c.slot] : colors.baseline;
        const dim = focus !== null && c.seq !== focus ? 0.3 : 1;
        ctx.fillStyle = color;
        ctx.globalAlpha = c.alpha * dim * 0.3;
        ctx.beginPath();
        ctx.roundRect(x, y, s, s, 2);
        ctx.fill();
        ctx.globalAlpha = c.alpha * dim;
        const h = Math.max(1, s * Math.min(1, c.fill / Math.max(c.alpha, 0.001)));
        ctx.beginPath();
        ctx.roundRect(x, y + s - h, s, h, 2);
        ctx.fill();
        ctx.globalAlpha = 1;
        if (focus !== null && c.seq === focus) {
          ctx.strokeStyle = colors.text;
          ctx.strokeRect(x - 0.5, y - 0.5, s + 1, s + 1);
        }
        const seq = c.seq >= 0 ? model.seqs.get(c.seq) : undefined;
        if (seq && g.per === 1 && s >= 13 && seq.blocks[0] === i && c.alpha > 0.5) {
          label(String(seq.id), x + s / 2, y + s / 2, colors.text, "center");
        }
      }
      if (i === hovered) {
        ctx.strokeStyle = colors.text;
        ctx.lineWidth = 1.5;
        ctx.strokeRect(x - 1, y - 1, s + 2, s + 2);
      }
    }

    const barY = g.gridH + 12;
    const barW = width - GUTTER;
    const used = model.totalBlocks - 1 - model.free();
    ctx.fillStyle = colors.border;
    ctx.beginPath();
    ctx.roundRect(GUTTER, barY, barW, BAR_H, 3);
    ctx.fill();
    ctx.fillStyle = colors.text;
    ctx.beginPath();
    ctx.roundRect(GUTTER, barY, Math.max(used ? 2 : 0, (barW * used) / model.totalBlocks), BAR_H, 3);
    ctx.fill();
    const shownW = Math.max(2, (barW * g.blocks) / model.totalBlocks);
    ctx.strokeStyle = colors.accent;
    ctx.lineWidth = 1;
    ctx.strokeRect(GUTTER + 0.5, barY - 2.5, shownW, BAR_H + 5);
    label("pool", GUTTER - 5, barY + BAR_H / 2, colors.muted, "right");

    const text = textRef.current;
    const last = Math.min(g.blocks, model.totalBlocks) - 1;
    const summary = `${fmt(used)} of ${fmt(model.totalBlocks)} blocks held (${((100 * used) / model.totalBlocks).toFixed(2)}%). The grid zooms in on blocks 0 to ${fmt(last)}${g.per > 1 ? `, ${g.per} per square` : ""}, where lowest-free-first allocation keeps every held block; the bar is the whole pool.`;
    if (text && summary !== textCache.current) {
      textCache.current = summary;
      text.textContent = summary;
    }
    return moving;
  }, 9);

  useEffect(() => {
    hoverRef.current = hover;
    kick();
  });

  useEffect(() => {
    if (hover === null) return;
    let timer = 0;
    const unsubscribe = feed.subscribe(() => {
      if (!timer) timer = window.setTimeout(() => {
        timer = 0;
        setVersion((v) => v + 1);
      }, 200);
    });
    return () => {
      unsubscribe();
      clearTimeout(timer);
    };
  }, [feed, hover]);

  const at = (cell: number): Hover | null => {
    const g = geoRef.current;
    if (!g || cell < 0 || cell >= Math.ceil(g.blocks / g.per)) return null;
    return { cell, x: GUTTER + (cell % g.cols) * g.size + g.size / 2, y: Math.floor(cell / g.cols) * g.size };
  };

  const point = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const g = geoRef.current;
    if (!g) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const col = Math.floor((e.clientX - rect.left - GUTTER) / g.size);
    const row = Math.floor((e.clientY - rect.top) / g.size);
    setHover(col < 0 || col >= g.cols || row < 0 || row >= g.rows ? null : at(row * g.cols + col));
  };

  const onKey = (e: React.KeyboardEvent) => {
    const g = geoRef.current;
    if (!g) return;
    if (e.key === "Escape") return setHover(null);
    const moves: Record<string, number> = { ArrowLeft: -1, ArrowRight: 1, ArrowUp: -g.cols, ArrowDown: g.cols };
    if (!(e.key in moves)) return;
    e.preventDefault();
    const next = at((hover?.cell ?? 0) + (hover ? moves[e.key] : 0));
    if (next) setHover(next);
  };

  const g = geo;
  const height = g?.gridH ?? 0;
  const tipW = 260;

  return (
    <div>
      <div className="relative" style={{ height: g ? g.gridH + BAR_H + 18 : 160 }}>
        <canvas
          ref={canvasRef}
          tabIndex={0}
          role="img"
          aria-label="KV block pool: one square per block of 16 token slots, colored by the request holding it and filled up to the slots written. Arrow keys read a block."
          aria-describedby="pool-tip"
          onKeyDown={onKey}
          onPointerMove={(e) => e.pointerType === "mouse" && point(e)}
          onPointerDown={point}
          onPointerLeave={(e) => e.pointerType === "mouse" && setHover(null)}
          onBlur={() => setHover(null)}
          className="block h-full w-full"
        />
        <div
          id="pool-tip"
          role="status"
          className={clsx("pointer-events-none absolute z-10 rounded border border-border bg-surface p-2 shadow-sm", !hover && "invisible")}
          style={
            hover && g
              ? { width: tipW, left: Math.max(0, Math.min(hover.x - tipW / 2, g.width - tipW)), ...(hover.y > height / 2 ? { bottom: `calc(100% - ${hover.y - 6}px)` } : { top: hover.y + g.size + 4 }) }
              : undefined
          }
        >
          {hover && g && <BlockDetails model={feed.model} first={hover.cell * g.per} per={g.per} live={feed.live} />}
        </div>
      </div>
      <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-[11px] text-muted">
        <span className="flex items-center gap-1.5">
          <span aria-hidden className="relative size-2.5 overflow-hidden rounded-sm bg-[var(--req-1)]/30">
            <span className="absolute inset-x-0 bottom-0 h-1/2 bg-[var(--req-1)]" />
          </span>
          held, filled to the slots written
        </span>
        <span className="flex items-center gap-1.5">
          <span aria-hidden className="size-2.5 rounded-sm bg-baseline" />
          held, request not seen
        </span>
        <span className="flex items-center gap-1.5">
          <span aria-hidden className="size-2.5 rounded-sm border border-warn" />
          block 0, never used
        </span>
      </div>
      <p ref={textRef} className="mt-2 text-[11px] leading-relaxed text-muted" />
    </div>
  );
}

function BlockDetails({ model, first, per, live }: { model: DashModel; first: number; per: number; live: boolean }) {
  if (per > 1) {
    let held = 0;
    for (let b = first; b < first + per; b++) if (model.owner[b] !== FREE) held++;
    return <p className="text-text">{`blocks ${fmt(first)} to ${fmt(first + per - 1)} · ${held} of ${per} held`}</p>;
  }
  const owner = model.owner[first];
  if (owner === RESERVED) return <p className="text-muted">block 0, the null block: never handed out, so a block table entry of 0 can mean nothing</p>;
  if (owner === FREE) return <p className="text-text">{`block ${fmt(first)} · free`}</p>;
  if (owner === UNKNOWN)
    return (
      <>
        <p className="text-text">{`block ${fmt(first)} · held`}</p>
        <p className="text-muted">by a request this page did not see start (before it connected, or during missed steps)</p>
      </>
    );
  const seq = model.seqs.get(owner);
  if (!seq) return null;
  const j = seq.blocks.indexOf(first);
  const filled = model.filled(seq, j);
  const from = j * model.blockSize;
  const outputs = seq.prompt === null ? [] : seq.outputs.slice(Math.max(0, from - seq.prompt), Math.max(0, from + filled - seq.prompt));
  const promptSlots = seq.prompt === null ? 0 : Math.max(0, Math.min(filled, seq.prompt - from));
  return (
    <div className="space-y-1">
      <p className="flex items-center gap-1.5 text-text">
        <span aria-hidden className="size-2 rounded-sm" style={{ background: slotColor(seq.slot) }} />
        {`block ${fmt(first)} · request #${seq.id}, logical block ${j} of ${seq.blocks.length}`}
      </p>
      <p className="text-muted">
        {filled ? `${filled} of ${model.blockSize} slots written, positions ${from} to ${from + filled - 1}` : "nothing written yet"}
        {` · handed out at step ${seq.since[j]}`}
      </p>
      {seq.prompt !== null && (promptSlots > 0 || outputs.length > 0) && (
        <p className="text-muted">
          {promptSlots > 0 && `${promptSlots} prompt token${promptSlots > 1 ? "s" : ""}`}
          {promptSlots > 0 && outputs.length > 0 && ", then "}
          {outputs.length > 0 && <span className="whitespace-pre-wrap text-text">{`"${outputs.join("")}"`}</span>}
        </p>
      )}
      {live && !seq.exact && <p className="text-muted">estimated: the step stream does not give prompt lengths, so which request took a new block is a best guess; the total matches the server</p>}
    </div>
  );
}
