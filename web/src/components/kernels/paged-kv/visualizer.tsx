"use client";

import clsx from "clsx";
import { useCallback, useEffect, useRef, useState } from "react";
import { kvDemo, kvReplay } from "@/lib/data";
import { usePrefersReducedMotion } from "@/lib/reduced-motion";
import { Toggle } from "../toggle";
import { type Cell, type Mode, Pool } from "./allocator";

const demo = kvDemo();
const realBlocks = kvReplay().meta.kv_blocks;
const B = 16;
const NB = 64;
const COLS = 8;
const kvHeads = 4;
const headDim = 64;
const blockStride = kvHeads * B * headDim * 2;
const plane = realBlocks * blockStride;
const stepMs = 90;

const kinds = demo.requests.map((r, i) => ({
  label: `L${i + 1}`,
  name: r.name,
  prompt: r.prompt,
  tokens: [...r.prompt_tokens, ...r.output_tokens],
  shape: { prompt: r.prompt_tokens.length, output: r.output_tokens.length, reserve: r.prompt_tokens.length + demo.max_new_tokens },
}));
const shapes = kinds.map((k) => k.shape);
const script = new Map<number, number[]>([
  [0, [2, 0]],
  [8, [1]],
  [16, [0]],
  [24, [2]],
  [40, [1, 0]],
  [60, [2]],
]);
const lastArrival = Math.max(...script.keys());
const seriesVar = ["--series-1", "--series-2", "--series-3"];

const fmt = (v: number) => v.toLocaleString("en-US");
const hex = (v: number) => `0x${v.toString(16).toUpperCase()}`;
const piece = (t: string) => (t === "<0x0A>" ? "⏎" : t.replaceAll("▁", "␣"));

class Sim {
  readonly paged = new Pool("paged", NB, B, shapes);
  readonly contiguous = new Pool("contiguous", NB, B, shapes);
  readonly scripted: boolean;
  nextId = 0;
  t = 0;

  constructor(scripted: boolean) {
    this.scripted = scripted;
  }

  add(kind: number) {
    const id = this.nextId++;
    this.paged.add(id, kind);
    this.contiguous.add(id, kind);
  }

  advance() {
    if (this.scripted) for (const kind of script.get(this.t) ?? []) this.add(kind);
    this.paged.step();
    this.contiguous.step();
    this.t++;
  }

  busy() {
    return this.paged.busy() || this.contiguous.busy() || (this.scripted && this.t <= lastArrival);
  }
}

type Shown = { h: number; ghost: number; kind: number; seq: number };

type Geo = { width: number; height: number; tw: number; th: number; hmax: number; ox: number; oy: number };

function geometry(width: number): Geo {
  const tw = Math.min((width - 40) / COLS, 72);
  const th = tw / 2;
  const hmax = tw * 0.8;
  const oy = 34 + hmax;
  return { width, height: oy + COLS * th + 30, tw, th, hmax, ox: width / 2, oy };
}

type Pt = [number, number];

function tile(g: Geo, id: number) {
  const r = Math.floor(id / COLS);
  const c = id % COLS;
  const cx = g.ox + ((c - r) * g.tw) / 2;
  const cy = g.oy + ((c + r) * g.th) / 2 + g.th / 2;
  const a = (g.tw / 2) * 0.84;
  const b = (g.th / 2) * 0.84;
  return { cx, cy, a, b };
}

function corners(g: Geo, id: number, z: number) {
  const { cx, cy, a, b } = tile(g, id);
  const up = z * g.hmax;
  return { T: [cx, cy - b - up] as Pt, R: [cx + a, cy - up] as Pt, Bt: [cx, cy + b - up] as Pt, L: [cx - a, cy - up] as Pt };
}

function poly(ctx: CanvasRenderingContext2D, pts: Pt[]) {
  ctx.beginPath();
  pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.closePath();
}

function silhouette(g: Geo, id: number, z: number): Pt[] {
  const lo = corners(g, id, 0);
  const hi = corners(g, id, z);
  return [lo.L, lo.Bt, lo.R, hi.R, hi.T, hi.L];
}

function inside([x, y]: Pt, pts: Pt[]) {
  let hit = false;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const [xi, yi] = pts[i];
    const [xj, yj] = pts[j];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) hit = !hit;
  }
  return hit;
}

const order = Array.from({ length: NB }, (_, i) => i).sort((p, q) => {
  const sp = Math.floor(p / COLS) + (p % COLS);
  const sq = Math.floor(q / COLS) + (q % COLS);
  return sp - sq || p - q;
});

type Colors = { text: string; muted: string; border: string; warn: string; bg: string; mono: string; series: string[] };

function readColors(): Colors {
  const s = getComputedStyle(document.documentElement);
  const v = (n: string) => s.getPropertyValue(n).trim();
  return { text: v("--text"), muted: v("--text-muted"), border: v("--border"), warn: v("--warn"), bg: v("--bg"), mono: v("--font-jetbrains") || "monospace", series: seriesVar.map(v) };
}

function prism(ctx: CanvasRenderingContext2D, g: Geo, id: number, z0: number, z1: number, fill: string, shade: boolean) {
  const lo = corners(g, id, z0);
  const hi = corners(g, id, z1);
  ctx.fillStyle = fill;
  poly(ctx, [lo.L, lo.Bt, hi.Bt, hi.L]);
  ctx.fill();
  if (shade) {
    ctx.fillStyle = "rgb(0 0 0 / 0.22)";
    ctx.fill();
  }
  ctx.fillStyle = fill;
  poly(ctx, [lo.Bt, lo.R, hi.R, hi.Bt]);
  ctx.fill();
  if (shade) {
    ctx.fillStyle = "rgb(0 0 0 / 0.38)";
    ctx.fill();
  }
  ctx.fillStyle = fill;
  poly(ctx, [hi.T, hi.R, hi.Bt, hi.L]);
  ctx.fill();
}

function drawPool(ctx: CanvasRenderingContext2D, g: Geo, shown: Shown[], hovered: number | null, c: Colors) {
  ctx.clearRect(0, 0, g.width, g.height);
  ctx.font = `10px ${c.mono}`;
  ctx.textBaseline = "middle";
  const focusSeq = hovered !== null && shown[hovered].ghost > 0.5 ? shown[hovered].seq : -1;
  for (const id of order) {
    const s = shown[id];
    const floor = corners(g, id, 0);
    poly(ctx, [floor.T, floor.R, floor.Bt, floor.L]);
    if (id === 0) {
      ctx.globalAlpha = 0.18;
      ctx.fillStyle = c.warn;
      ctx.fill();
      ctx.globalAlpha = 1;
    }
    ctx.strokeStyle = id === 0 ? c.warn : c.border;
    ctx.lineWidth = 1;
    ctx.stroke();
    const dim = focusSeq >= 0 && s.seq !== focusSeq ? 0.3 : 1;
    if (s.h > 0.004) {
      ctx.globalAlpha = dim;
      prism(ctx, g, id, 0, s.h, c.series[s.kind], true);
      ctx.globalAlpha = 1;
    }
    if (s.ghost > 0.01 && s.h < 0.995) {
      ctx.globalAlpha = 0.1 * s.ghost * dim;
      prism(ctx, g, id, s.h, 1, c.warn, false);
      ctx.globalAlpha = 0.6 * s.ghost * dim;
      const hi = corners(g, id, 1);
      const lo = corners(g, id, s.h);
      ctx.strokeStyle = c.warn;
      ctx.setLineDash([2, 2]);
      poly(ctx, [hi.T, hi.R, hi.Bt, hi.L]);
      ctx.stroke();
      ctx.beginPath();
      for (const k of ["L", "Bt", "R"] as const) {
        ctx.moveTo(...lo[k]);
        ctx.lineTo(...hi[k]);
      }
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
    }
    if (s.ghost > 0.5 && g.tw >= 44) {
      const top = corners(g, id, s.h);
      ctx.globalAlpha = dim;
      ctx.fillStyle = s.h > 0.05 ? c.bg : c.muted;
      ctx.textAlign = "center";
      ctx.fillText(`#${s.seq}`, top.T[0], (top.T[1] + top.Bt[1]) / 2);
      ctx.globalAlpha = 1;
    }
  }
  if (hovered !== null) {
    const s = shown[hovered];
    poly(ctx, silhouette(g, hovered, s.ghost > 0.5 ? 1 : s.h));
    ctx.strokeStyle = c.text;
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }
  ctx.fillStyle = c.muted;
  ctx.textAlign = "right";
  for (let r = 0; r < COLS; r++) {
    const { L } = corners(g, r * COLS, 0);
    ctx.fillText(String(r * COLS), L[0] - 6, L[1] + 4);
  }
  ctx.textAlign = "left";
  for (let col = 0; col < COLS; col++) {
    const { R } = corners(g, (COLS - 1) * COLS + col, 0);
    ctx.fillText(`+${col}`, R[0] - 2, R[1] + 12);
  }
  ctx.textAlign = "center";
  ctx.fillText(g.width < 520 ? "height = slots written, 16 per block" : "block id = row start + column; height = slots written, 16 per block", g.width / 2, 12);
}

export function KvVisualizer() {
  const reduced = usePrefersReducedMotion();
  const [sim, setSim] = useState(() => new Sim(true));
  const [, setVersion] = useState(0);
  const [mode, setMode] = useState<Mode>("paged");
  const [playing, setPlaying] = useState(false);
  const [hover, setHover] = useState<number | null>(null);
  const [width, setWidth] = useState(0);
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const shown = useRef<Shown[]>(Array.from({ length: NB }, () => ({ h: 0, ghost: 0, kind: 0, seq: -1 })));
  const colors = useRef<Colors | null>(null);
  const raf = useRef(0);
  const started = useRef(false);
  const g = geometry(width || 560);
  const pool = sim[mode];
  const simRef = useRef(sim);
  const modeRef = useRef(mode);
  const reducedRef = useRef(reduced);
  const hoverRef = useRef(hover);

  const kick = useCallback(() => {
    if (raf.current) return;
    let last = performance.now();
    const frame = (now: number) => {
      raf.current = 0;
      const canvas = canvasRef.current;
      const ctx = canvas?.getContext("2d");
      if (!canvas || !ctx || !canvas.clientWidth) return;
      colors.current ??= readColors();
      const geo = geometry(canvas.clientWidth);
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      if (canvas.width !== Math.round(geo.width * dpr)) {
        canvas.width = Math.round(geo.width * dpr);
        canvas.height = Math.round(geo.height * dpr);
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const dt = Math.min((now - last) / 1000, 0.1);
      last = now;
      const cells = simRef.current[modeRef.current].cells();
      const snap = reducedRef.current;
      const kh = snap ? 1 : 1 - Math.exp(-dt / 0.16);
      const kg = snap ? 1 : 1 - Math.exp(-dt / 0.24);
      let moving = false;
      shown.current.forEach((s, i) => {
        const cell = cells[i];
        const th = cell ? cell.filled / B : 0;
        const tg = cell ? 1 : 0;
        if (cell && (s.seq !== cell.seq || s.h < 0.004)) {
          if (s.seq !== cell.seq && s.h > 0.004) {
            s.h += (0 - s.h) * kh;
            s.ghost += (0 - s.ghost) * kg;
            moving = true;
            return;
          }
          s.seq = cell.seq;
          s.kind = cell.kind;
        }
        s.h += (th - s.h) * kh;
        s.ghost += (tg - s.ghost) * kg;
        if (Math.abs(th - s.h) < 0.002) s.h = th;
        if (Math.abs(tg - s.ghost) < 0.002) s.ghost = tg;
        if (s.h !== th || s.ghost !== tg) moving = true;
        if (!cell && s.ghost === 0 && s.h === 0) s.seq = -1;
      });
      drawPool(ctx, geo, shown.current, hoverRef.current, colors.current);
      if (moving) raf.current = requestAnimationFrame(frame);
    };
    raf.current = requestAnimationFrame(frame);
  }, []);

  useEffect(() => {
    simRef.current = sim;
    modeRef.current = mode;
    reducedRef.current = reduced;
    hoverRef.current = hover;
    kick();
  });

  useEffect(() => {
    const wrap = wrapRef.current;
    if (!wrap) return;
    const resize = new ResizeObserver(() => setWidth(wrap.clientWidth));
    resize.observe(wrap);
    const seen = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting && !started.current) {
        started.current = true;
        if (!window.matchMedia("(prefers-reduced-motion: reduce)").matches) setPlaying(true);
      }
    });
    seen.observe(wrap);
    const theme = new MutationObserver(() => {
      colors.current = readColors();
      kick();
    });
    theme.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => {
      resize.disconnect();
      seen.disconnect();
      theme.disconnect();
      cancelAnimationFrame(raf.current);
      raf.current = 0;
    };
  }, [kick]);

  useEffect(() => {
    if (!playing) return;
    const timer = setInterval(() => {
      sim.advance();
      setVersion((v) => v + 1);
      if (!sim.busy()) setPlaying(false);
    }, stepMs);
    return () => clearInterval(timer);
  }, [playing, sim]);

  const restart = (scripted: boolean) => {
    setSim(new Sim(scripted));
    setHover(null);
    setPlaying(scripted);
  };

  const pick = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const p: Pt = [e.clientX - rect.left, e.clientY - rect.top];
    const geo = geometry(rect.width);
    for (let k = order.length - 1; k >= 0; k--) {
      const id = order[k];
      const s = shown.current[id];
      if (inside(p, silhouette(geo, id, s.ghost > 0.5 ? 1 : s.h))) return setHover(id);
    }
    setHover(null);
  };

  const cells = pool.cells();
  const busy = sim.busy();
  const totals = { paged: sim.paged.totals(), contiguous: sim.contiguous.totals() };
  const rows: { label: string; value: (t: ReturnType<Pool["totals"]>) => string; warn?: boolean }[] = [
    { label: "requests running", value: (t) => String(t.running) },
    { label: "waiting for room", value: (t) => String(t.waiting), warn: true },
    { label: `blocks held of ${NB - 1}`, value: (t) => String(t.held) },
    { label: "slots written", value: (t) => fmt(t.written) },
    {
      label: "slots held, empty",
      value: (t) => `${fmt(t.empty)}${t.held ? ` (${Math.round((100 * t.empty) / (t.held * B))}%)` : ""}`,
      warn: true,
    },
    { label: "preemptions", value: (t) => String(t.preemptions) },
  ];

  const tipAt = hover === null ? null : corners(g, hover, cells[hover] ? 1 : 0).T;
  const tipW = Math.min(288, g.width - 16);

  return (
    <div className="grid gap-6 font-mono text-xs lg:grid-cols-[minmax(0,1fr)_19rem]">
      <div className="space-y-3">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-muted">add a request</span>
          {kinds.map((k, i) => (
            <button
              key={k.label}
              type="button"
              onClick={() => {
                sim.add(i);
                setVersion((v) => v + 1);
                setPlaying(true);
              }}
              title={`"${k.prompt}": ${k.shape.prompt} prompt tokens, ${k.shape.output} generated`}
              className="pressable flex items-center gap-1.5 rounded border border-border px-2.5 py-1 text-text"
            >
              <span aria-hidden className="size-2 rounded-sm" style={{ background: `var(${seriesVar[i]})` }} />
              {`+ ${k.label} · ${k.shape.prompt + k.shape.output} tokens`}
            </button>
          ))}
        </div>
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
          <Toggle label="cache" choices={["paged", "contiguous"]} value={mode} onChange={(v) => setMode(v as Mode)} />
          <div className="flex gap-2">
            <button type="button" disabled={!busy} onClick={() => setPlaying(!playing)} aria-pressed={playing} className="pressable rounded border border-border px-2.5 py-1 text-accent disabled:opacity-40">
              {playing ? "pause" : "play"}
            </button>
            <button
              type="button"
              disabled={!busy || playing}
              onClick={() => {
                sim.advance();
                setVersion((v) => v + 1);
              }}
              className="pressable rounded border border-border px-2.5 py-1 text-text disabled:opacity-40"
            >
              step
            </button>
            <button type="button" onClick={() => restart(true)} className="pressable rounded border border-border px-2.5 py-1 text-text">
              replay demo
            </button>
            <button type="button" onClick={() => restart(false)} className="pressable rounded border border-border px-2.5 py-1 text-text">
              clear
            </button>
          </div>
        </div>
        <div ref={wrapRef} className="relative">
          <canvas
            ref={canvasRef}
            className="block w-full touch-manipulation"
            style={{ height: g.height }}
            role="img"
            aria-label={`A ${NB}-block KV pool drawn in 3D, ${mode} allocation, step ${sim.t}: ${totals[mode].running} requests hold ${totals[mode].held} blocks with ${totals[mode].written} slots written; the block tables are listed beside it`}
            onPointerMove={pick}
            onPointerDown={pick}
            onPointerLeave={(e) => e.pointerType === "mouse" && setHover(null)}
          />
          {hover !== null && tipAt && (
            <div
              className="pointer-events-none absolute z-10 space-y-1.5 rounded border border-border bg-surface p-2.5 shadow-sm"
              style={{
                width: tipW,
                left: Math.max(0, Math.min(tipAt[0] - tipW / 2, g.width - tipW)),
                ...(tipAt[1] > g.height / 2 ? { bottom: g.height - tipAt[1] + 8 } : { top: tipAt[1] + g.th + 8 }),
              }}
            >
              <BlockDetails id={hover} cell={cells[hover]} pool={pool} />
            </div>
          )}
        </div>
        <div className="flex flex-wrap gap-x-4 gap-y-1 text-muted">
          {kinds.map((k, i) => (
            <span key={k.label} className="flex items-center gap-1.5">
              <span aria-hidden className="size-2 rounded-sm" style={{ background: `var(${seriesVar[i]})` }} />
              {`${k.label} ${k.name}: ${k.shape.prompt} + ${k.shape.output}`}
            </span>
          ))}
          <span className="flex items-center gap-1.5">
            <span aria-hidden className="size-2 rounded-sm border border-dashed border-warn" />
            held, not written
          </span>
        </div>
      </div>

      <div className="space-y-5">
        <table className="w-full tabular-nums">
          <thead className="text-left text-muted">
            <tr>
              <th className="pb-1 font-normal">{`step ${sim.t}`}</th>
              {(["paged", "contiguous"] as const).map((m) => (
                <th key={m} className={clsx("pb-1 text-right font-normal", m === mode ? "text-accent" : "text-muted")}>
                  {m}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.label} className="border-t border-border">
                <td className="py-1 text-muted">{r.label}</td>
                {(["paged", "contiguous"] as const).map((m) => {
                  const v = r.value(totals[m]);
                  return (
                    <td key={m} className={clsx("py-1 text-right", r.warn && !v.startsWith("0") ? "text-warn" : m === mode ? "text-text" : "text-muted")}>
                      {v}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>

        <div className="space-y-2">
          <div className="text-muted">{mode === "paged" ? "block tables, logical block → physical block" : "reservations, prompt + 256 tokens each"}</div>
          <ul className="max-h-72 space-y-2 overflow-y-auto pr-1">
            {pool.running.map((id) => {
              const s = pool.seqs.get(id)!;
              const k = kinds[s.kind];
              return (
                <li key={id} className="space-y-1">
                  <div className="flex items-center gap-1.5">
                    <span aria-hidden className="size-2 rounded-sm" style={{ background: `var(${seriesVar[s.kind]})` }} />
                    <span className="text-text">{`#${id} ${k.label}`}</span>
                    <span className="text-muted">{`${s.slots} / ${k.shape.prompt + k.shape.output} tokens`}</span>
                  </div>
                  <div className="flex flex-wrap gap-1">
                    {(mode === "paged" ? s.table : [s.table[0], s.table[s.table.length - 1]]).map((b, j) => (
                      <button
                        key={`${j}-${b}`}
                        type="button"
                        onMouseEnter={() => setHover(b)}
                        onMouseLeave={() => setHover(null)}
                        onFocus={() => setHover(b)}
                        onBlur={() => setHover(null)}
                        aria-label={`request ${id}, logical block ${mode === "paged" ? j : j * (s.table.length - 1)}: physical block ${b}`}
                        className={clsx("rounded border px-1.5 py-0.5 tabular-nums", hover === b ? "border-text text-text" : "border-border text-muted")}
                      >
                        {mode === "paged" ? b : j ? `to ${b}` : b}
                      </button>
                    ))}
                  </div>
                </li>
              );
            })}
            {pool.waiting.map((id) => {
              const s = pool.seqs.get(id)!;
              return (
                <li key={id} className="text-warn">
                  {`#${id} ${kinds[s.kind].label} waits: needs ${pool.need(s)} ${mode === "paged" ? `free blocks, ${pool.numFree()} free` : `blocks in a row, longest free run ${pool.longestRun()} (${pool.numFree()} free in all)`}`}
                </li>
              );
            })}
            {!pool.running.length && !pool.waiting.length && <li className="text-muted">nothing running; add a request</li>}
          </ul>
        </div>
      </div>
    </div>
  );
}

function BlockDetails({ id, cell, pool }: { id: number; cell: Cell | null; pool: Pool }) {
  if (id === 0) return <p className="text-muted">block 0, the null block: never handed out, so a table entry of 0 can mean nothing here</p>;
  const k = `K, layer 0: pool + ${hex(id * blockStride)}`;
  const v = `V, layer 0: pool + ${hex(plane + id * blockStride)}`;
  const address = (
    <div className="space-y-0.5 border-t border-border pt-1.5 text-[11px] text-muted">
      <div className="text-text">{k}</div>
      <div className="text-text">{v}</div>
      <div>{`layer l adds 2l × ${fmt(plane)} B (offsets in the engine's ${fmt(realBlocks)}-block pool)`}</div>
      <div>{`[block, kv head, slot, dim] strides ${blockStride / 2} · ${(B * headDim)} · ${headDim} · 1 BF16, ${kvHeads} KV heads × ${headDim} dims`}</div>
    </div>
  );
  if (!cell) {
    const next = pool.free.indexOf(true, 1);
    return (
      <>
        <p className="text-text">{`block ${id} · free`}</p>
        <p className="text-muted">{next === id ? "the lowest free block, so the next one handed out" : `handed out after the lower free blocks (next is ${next})`}</p>
        {address}
      </>
    );
  }
  const s = pool.seqs.get(cell.seq)!;
  const kind = kinds[s.kind];
  const from = cell.logical * B;
  const toks = kind.tokens.slice(from, from + cell.filled);
  return (
    <>
      <p className="text-text">{`block ${id} · request #${s.id}, ${kind.label} ${kind.name}`}</p>
      <p className="text-muted">
        {`logical block ${cell.logical} of ${s.table.length} · `}
        {cell.filled ? `slots ${cell.filled} / ${B}, positions ${from} to ${from + cell.filled - 1}` : "nothing written yet"}
        {cell.filled < B && <span className="text-warn">{` · ${B - cell.filled} empty`}</span>}
      </p>
      {toks.length > 0 && (
        <div className="flex flex-wrap gap-0.5">
          {toks.map((t, i) => (
            <span key={i} className={clsx("rounded px-1 py-px", from + i < kind.shape.prompt ? "bg-border text-muted" : "bg-accent-dim text-text")}>
              {piece(t)}
            </span>
          ))}
        </div>
      )}
      {address}
    </>
  );
}
