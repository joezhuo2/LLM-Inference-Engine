"use client";

import clsx from "clsx";
import { useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { roofline } from "@/content/kernels/gemm";
import { ceilings, hardware } from "@/lib/data";
import { usePrefersReducedMotion } from "@/lib/reduced-motion";
import { EasedBar } from "../eased-bar";
import { type Prediction, Predict } from "../predict";
import { Toggle } from "../toggle";
import type { Projection } from "./figures";
import { gemmNumbers } from "./story";

const peak = ceilings();
const sms = hardware().sms;
const ridge = (peak.tflops * 1e12) / (peak.gbs * 1e9);
const ops = gemmNumbers.projections;
const opNames = ops.map((p) => p.label);
const rowNames = ["1", "16", "512"];
const measuredRows = new Set(["1", "16"]);
const N = 16;
const NT = N * N;
const waves = Math.ceil(NT / sms);
const BF16 = 2;
const waveMs = 900;
const tweenMs = 700;
const slab = 0.14;
const lv = 0.086;
const seriesVar = ["--series-1", "--series-2", "--series-3"];
const start = { op: "gate/up", rows: "1" };

const fmt = (v: number) => v.toLocaleString("en-US");
const size = (b: number) => (b >= 1e9 ? `${(b / 1e9).toFixed(2)} GB` : b >= 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${(b / 1e3).toFixed(1)} KB`);
const usText = (v: number) => (v >= 100 ? fmt(Math.round(v)) : v.toFixed(1));
const plural = (m: number, one: string, many: string) => (m === 1 ? one : many);
const levels = (m: number) => Math.log2(m) + 1;
const ridgeZ = slab + levels(ridge) * lv;
const pillarTop = (m: number) => slab + levels(m) * lv;
const edge = (n: number, i: number) => Math.floor((i * n) / N);

const readAt = Array.from({ length: NT }, (_, id) => (id % N) * N + Math.floor(id / N));
const smOf = (id: number) => readAt[id] % sms;
const waveOf = (id: number) => Math.floor(readAt[id] / sms);

function cost(p: Projection, m: number) {
  const weights = p.out * p.in * BF16;
  const act = m * (p.in + p.out) * BF16;
  const bytes = (weights + act) * p.per;
  const flops = 2 * m * p.out * p.in * p.per;
  const memUs = bytes / (peak.gbs * 1e3);
  const computeUs = flops / (peak.tflops * 1e6);
  return { weights, act, bytes, flops, intensity: flops / bytes, memUs, computeUs, bound: computeUs > memUs ? "compute" : "memory" };
}

const measuredUs = (p: Projection, rows: string) => (measuredRows.has(rows) ? roofline(p.op, rows, "us/step") : null);

function tileShape(p: Projection, id: number) {
  const r = Math.floor(id / N);
  const c = id % N;
  const r0 = edge(p.out, r);
  const r1 = edge(p.out, r + 1);
  const c0 = edge(p.in, c);
  const c1 = edge(p.in, c + 1);
  return { r0, r1, c0, c1, weights: (r1 - r0) * (c1 - c0) };
}

const readBytes = (p: Projection, done: number) => readAt.reduce((s, t, id) => (t < done ? s + tileShape(p, id).weights * BF16 : s), 0);

const gateUp = ops.find((p) => p.op === "gate_up");
if (!gateUp) throw new Error("the gemm projections have no gate_up");
const us1 = roofline("gate_up", "1", "us/step");
const us16 = roofline("gate_up", "16", "us/step");
const slowdown = us16 / us1;
const guesses = [
  { label: "about the same", f: 1 },
  { label: "about 4× longer", f: 4 },
  { label: "about 16× longer", f: 16 },
];
const closest = guesses.reduce((a, b) => (Math.abs(Math.log(b.f / slowdown)) < Math.abs(Math.log(a.f / slowdown)) ? b : a));

const prediction: Prediction = {
  question: "Switch rows M from 1 to 16: gate/up then does 16 times the multiply-adds. How long does its step take?",
  choices: guesses.map((g) => g.label),
  answer: closest.label,
  why: `Measured: ${fmt(us1)} µs per step at 1 row and ${fmt(us16)} µs at 16, ${slowdown.toFixed(2)}×. Each weight is still read from DRAM once, it just feeds 16 multiply-adds instead of 1: ${cost(gateUp, 16).intensity.toFixed(1)} FLOP per byte, far under the ridge at ${Math.round(ridge)}, so the time is still the time to read ${size(cost(gateUp, 16).bytes)}. Try 512 next.`,
};

type Geo = { width: number; height: number; tw: number; th: number; hmax: number; ox: number; oy: number };

const span = N + 1.8;

function geometry(width: number): Geo {
  const margin = width < 520 ? 16 : 44;
  const tw = Math.min((width - 2 * margin) / span, 36);
  const th = tw / 2;
  const hmax = tw * 2.8;
  const oy = 30 + hmax;
  return { width, height: oy + span * th + 30, tw, th, hmax, ox: width / 2, oy };
}

type Pt = [number, number];
type Cell = { c: number; r: number; sc: number; sr: number };

const wCell = (id: number): Cell => ({ c: (id % N) + 0.5, r: Math.floor(id / N) + 0.5, sc: 0.42, sr: 0.42 });
const pCell = (id: number): Cell => ({ ...wCell(id), sc: 0.2, sr: 0.2 });
const xCell = (k: number): Cell => ({ c: k + 0.5, r: N + 0.55, sc: 0.42, sr: 0.22 });
const yCell = (k: number): Cell => ({ c: N + 0.55, r: k + 0.5, sc: 0.22, sr: 0.42 });

function at(g: Geo, c: number, r: number, z: number): Pt {
  return [g.ox + ((c - r) * g.tw) / 2, g.oy + ((c + r) * g.th) / 2 - z * g.hmax];
}

function corners(g: Geo, { c, r, sc, sr }: Cell, z: number) {
  return { T: at(g, c - sc, r - sr, z), R: at(g, c + sc, r - sr, z), Bt: at(g, c + sc, r + sr, z), L: at(g, c - sc, r + sr, z) };
}

function poly(ctx: CanvasRenderingContext2D, pts: Pt[]) {
  ctx.beginPath();
  pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.closePath();
}

function silhouette(g: Geo, cell: Cell, z0: number, z1: number): Pt[] {
  const lo = corners(g, cell, z0);
  const hi = corners(g, cell, z1);
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

type Item = { kind: "w" | "x" | "y"; k: number; d: number };

const drawList: Item[] = [
  ...Array.from({ length: NT }, (_, id): Item => ({ kind: "w", k: id, d: wCell(id).c + wCell(id).r })),
  ...Array.from({ length: N }, (_, k): Item => ({ kind: "x", k, d: xCell(k).c + xCell(k).r })),
  ...Array.from({ length: N }, (_, k): Item => ({ kind: "y", k, d: yCell(k).c + yCell(k).r })),
].sort((a, b) => a.d - b.d);

const tilesFrontToBack = drawList.filter((it) => it.kind === "w").reverse();

type Colors = { text: string; muted: string; border: string; warn: string; accent: string; baseline: string; mono: string; series: string[] };

function readColors(el: Element): Colors {
  const s = getComputedStyle(el);
  const v = (n: string) => s.getPropertyValue(n).trim();
  return { text: v("--text"), muted: v("--text-muted"), border: v("--border"), warn: v("--warn"), accent: v("--accent"), baseline: v("--baseline"), mono: v("--font-jetbrains") || "monospace", series: seriesVar.map(v) };
}

function prism(ctx: CanvasRenderingContext2D, g: Geo, cell: Cell, z0: number, z1: number, fill: string) {
  const lo = corners(g, cell, z0);
  const hi = corners(g, cell, z1);
  ctx.fillStyle = fill;
  poly(ctx, [lo.L, lo.Bt, hi.Bt, hi.L]);
  ctx.fill();
  ctx.fillStyle = "rgb(25 24 19 / 0.22)";
  ctx.fill();
  ctx.fillStyle = fill;
  poly(ctx, [lo.Bt, lo.R, hi.R, hi.Bt]);
  ctx.fill();
  ctx.fillStyle = "rgb(25 24 19 / 0.38)";
  ctx.fill();
  ctx.fillStyle = fill;
  poly(ctx, [hi.T, hi.R, hi.Bt, hi.L]);
  ctx.fill();
}

function ghost(ctx: CanvasRenderingContext2D, g: Geo, cell: Cell, z0: number, z1: number, color: string) {
  const lo = corners(g, cell, z0);
  const hi = corners(g, cell, z1);
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
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
}

type Tw = { from: number; to: number; t0: number };

const easeInOut = (u: number) => (u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2);
const val = (t: Tw, now: number) => t.from + (t.to - t.from) * easeInOut(Math.min(1, Math.max(0, (now - t.t0) / tweenMs)));
const moving = (t: Tw, now: number) => t.from !== t.to && now - t.t0 < tweenMs;

function aim(t: Tw, to: number, now: number, snap: boolean, delay = 0) {
  if (snap) {
    t.from = to;
    t.to = to;
    return;
  }
  if (t.to === to) return;
  t.from = val(t, now);
  t.to = to;
  t.t0 = now + delay;
}

const tw0 = (): Tw => ({ from: 0, to: 0, t0: 0 });

type Shown = { h: Tw[]; u: Tw[]; strip: Tw; yf: Tw[] };

type Scene = { h: number[]; u: number[]; strip: number; yf: number[]; active: boolean[]; hover: number | null; labels: { x: string; y: string; head: string } };

function drawScene(ctx: CanvasRenderingContext2D, g: Geo, s: Scene, c: Colors) {
  ctx.clearRect(0, 0, g.width, g.height);
  ctx.font = `10px ${c.mono}`;
  ctx.textBaseline = "middle";
  for (const it of drawList) {
    if (it.kind === "w") {
      const id = it.k;
      const cell = wCell(id);
      const floor = corners(g, cell, 0);
      poly(ctx, [floor.T, floor.R, floor.Bt, floor.L]);
      ctx.strokeStyle = c.border;
      ctx.lineWidth = 1;
      ctx.stroke();
      const h = s.h[id] * slab;
      if (h > 0.002) prism(ctx, g, cell, 0, h, c.series[smOf(id) % c.series.length]);
      const u = s.u[id];
      if (u > 0.002) {
        const p = pCell(id);
        const z1 = h + u;
        prism(ctx, g, p, h, Math.min(z1, ridgeZ), c.accent);
        if (z1 > ridgeZ) prism(ctx, g, p, ridgeZ, z1, c.warn);
      }
    } else if (it.kind === "x") {
      prism(ctx, g, xCell(it.k), 0, Math.max(s.strip, 0.004), s.active[it.k] ? c.accent : c.baseline);
    } else {
      const cell = yCell(it.k);
      const f = s.yf[it.k] * s.strip;
      const floor = corners(g, cell, 0);
      poly(ctx, [floor.T, floor.R, floor.Bt, floor.L]);
      ctx.strokeStyle = c.border;
      ctx.stroke();
      if (f > 0.002) prism(ctx, g, cell, 0, f, c.accent);
      if (s.strip - f > 0.004) {
        ctx.globalAlpha = 0.7;
        ghost(ctx, g, cell, f, s.strip, c.muted);
        ctx.globalAlpha = 1;
      }
    }
  }
  if (s.hover !== null) {
    const id = s.hover;
    const h = s.h[id] * slab;
    ctx.strokeStyle = c.text;
    ctx.lineWidth = 1.5;
    poly(ctx, silhouette(g, wCell(id), 0, Math.max(h, 0)));
    ctx.stroke();
    if (s.u[id] > 0.002) {
      poly(ctx, silhouette(g, pCell(id), h, h + s.u[id]));
      ctx.stroke();
    }
    poly(ctx, silhouette(g, xCell(id % N), 0, Math.max(s.strip, 0.004)));
    ctx.stroke();
    poly(ctx, silhouette(g, yCell(Math.floor(id / N)), 0, Math.max(s.strip, 0.004)));
    ctx.stroke();
  }
  ctx.globalAlpha = 0.85;
  ctx.strokeStyle = c.warn;
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 3]);
  poly(ctx, [at(g, 0, 0, ridgeZ), at(g, N, 0, ridgeZ), at(g, N, N, ridgeZ), at(g, 0, N, ridgeZ)]);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.globalAlpha = 1;
  const back = at(g, 0, 0, ridgeZ);
  ctx.fillStyle = c.warn;
  ctx.textAlign = "center";
  ctx.fillText(g.width < 520 ? `ridge ${Math.round(ridge)}` : `ridge: ${Math.round(ridge)} uses per weight`, back[0], back[1] - 8);
  ctx.fillStyle = c.muted;
  const xl = at(g, N / 2, N + 0.8, 0);
  ctx.textAlign = "right";
  ctx.fillText(s.labels.x, xl[0] - 4, xl[1] + 8);
  const yl = at(g, N + 0.8, N / 2, 0);
  ctx.textAlign = "left";
  ctx.fillText(s.labels.y, yl[0] + 4, yl[1] + 8);
  ctx.textAlign = "center";
  ctx.fillText(s.labels.head, g.width / 2, 12);
}

export function GemmStream({ pages }: { pages?: { view: HTMLElement | null; table: HTMLElement | null } } = {}) {
  const reduced = usePrefersReducedMotion();
  const [op, setOp] = useState(start.op);
  const [rows, setRows] = useState(start.rows);
  const [done, setDone] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [hover, setHover] = useState<number | null>(null);
  const [width, setWidth] = useState(0);
  const [wrap, setWrap] = useState<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const shown = useRef<Shown>({ h: Array.from({ length: NT }, tw0), u: Array.from({ length: NT }, tw0), strip: tw0(), yf: Array.from({ length: N }, tw0) });
  const colors = useRef<Colors | null>(null);
  const raf = useRef(0);
  const started = useRef(false);
  const state = useRef({ op, rows, done, hover, reduced });
  const g = geometry(width || 560);
  const p = ops[opNames.indexOf(op)];
  const m = Number(rows);

  const kick = useCallback(() => {
    if (raf.current) return;
    const frame = () => {
      raf.current = 0;
      const canvas = canvasRef.current;
      const ctx = canvas?.getContext("2d");
      if (!canvas || !ctx || !canvas.clientWidth) return;
      colors.current ??= readColors(canvas);
      const geo = geometry(canvas.clientWidth);
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      if (canvas.width !== Math.round(geo.width * dpr) || canvas.height !== Math.round(geo.height * dpr)) {
        canvas.width = Math.round(geo.width * dpr);
        canvas.height = Math.round(geo.height * dpr);
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const now = performance.now();
      const st = state.current;
      const snap = st.reduced;
      const mm = Number(st.rows);
      const pp = ops[opNames.indexOf(st.op)];
      const sh = shown.current;
      const use = pillarTop(mm) - slab;
      let busy = false;
      for (let id = 0; id < NT; id++) {
        const read = readAt[id] < st.done;
        const rising = read && sh.h[id].to === 0;
        aim(sh.h[id], read ? 1 : 0, now, snap);
        aim(sh.u[id], read ? use : 0, now, snap, rising ? 300 : 0);
        busy ||= moving(sh.h[id], now) || moving(sh.u[id], now);
      }
      aim(sh.strip, levels(mm) * lv, now, snap);
      busy ||= moving(sh.strip, now);
      sh.yf.forEach((t, r) => {
        aim(t, Math.max(0, Math.min(N, Math.ceil((st.done - r) / N))) / N, now, snap);
        busy ||= moving(t, now);
      });
      const waveStart = (Math.ceil(st.done / sms) - 1) * sms;
      const active = Array.from({ length: N }, () => false);
      if (st.done > 0 && st.done < NT) {
        for (let id = 0; id < NT; id++) if (readAt[id] >= waveStart && readAt[id] < st.done) active[id % N] = true;
      }
      const narrow = geo.width < 520;
      drawScene(
        ctx,
        geo,
        {
          h: sh.h.map((t) => val(t, now)),
          u: sh.u.map((t) => val(t, now)),
          strip: val(sh.strip, now),
          yf: sh.yf.map((t) => val(t, now)),
          active,
          hover: st.hover,
          labels: {
            x: `x: ${fmt(mm)} × ${fmt(pp.in)}`,
            y: `y: ${fmt(mm)} × ${fmt(pp.out)}`,
            head: narrow ? "slab = bytes, pillar = uses per weight" : "slab = a tile's bytes, by SM; pillar = uses per weight, one step per doubling",
          },
        },
        colors.current,
      );
      if (busy) raf.current = requestAnimationFrame(frame);
    };
    raf.current = requestAnimationFrame(frame);
  }, []);

  useEffect(() => {
    state.current = { op, rows, done, hover, reduced };
    kick();
  });

  useEffect(() => {
    if (!wrap) return;
    colors.current = null;
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
      colors.current = null;
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
  }, [kick, wrap]);

  useEffect(() => {
    if (!playing) return;
    const timer = setTimeout(() => {
      const next = Math.min(NT, done + sms);
      setDone(next);
      if (next >= NT) setPlaying(false);
    }, waveMs);
    return () => clearTimeout(timer);
  }, [playing, done]);

  const pickOp = (v: string) => {
    setOp(v);
    setDone(0);
    setHover(null);
    setPlaying(!reduced);
  };

  const hit = (pt: Pt, geo: Geo) => {
    for (const it of tilesFrontToBack) {
      const id = it.k;
      if (readAt[id] < done ? inside(pt, silhouette(geo, pCell(id), slab, pillarTop(m))) || inside(pt, silhouette(geo, wCell(id), 0, slab)) : inside(pt, silhouette(geo, wCell(id), 0, 0))) return id;
    }
    return null;
  };

  const pick = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const scale = e.currentTarget.clientWidth / rect.width;
    setHover(hit([(e.clientX - rect.left) * scale, (e.clientY - rect.top) * scale], geometry(e.currentTarget.clientWidth)));
  };

  const keys: Record<string, [number, number]> = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] };
  const onKey = (e: React.KeyboardEvent<HTMLCanvasElement>) => {
    if (e.key === "Escape") return setHover(null);
    const mv = keys[e.key];
    if (!mv) return;
    e.preventDefault();
    const cur = hover ?? 0;
    const r = Math.min(N - 1, Math.max(0, Math.floor(cur / N) + mv[0]));
    const c = Math.min(N - 1, Math.max(0, (cur % N) + mv[1]));
    setHover(r * N + c);
  };

  const costs = rowNames.map((r) => ({ rows: r, k: cost(p, Number(r)), us: measuredUs(p, r) }));
  const sel = costs[rowNames.indexOf(rows)];
  const read = Math.min(done, NT);
  const wave = Math.ceil(read / sms);
  const readB = readBytes(p, read);
  const layerW = p.out * p.in * BF16;
  const tile0 = tileShape(p, 0);

  const tableRows: { label: string; value: (x: (typeof costs)[number]) => string; warn?: (x: (typeof costs)[number]) => boolean }[] = [
    { label: "W, one layer", value: (x) => size(x.k.weights) },
    { label: "x and y, one layer", value: (x) => size(x.k.act) },
    { label: p.per > 1 ? `bytes per step, × ${p.per}` : "bytes per step, once", value: (x) => size(x.k.bytes) },
    { label: "GFLOP per step", value: (x) => (x.k.flops / 1e9).toFixed(x.k.flops < 1e10 ? 2 : 0) },
    { label: `FLOP per byte, ridge ${Math.round(ridge)}`, value: (x) => x.k.intensity.toFixed(1), warn: (x) => x.k.intensity > ridge },
    { label: "memory roof, µs", value: (x) => usText(x.k.memUs) },
    { label: "compute roof, µs", value: (x) => usText(x.k.computeUs), warn: (x) => x.k.bound === "compute" },
    { label: "the ceiling that binds", value: (x) => x.k.bound, warn: (x) => x.k.bound === "compute" },
    { label: "measured, µs per step", value: (x) => (x.us === null ? "not measured" : usText(x.us)) },
  ];

  const longest = Math.max(sel.k.memUs, sel.k.computeUs, sel.us ?? 0);
  const bars = [
    { label: "at the memory roof", value: sel.k.memUs, bar: "bg-accent", text: "text-accent" },
    { label: "at the compute roof", value: sel.k.computeUs, bar: "bg-warn", text: "text-warn" },
    ...(sel.us === null ? [] : [{ label: "measured", value: sel.us, bar: "bg-baseline", text: "text-text" }]),
  ];

  const tipAt = hover === null ? null : corners(g, wCell(hover), readAt[hover] < done ? pillarTop(m) : 0).T;
  const tipW = Math.min(288, g.width - 16);

  const toggles = (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
      <Toggle label="matrix" choices={opNames} value={op} onChange={pickOp} />
      <Toggle label="rows M" choices={rowNames} value={rows} onChange={setRows} />
    </div>
  );
  const buttons = (
    <div className="flex flex-wrap items-center gap-2">
      <button type="button" onClick={() => (done >= NT ? (setDone(0), setPlaying(true)) : setPlaying(!playing))} aria-pressed={playing} className="pressable rounded border border-border px-2.5 py-1 text-accent">
        {playing ? "pause" : "play"}
      </button>
      <button type="button" disabled={done >= NT || playing} onClick={() => setDone(Math.min(NT, done + sms))} className="pressable rounded border border-border px-2.5 py-1 text-text disabled:opacity-40">
        step one wave
      </button>
      <button type="button" onClick={() => (setDone(0), setHover(null), setPlaying(!reduced))} className="pressable rounded border border-border px-2.5 py-1 text-text">
        replay
      </button>
    </div>
  );
  const status = (
    <div className="text-muted" aria-live="polite">
      <span className="text-text">{`wave ${wave} of ${waves}`}</span>
      {` · ${read} of ${NT} tiles read by ${sms} SMs · ${size(readB)} of ${size(layerW)} of one layer's W · ${usText(readB / (peak.gbs * 1e3))} µs at ${peak.gbs} GB/s`}
    </div>
  );
  const view = (
    <div ref={setWrap} className="relative">
      <canvas
        ref={canvasRef}
        tabIndex={0}
        className="block w-full touch-manipulation"
        style={{ height: g.height }}
        role="img"
        aria-label={`The ${p.label} weight matrix, ${fmt(p.out)} by ${fmt(p.in)}, drawn in 3D as ${N} by ${N} tiles at ${m} ${plural(m, "row", "rows")}: wave ${wave} of ${waves}, ${read} of ${NT} tiles read. Arrow keys move between tiles: up and down change the W rows, left and right the columns. The bytes and ceilings are listed beside it.`}
        onPointerMove={pick}
        onPointerDown={pick}
        onPointerLeave={(e) => e.pointerType === "mouse" && setHover(null)}
        onKeyDown={onKey}
        onFocus={() => setHover((h) => h ?? 0)}
        onBlur={() => setHover(null)}
      />
      {hover !== null && tipAt && (
        <div
          role="status"
          className="pointer-events-none absolute z-10 space-y-1.5 rounded border border-border bg-surface p-2.5 shadow-sm"
          style={{
            width: tipW,
            left: Math.max(0, Math.min(tipAt[0] - tipW / 2, g.width - tipW)),
            ...(tipAt[1] > g.height / 2 ? { bottom: g.height - tipAt[1] + 8 } : { top: tipAt[1] + g.th + 8 }),
          }}
        >
          <TileDetails id={hover} p={p} m={m} done={done} />
        </div>
      )}
    </div>
  );
  const legend = (
    <div className="flex flex-wrap gap-x-4 gap-y-1 text-muted">
      {seriesVar.map((v, i) => (
        <span key={v} className="flex items-center gap-1.5">
          <span aria-hidden className="size-2 rounded-sm" style={{ background: `var(${v})` }} />
          {`SMs ${i}, ${i + 3}, ${i + 6}, …`}
        </span>
      ))}
      <span className="flex items-center gap-1.5">
        <span aria-hidden className="size-2 rounded-sm bg-accent" />
        uses under the ridge
      </span>
      <span className="flex items-center gap-1.5">
        <span aria-hidden className="size-2 rounded-sm bg-warn" />
        over the ridge
      </span>
      <span className="flex items-center gap-1.5">
        <span aria-hidden className="size-2 rounded-sm border border-border" />
        still in DRAM
      </span>
    </div>
  );
  const caption = (
    <p className="text-muted">{`The board is downsampled: each of the ${NT} tiles stands for about ${fmt(tile0.r1 - tile0.r0)} rows by ${fmt(tile0.c1 - tile0.c0)} columns of W, and which SM reads which tile is illustrative (cuBLAS picks its own). Bytes and FLOPs are computed from the shapes, the ceilings (${peak.gbs} GB/s, ${peak.tflops} TFLOP/s) and the µs per step at 1 and 16 rows are measured.`}</p>
  );
  const table = (
    <div className="space-y-2">
      <div className="text-muted">{`${p.label}: W is [${fmt(p.out)} × ${fmt(p.in)}], BF16, ${p.per > 1 ? `read once in each of ${p.per} layers` : "read once per step"}`}</div>
      <table className="w-full tabular-nums">
        <thead className="text-left text-muted">
          <tr>
            <th className="pb-1 font-normal">rows M</th>
            {rowNames.map((r) => (
              <th key={r} className={clsx("pb-1 text-right font-normal", r === rows ? "text-accent" : "text-muted")}>
                {r}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {tableRows.map((row) => (
            <tr key={row.label} className="border-t border-border">
              <td className="py-1 pr-2 text-muted">{row.label}</td>
              {costs.map((x) => (
                <td key={x.rows} className={clsx("py-1 pl-1 text-right", row.warn?.(x) ? "text-warn" : x.rows === rows ? "text-text" : "text-muted")}>
                  {row.value(x)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
  const ceilingBars = (
    <div className="space-y-2">
      <div className="text-muted">{`${p.label} at ${m} ${plural(m, "row", "rows")}, µs per step`}</div>
      <div className="space-y-1">
        {bars.map((b) => (
          <div key={b.label} className="space-y-1">
            <div className="flex justify-between text-muted">
              <span>{b.label}</span>
              <span className={clsx("tabular-nums", b.text)}>{`${usText(b.value)} µs`}</span>
            </div>
            <div className="h-2 rounded-full bg-border">
              <EasedBar fraction={b.value / longest} className={b.bar} />
            </div>
          </div>
        ))}
        {sel.us === null && <div className="text-muted">{`measured: not measured at ${m} rows`}</div>}
      </div>
      <p className={clsx(sel.k.bound === "compute" && "text-warn")}>
        {`${sel.k.bound === "memory" ? "Memory" : "Compute"} bound: ${sel.k.intensity.toFixed(1)} FLOP per byte against the ridge at ${Math.round(ridge)}, so the time is ${sel.k.bound === "memory" ? `reading ${size(sel.k.bytes)} at ${peak.gbs} GB/s` : `${(sel.k.flops / 1e9).toFixed(0)} GFLOP at ${peak.tflops} TFLOP/s`}.`}
      </p>
    </div>
  );

  if (pages) {
    return (
      <>
        {pages.view &&
          createPortal(
            <div className="space-y-3 tabular-nums text-xs">
              {toggles}
              {buttons}
              {status}
              {view}
              {legend}
              {caption}
            </div>,
            pages.view,
          )}
        {pages.table &&
          createPortal(
            <div className="space-y-5 tabular-nums text-xs">
              <Predict {...prediction} />
              {toggles}
              {table}
              {ceilingBars}
            </div>,
            pages.table,
          )}
      </>
    );
  }
  return (
    <div className="grid gap-6 tabular-nums text-xs lg:grid-cols-[minmax(0,1fr)_19rem]">
      <div className="min-w-0 space-y-3">
        {toggles}
        {buttons}
        {status}
        {view}
        {legend}
        {caption}
      </div>

      <div className="space-y-5">
        <Predict {...prediction} />
        {table}
        {ceilingBars}
      </div>
    </div>
  );
}

function TileDetails({ id, p, m, done }: { id: number; p: Projection; m: number; done: number }) {
  const t = tileShape(p, id);
  const bytes = t.weights * BF16;
  const read = readAt[id] < done;
  const sm = smOf(id);
  const wave = waveOf(id) + 1;
  return (
    <>
      <p className="text-text">{`${p.label} W rows ${fmt(t.r0)} to ${fmt(t.r1 - 1)}, columns ${fmt(t.c0)} to ${fmt(t.c1 - 1)}`}</p>
      <p className="text-muted">{`${fmt(t.r1 - t.r0)} × ${fmt(t.c1 - t.c0)} weights, ${fmt(bytes)} bytes in BF16`}</p>
      <p className={read ? "text-text" : "text-muted"}>{read ? `read from DRAM by SM ${sm} in wave ${wave} of ${waves}` : `still in DRAM; SM ${sm} reads it in wave ${wave} of ${waves}`}</p>
      <p className="border-t border-border pt-1.5 text-muted">
        {`at ${fmt(m)} ${plural(m, "row", "rows")} each weight feeds ${fmt(m)} ${plural(m, "multiply-add", "multiply-adds")}: `}
        <span className={m > ridge ? "text-warn" : "text-text"}>{`${fmt(2 * m * t.weights)} FLOPs`}</span>
        {` from these bytes, ${fmt(m)} FLOP per byte`}
      </p>
    </>
  );
}
