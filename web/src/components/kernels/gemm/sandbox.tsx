"use client";

import NumberFlow from "@number-flow/react";
import clsx from "clsx";
import { motion } from "motion/react";
import { useEffect, useState } from "react";
import { ceilings, hardware } from "@/lib/data";
import { usePrefersReducedMotion } from "@/lib/reduced-motion";
import { EasedBar } from "../eased-bar";
import { Frame, Label } from "../frame";
import { Toggle } from "../toggle";
import { blockLoadBytes, kSteps, rowChoices, type Shape, tileSizes, tiling } from "./tiling";

const peak = ceilings();
const sharedLimit = hardware().shared_kb_per_sm * 1024;
const sms = hardware().sms;

const ops: { name: string; shape: Shape }[] = [
  { name: "qkv", shape: { out: 2560, in: 2048 } },
  { name: "o_proj", shape: { out: 2048, in: 2048 } },
  { name: "gate/up", shape: { out: 11264, in: 2048 } },
  { name: "down", shape: { out: 2048, in: 5632 } },
];
const opNames = ops.map((o) => o.name);
const rowNames = rowChoices.map(String);
const bkNames = kSteps.map(String);
const start = { op: "gate/up", rows: "16", tile: 1, bk: "32" };

const fmt = (v: number) => v.toLocaleString("en-US");
const mb = (b: number) => b / 1e6;
const kb = (b: number) => b / 1024;
const px = 0.6;
const stripW = 520;
const ease = "cubic-bezier(0.65, 0, 0.35, 1)";

function Readout({ label, children, warn }: { label: string; children: React.ReactNode; warn?: boolean }) {
  return (
    <tr className="border-t border-border">
      <td className="py-1 text-muted">{label}</td>
      <td className={clsx("py-1 text-right", warn ? "text-warn" : "text-text")}>{children}</td>
    </tr>
  );
}

function Num({ value, digits = 0, suffix }: { value: number; digits?: number; suffix?: string }) {
  return <NumberFlow value={value} suffix={suffix} format={{ minimumFractionDigits: digits, maximumFractionDigits: digits }} />;
}

function Button({ children, ...props }: React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button type="button" {...props} className={clsx("pressable rounded border border-border px-2.5 py-1 disabled:opacity-40", props.className ?? "text-text")}>
      {children}
    </button>
  );
}

export function TilingSandbox() {
  const reduced = usePrefersReducedMotion();
  const [op, setOp] = useState(start.op);
  const [rows, setRows] = useState(start.rows);
  const [tileIdx, setTileIdx] = useState(start.tile);
  const [bk, setBk] = useState(start.bk);
  const [done, setDone] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [guess, setGuess] = useState<"yes" | "no" | null>(null);

  const shape = ops[opNames.indexOf(op)].shape;
  const m = Number(rows);
  const tile = tileSizes[tileIdx];
  const t = tiling(shape, m, tile, Number(bk), peak);
  const stride = Math.max(1, Math.round(t.kIters / 24));
  const live = Math.min(tile, m);
  const p = done / t.kIters;
  const shared = t.sharedBytes / sharedLimit;
  const over = shared > 1;

  const change = (fn: () => void) => {
    fn();
    setDone(0);
    setPlaying(false);
  };

  useEffect(() => {
    if (!playing) return;
    const timer = setInterval(() => {
      const next = Math.min(t.kIters, done + stride);
      setDone(next);
      if (next >= t.kIters) setPlaying(false);
    }, 700);
    return () => clearInterval(timer);
  }, [playing, done, stride, t.kIters]);

  const reset = () => {
    setOp(start.op);
    setRows(start.rows);
    setTileIdx(start.tile);
    setBk(start.bk);
    setDone(0);
    setPlaying(false);
    setGuess(null);
  };

  const one = tiling(shape, 1, 16, Number(bk), peak);
  const wide = tiling(shape, 1, 128, Number(bk), peak);
  const slide = (extra: string) => ({ transition: reduced ? "none" : `${extra} 600ms ${ease}` });
  const box = { transformBox: "fill-box" as const, transformOrigin: "left center" };
  const chunk = stripW / t.kIters;
  const tileH = tile * px;
  const tileW = Math.max(8, Number(bk) * px);
  const accumClip = `inset(${((1 - p) * 100).toFixed(2)}% 0 0 0)`;

  return (
    <div className="space-y-6 font-mono text-xs">
      <div className="space-y-2 rounded border border-border p-3">
        <div className="text-text">predict first: at 1 row (M = 1), does a bigger output tile cut the bytes read from DRAM?</div>
        <div className="flex flex-wrap items-center gap-2" role="group" aria-label="your prediction">
          {(["yes", "no"] as const).map((g) => (
            <Button key={g} aria-pressed={guess === g} onClick={() => setGuess(g)} className={guess === g ? "bg-accent-dim text-accent" : "text-text"}>
              {g === "yes" ? "yes, it helps" : "no, it does not"}
            </Button>
          ))}
        </div>
        <div role="status" aria-live="polite" className="max-w-2xl text-muted">
          {guess &&
            `${guess === "no" ? "Right." : "Not quite."} For ${op} at M = 1, tile 16 loads ${mb(one.loadBytes).toFixed(1)} MB and tile 128 loads ${mb(wide.loadBytes).toFixed(1)} MB, against ${mb(one.floorBytes).toFixed(1)} MB for the weights and x read once. There is one row of x to pair with each weight, so a weight is used once however big the tile, and the 127 extra tile rows are padding. Now set rows to 512 and drag the tile again.`}
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_19rem]">
        <div className="space-y-3">
          <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
            <Toggle label="matrix" choices={opNames} value={op} onChange={(v) => change(() => setOp(v))} />
            <Toggle label="rows M" choices={rowNames} value={rows} onChange={(v) => change(() => setRows(v))} />
            <Toggle label="k step" choices={bkNames} value={bk} onChange={(v) => change(() => setBk(v))} />
          </div>
          <label className="flex items-center gap-3">
            <span className="w-14 shrink-0 text-muted">tile</span>
            <input
              type="range"
              min={0}
              max={tileSizes.length - 1}
              step={1}
              value={tileIdx}
              onChange={(e) => change(() => setTileIdx(Number(e.target.value)))}
              aria-valuetext={`${tile} by ${tile} outputs`}
              className="min-w-40 flex-1 accent-[var(--accent)]"
            />
            <span className="w-20 shrink-0 text-text tabular-nums">{`${tile} × ${tile}`}</span>
          </label>
          <label className="flex items-center gap-3">
            <span className="w-14 shrink-0 text-muted">k loop</span>
            <input
              type="range"
              min={0}
              max={t.kIters}
              step={1}
              value={done}
              onChange={(e) => {
                setPlaying(false);
                setDone(Number(e.target.value));
              }}
              aria-valuetext={`${done} of ${t.kIters} k steps`}
              className="min-w-40 flex-1 accent-[var(--accent)]"
            />
            <span className="w-20 shrink-0 text-text tabular-nums">{`${done} / ${t.kIters}`}</span>
          </label>
          <div className="flex flex-wrap gap-2">
            <Button onClick={() => (done >= t.kIters ? (setDone(0), setPlaying(true)) : setPlaying(!playing))} aria-pressed={playing} className="text-accent">
              {playing ? "pause" : "play"}
            </Button>
            <Button disabled={done >= t.kIters} onClick={() => (setPlaying(false), setDone(done + 1))}>
              step
            </Button>
            <Button onClick={reset}>reset</Button>
          </div>
          <div className="aspect-[14/10] w-full max-w-2xl">
            <Frame label={`One thread block computing a ${tile} by ${tile} output tile of ${op} at ${m} rows: k step ${done} of ${t.kIters} loaded, ${fmt(Math.round(blockLoadBytes(m, tile, Number(bk), done, shape.in) / 1024))} KB brought in from DRAM so far`}>
              <Label x={20} y={24}>{`A panel: ${live} row${live > 1 ? "s" : ""} of x, K = ${fmt(shape.in)} wide`}</Label>
              <rect x={20} y={32} width={stripW} height={20} rx={3} className="fill-border" />
              <rect x={20} y={32} width={stripW} height={20} rx={3} className="fill-accent-dim" style={{ ...box, ...slide("transform"), transform: `scaleX(${p})` }} />
              <rect x={20} y={32} width={Math.max(3, chunk)} height={20} className="fill-accent" style={{ ...slide("transform"), transform: `translateX(${Math.max(0, done - 1) * chunk}px)`, opacity: done > 0 ? 1 : 0 }} />
              <Label x={20} y={76}>{`B panel: ${tile} rows of W, K = ${fmt(shape.in)} wide`}</Label>
              <rect x={20} y={84} width={stripW} height={20} rx={3} className="fill-border" />
              <rect x={20} y={84} width={stripW} height={20} rx={3} className="fill-accent-dim" style={{ ...box, ...slide("transform"), transform: `scaleX(${p})` }} />
              <rect x={20} y={84} width={Math.max(3, chunk)} height={20} className="fill-accent" style={{ ...slide("transform"), transform: `translateX(${Math.max(0, done - 1) * chunk}px)`, opacity: done > 0 ? 1 : 0 }} />
              <rect x={20} y={128} width={340} height={150} rx={4} fill="none" strokeDasharray="4 3" className={over ? "stroke-warn" : "stroke-border"} />
              <Label x={30} y={146} className={over ? "fill-warn" : "fill-muted"}>{`shared memory: ${kb(t.sharedBytes)} KB of ${kb(sharedLimit)} KB, 2 stages`}</Label>
              <rect x={40} y={158} width={tileW} height={tileH} className="stroke-border" fill="none" strokeDasharray="3 2" />
              <motion.rect key={`a${done}`} x={40} y={158} width={tileW} height={Math.max(2, live * px)} className="fill-accent" initial={reduced || done === 0 ? false : { opacity: 0.3 }} animate={{ opacity: done === 0 ? 0.25 : 1 }} transition={{ duration: reduced ? 0 : 0.6, ease: "easeInOut" }} />
              <Label x={40} y={158 + tileH + 16}>{`A ${live} × ${bk}`}</Label>
              <rect x={200} y={158} width={tileW} height={tileH} className="stroke-border" fill="none" strokeDasharray="3 2" />
              <motion.rect key={`b${done}`} x={200} y={158} width={tileW} height={tileH} className="fill-accent" initial={reduced || done === 0 ? false : { opacity: 0.3 }} animate={{ opacity: done === 0 ? 0.25 : 1 }} transition={{ duration: reduced ? 0 : 0.6, ease: "easeInOut" }} />
              <Label x={200} y={158 + tileH + 16}>{`B ${tile} × ${bk}`}</Label>
              <Label x={374} y={146}>{`output tile ${tile} × ${tile}`}</Label>
              <rect x={374} y={158} width={tileH} height={tileH} className="stroke-border" fill="none" strokeDasharray="3 2" />
              <rect x={374} y={158} width={tileH} height={Math.max(2, live * px)} className="fill-border" />
              <rect x={374} y={158} width={tileH} height={Math.max(2, live * px)} className="fill-accent" style={{ clipPath: accumClip, transition: reduced ? "none" : `clip-path 600ms ${ease}` }} />
              <Label x={374} y={158 + tileH + 16}>accumulators, in registers</Label>
              <Label x={20} y={304} className="fill-text">{`k step ${done} of ${t.kIters}: ${fmt(Math.round(kb(blockLoadBytes(m, tile, Number(bk), done, shape.in))))} KB loaded by this block`}</Label>
              <Label x={20} y={324}>{`each loaded x value feeds ${tile} outputs, each W value feeds ${live}`}</Label>
              <Label x={20} y={344}>{`${fmt(t.blocks)} blocks like this one, on ${sms} SMs`}</Label>
              <Label x={20} y={372}>conceptual model of tiling, not the kernel cuBLAS picks</Label>
            </Frame>
          </div>
        </div>

        <div className="space-y-5">
          <div className="space-y-2">
            <div className="text-muted">DRAM loads, computed</div>
            <div className="space-y-1">
              <div className="flex justify-between text-muted">
                <span>naive</span>
                <span className="text-text tabular-nums"><Num value={mb(t.naiveBytes)} suffix=" MB" /></span>
              </div>
              <div className="h-2 rounded-full bg-border"><EasedBar fraction={1} className="bg-baseline" /></div>
              <div className="flex justify-between text-muted">
                <span>tiled</span>
                <span className="text-accent tabular-nums"><Num value={mb(t.loadBytes)} suffix=" MB" digits={t.loadBytes < 1e8 ? 1 : 0} /></span>
              </div>
              <div className="h-2 rounded-full bg-border"><EasedBar fraction={t.loadBytes / t.naiveBytes} className="bg-accent" /></div>
              <div className="flex justify-between text-muted">
                <span>{`shared memory of ${kb(sharedLimit)} KB`}</span>
                <span className={clsx("tabular-nums", over ? "text-warn" : "text-text")}><Num value={kb(t.sharedBytes)} suffix=" KB" /></span>
              </div>
              <div className="h-2 rounded-full bg-border"><EasedBar fraction={shared} className={over ? "bg-warn" : "bg-series-1"} /></div>
            </div>
          </div>
          <table className="w-full tabular-nums">
            <tbody>
              <Readout label="output tiles (blocks)"><Num value={t.blocks} /></Readout>
              <Readout label="tile loads, all blocks"><Num value={t.tileLoads} /></Readout>
              <Readout label="reuse, naive / tiled"><Num value={t.reuse} digits={1} suffix="×" /></Readout>
              <Readout label="FLOP per byte, naive"><Num value={t.naiveIntensity} digits={2} /></Readout>
              <Readout label="FLOP per byte, tiled"><Num value={t.intensity} digits={1} /></Readout>
              <Readout label="padded work" warn={t.padWaste > 0.5}><Num value={t.padWaste * 100} suffix="%" /></Readout>
              <Readout label="memory time, at the roof"><Num value={t.memUs} digits={1} suffix=" µs" /></Readout>
              <Readout label="compute time, at the roof"><Num value={t.computeUs} digits={1} suffix=" µs" /></Readout>
            </tbody>
          </table>
          <p className={clsx(over && "text-warn")}>
            {over
              ? `This tile does not fit: ${kb(t.sharedBytes)} KB of shared memory against ${kb(sharedLimit)} KB per SM. Lower the tile or the k step.`
              : `${t.bound}-bound: ${t.intensity.toFixed(1)} FLOP per byte against the ridge at ${t.ridge.toFixed(0)}, so the slower of the two ceilings is ${t.bound === "memory" ? "bandwidth" : "the tensor cores"}.`}
          </p>
          <p className="text-muted">
            {`Computed from the tile sizes: every tile load counted as DRAM traffic, no L2 cache, 2 stages, BF16. Ceilings are the measured ${peak.gbs} GB/s and ${peak.tflops} TFLOP/s, ${hardware().shared_kb_per_sm} KB shared memory per SM from docs/hardware.md.`}
          </p>
        </div>
      </div>
    </div>
  );
}
