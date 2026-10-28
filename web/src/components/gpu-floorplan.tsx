"use client";

import { animate, motion, useMotionValue, useTransform } from "motion/react";
import { useEffect } from "react";
import { Frame, Label } from "@/components/kernels/frame";
import { useLoop } from "@/components/kernels/loop";
import { milestones } from "@/content/build-log";

export type Flow = "idle" | "bandwidth" | "weights" | "smem" | "vram" | "warps";

type Box = { x: number; y: number; width: number; height: number };

export const metric = (branch: string) => {
  const m = milestones.flatMap((s) => s.entries).find((e) => e.branch === branch)?.metric;
  return m?.kind === "value" ? m.value : "";
};
const bandwidth = metric("exercises");
const loadTime = metric("loader/safetensors-parser");
const checksum = metric("tools/checksum");

const sms = [0, 1, 2];
const warps = [0, 1, 2, 3, 4];
const smY = (i: number) => 42 + i * 94;
const regs = (i: number): Box => ({ x: 70, y: smY(i) + 6, width: 202, height: 14 });
const smem = (i: number): Box => ({ x: 32, y: smY(i) + 26, width: 240, height: 30 });
const warp = (i: number, j: number): Box => ({ x: 32 + j * 36, y: smY(i) + 62, width: 30, height: 16 });
const vram: Box = { x: 350, y: 30, width: 198, height: 258 };
const disk: Box = { x: 440, y: 336, width: 108, height: 40 };
const weightTier: Box = { x: 356, y: 36, width: 186, height: 128 };
const kvTier: Box = { x: 356, y: 184, width: 186, height: 98 };
const bx = (k: number) => 361 + (k % 6) * 30;
const wy = (k: number) => 42 + Math.floor(k / 6) * 24;
const ky = (k: number) => 190 + Math.floor(k / 6) * 24;
const weightBlocks = Array.from({ length: 30 }, (_, k) => k);
const kvBlocks = Array.from({ length: 18 }, (_, k) => k);

function Lit(box: Box) {
  return <rect {...box} rx={3} className="fill-accent/10 stroke-accent" />;
}

function Floorplan() {
  return (
    <g>
      <Label x={12} y={20}>GPU chip</Label>
      <rect x={12} y={30} width={318} height={298} rx={6} className="fill-none stroke-border" />
      {sms.map((i) => (
        <g key={i}>
          <rect x={24} y={smY(i)} width={256} height={86} rx={4} className="fill-surface stroke-border" />
          <Label x={32} y={smY(i) + 17}>{`SM ${i}`}</Label>
          <rect {...regs(i)} rx={2} className="fill-baseline/20 stroke-border" />
          <rect {...smem(i)} rx={2} className="fill-baseline/10 stroke-border" />
          {warps.map((j) => (
            <rect key={j} {...warp(i, j)} rx={2} className="fill-none stroke-border" />
          ))}
          <line x1={318} y1={smY(i) + 13} x2={350} y2={smY(i) + 13} className="stroke-border" />
        </g>
      ))}
      <Label x={76} y={59}>registers</Label>
      <Label x={40} y={87}>shared memory</Label>
      <Label x={218} y={116}>warps</Label>
      <rect x={290} y={42} width={28} height={274} rx={2} className="fill-baseline/20 stroke-border" />
      <Label x={304} y={182} anchor="middle">
        L2
      </Label>
      <Label x={350} y={20}>global VRAM</Label>
      <rect {...vram} rx={6} className="fill-none stroke-border" />
      {weightBlocks.map((k) => (
        <rect key={k} x={bx(k)} y={wy(k)} width={26} height={20} rx={2} className="fill-baseline/10 stroke-border" />
      ))}
      <Label x={361} y={180}>KV page blocks</Label>
      {kvBlocks.map((k) => (
        <rect key={k} x={bx(k)} y={ky(k)} width={26} height={20} rx={2} className="fill-none stroke-border" strokeDasharray="3 2" />
      ))}
      <line x1={494} y1={288} x2={494} y2={336} className="stroke-border" />
      <Label x={500} y={316}>PCIe</Label>
      <rect {...disk} rx={4} className="fill-surface stroke-border" />
      <Label x={494} y={360} anchor="middle">
        disk (host)
      </Label>
    </g>
  );
}

function Bandwidth() {
  const { move, reduced } = useLoop(5);
  const peak = parseFloat(bandwidth);
  const value = useMotionValue(0);
  const text = useTransform(value, (v) => `${v.toFixed(1)} GB/s`);
  useEffect(() => {
    if (reduced) {
      value.set(peak);
      return;
    }
    const run = animate(value, [0, peak, peak], { duration: 5, times: [0, 0.6, 1], ease: ["easeOut", "linear"], repeat: Infinity });
    return () => run.stop();
  }, [reduced, value, peak]);
  return (
    <g>
      <Lit {...vram} />
      {sms.map((i) => (
        <line key={i} x1={150} y1={smY(i) + 13} x2={378} y2={smY(i) + 13} className="stroke-accent/30" />
      ))}
      {sms.flatMap((i) =>
        [0, 1, 2, 3].map((j) => {
          const t = j * 0.22 + i * 0.07;
          return <motion.rect key={`${i}-${j}`} x={366} y={smY(i) + 10} width={12} height={6} rx={1} className="fill-accent" {...move(t, t + 0.2, -216, 0, false)} />;
        }),
      )}
      <motion.text x={24} y={364} className="fill-accent text-[22px] tabular-nums">
        {text}
      </motion.text>
      <Label x={24} y={386}>measured DRAM read bandwidth</Label>
    </g>
  );
}

const tensors = ["embed_tokens", "layers.0.attn.q", "layers.0.mlp.up", "layers.11.attn.o", "layers.21.mlp.down", "lm_head"];

function Weights() {
  const { move, on } = useLoop(6);
  const fillAt = (k: number) => 0.06 + k * 0.02;
  return (
    <g>
      <Lit {...disk} />
      <Lit {...weightTier} />
      <line x1={494} y1={288} x2={494} y2={336} className="stroke-accent/50" />
      {weightBlocks.map((k) => (
        <motion.rect key={k} x={bx(k)} y={wy(k)} width={26} height={20} rx={2} className="fill-accent/70" {...on(fillAt(k))} />
      ))}
      {Array.from({ length: 10 }, (_, n) => {
        const k = n * 3 + 2;
        return <motion.rect key={n} x={483} y={340} width={22} height={5} rx={1} className="fill-accent" {...move(fillAt(k) - 0.08, fillAt(k), bx(k) + 2 - 483, wy(k) + 8 - 340, false)} />;
      })}
      {tensors.map((name, k) => {
        const t = 0.06 + k * 0.1;
        const last = k === tensors.length - 1;
        return (
          <motion.g key={name} {...on(t, last ? 0.94 : t + 0.075, last)}>
            <Label x={24} y={352} className="fill-text">
              {name}
            </Label>
          </motion.g>
        );
      })}
      <Label x={24} y={376}>{`2.20 GB in ${loadTime}`}</Label>
      <motion.g {...on(0.72)}>
        <rect x={250} y={338} width={150} height={40} rx={4} className="fill-accent/10 stroke-accent" />
        <text x={325} y={363} textAnchor="middle" className="fill-accent text-[15px]">{`${checksum} ✓`}</text>
      </motion.g>
    </g>
  );
}

const ghost = [
  { t: 0.04, x: 274, dx: 130 },
  { t: 0.2, x: 404, dx: -130 },
  { t: 0.38, x: 274, dx: 130 },
  { t: 0.54, x: 404, dx: -130 },
];

function Smem() {
  const { move, on, pulse } = useLoop(6);
  const y0 = smY(0);
  return (
    <g>
      <Lit {...regs(0)} />
      <Lit {...smem(0)} />
      <line x1={272} y1={y0 + 46} x2={418} y2={y0 + 46} className="stroke-warn/40" strokeDasharray="3 3" />
      {ghost.map((g) => (
        <motion.rect key={g.t} x={g.x} y={y0 + 43} width={14} height={6} rx={1} className="fill-warn/40" {...move(g.t, g.t + 0.14, g.dx, 0, false)} />
      ))}
      <line x1={150} y1={y0 + 13} x2={414} y2={y0 + 13} className="stroke-accent/30" />
      <motion.rect x={390} y={y0 + 9} width={24} height={8} rx={1} className="fill-accent" {...move(0.04, 0.2, -236, 0, false)} />
      {[0, 1, 2].map((j) => (
        <motion.circle key={j} cx={170 + j * 24} cy={y0 + 22} r={3} className="fill-accent" {...move(0.22 + j * 0.02, 0.32 + j * 0.02, 0, 20, false)} />
      ))}
      <motion.g {...on(0.3, 0.64, false)}>
        <motion.rect {...smem(0)} rx={2} className="fill-accent/20" {...pulse(0)} />
      </motion.g>
      <motion.g {...on(0.3)}>
        <Label x={264} y={y0 + 45} anchor="end" className="fill-accent">
          Σx², variance
        </Label>
      </motion.g>
      <motion.rect x={154} y={y0 + 9} width={24} height={8} rx={1} className="fill-accent" {...move(0.66, 0.82, 236, 0, false)} />
      <Label x={24} y={356} className="fill-accent">
        fused: read once, write once
      </Label>
      <Label x={24} y={378} className="fill-warn">
        before: out to VRAM and back, twice
      </Label>
    </g>
  );
}

const otherKv = [0, 1, 2, 3, 4, 6, 8];
const blockTable = [5, 7, 9];
const kvWrites = [
  { b: 7, s: 2, t: 0.08 },
  { b: 7, s: 3, t: 0.3 },
  { b: 9, s: 0, t: 0.66 },
];
const slot = (b: number, s: number): Box => ({ x: bx(b) + 1 + s * 6, y: ky(b) + 2, width: 6, height: 16 });

function Vram() {
  const { move, on } = useLoop(7);
  const alloc = 0.52;
  const wx = 150;
  const wyTop = smY(0) + 62;
  return (
    <g>
      <Lit {...kvTier} />
      {otherKv.map((b) => (
        <rect key={b} x={bx(b)} y={ky(b)} width={26} height={20} rx={2} className="fill-baseline/40" />
      ))}
      <rect x={bx(5)} y={ky(5)} width={26} height={20} rx={2} className="fill-accent/70" />
      {[0, 1].map((s) => (
        <rect key={s} {...slot(7, s)} className="fill-accent/70" />
      ))}
      <motion.rect x={bx(9)} y={ky(9)} width={26} height={20} rx={2} className="fill-none stroke-accent" {...on(alloc)} />
      {kvWrites.map(({ b, s, t }) => (
        <g key={t}>
          <motion.rect x={bx(b) - 2} y={ky(b) - 2} width={30} height={24} rx={3} className="fill-none stroke-accent" {...on(t, t + 0.16, false)} />
          <motion.g {...move(t, t + 0.12, slot(b, s).x - wx, slot(b, s).y + 3 - wyTop, false)}>
            <rect x={wx} y={wyTop} width={6} height={4} className="fill-accent" />
            <rect x={wx} y={wyTop + 5} width={6} height={4} className="fill-accent/50" />
          </motion.g>
          <motion.rect {...slot(b, s)} className="fill-accent/70" {...on(t + 0.12)} />
        </g>
      ))}
      <Label x={24} y={356}>block table</Label>
      {blockTable.map((p, k) => {
        const cell = (
          <>
            <rect x={110 + k * 56} y={340} width={50} height={24} rx={3} className="fill-surface stroke-border" />
            <Label x={135 + k * 56} y={356} anchor="middle" className="fill-text">{`${k} → ${p}`}</Label>
          </>
        );
        return (
          <g key={k}>
            {k < 2 ? cell : <motion.g {...on(alloc)}>{cell}</motion.g>}
            {k > 0 && (
              <motion.rect
                x={110 + k * 56}
                y={340}
                width={50}
                height={24}
                rx={3}
                className="fill-accent/10 stroke-accent"
                {...(k === 1 ? on(0.06, 0.44, false) : on(0.62, 0.84, false))}
              />
            )}
          </g>
        );
      })}
      <motion.g {...on(alloc)}>
        <Label x={280} y={356} className="fill-accent">
          ← new block
        </Label>
      </motion.g>
      <Label x={24} y={386}>write_kv_kernel: K and V into the next slot</Label>
    </g>
  );
}

const chunkBlocks = [2, 5, 7, 9, 12, 16];

function Warps() {
  const { move, on, pulse } = useLoop(6);
  const y0 = smY(0);
  const step = 0.14;
  return (
    <g>
      <Lit {...regs(0)} />
      {warps.map((j) => (
        <motion.rect key={j} {...warp(0, j)} rx={2} className="fill-accent/40" {...pulse(j * 0.2)} />
      ))}
      {chunkBlocks.map((b) => (
        <rect key={b} x={bx(b)} y={ky(b)} width={26} height={20} rx={2} className="fill-accent/30" />
      ))}
      <rect x={196} y={y0 + 32} width={44} height={18} rx={2} className="fill-accent/15 stroke-accent" />
      <motion.g {...pulse(0)}>
        <Label x={266} y={y0 + 17} anchor="end" className="fill-accent">
          max m, sum l
        </Label>
      </motion.g>
      {chunkBlocks.map((b, k) => {
        const t = 0.04 + k * step;
        const last = k === chunkBlocks.length - 1;
        return (
          <g key={b}>
            <motion.rect x={bx(b) - 2} y={ky(b) - 2} width={30} height={24} rx={3} className="fill-none stroke-accent" {...on(t, t + step, false)} />
            <motion.rect x={bx(b)} y={ky(b) + 6} width={26} height={8} rx={1} className="fill-accent" {...move(t, t + 0.1, 200 - bx(b), y0 + 37 - ky(b) - 6, false)} />
            <motion.g {...on(t, last ? 0.94 : t + step - 0.025, last)}>
              <Label x={24} y={356} className="fill-text">{`chunk ${k + 1} of ${chunkBlocks.length}`}</Label>
            </motion.g>
          </g>
        );
      })}
      <Label x={24} y={378}>shared memory holds one chunk, never the context</Label>
    </g>
  );
}

const flows: Record<Flow, () => React.ReactNode> = {
  idle: () => null,
  bandwidth: Bandwidth,
  weights: Weights,
  smem: Smem,
  vram: Vram,
  warps: Warps,
};

const labels: Record<Flow, string> = {
  idle: "GPU floorplan: three SMs with registers, shared memory and warps, an L2 bar, global VRAM with KV page blocks, and the host disk",
  bandwidth: `Packets stream from global VRAM through L2 into the SMs at the measured ${bandwidth}`,
  weights: `Tensors stream from disk over PCIe into VRAM, 2.20 GB in ${loadTime}, and ${checksum} checksums match`,
  smem: "A row is read from VRAM into one SM once, its sum of squares stays in registers and shared memory, and the result is written back once; the unfused path went out to VRAM and back twice",
  vram: "write_kv_kernel looks up the block table, writes K and V into the next slot of a KV page block, and a new block is allocated when one fills",
  warps: "Warps in one SM read KV blocks one chunk at a time, keeping a running max and sum in registers while shared memory holds a single chunk",
};

export function GpuFloorplan({ flow }: { flow: Flow }) {
  const Layer = flows[flow];
  return (
    <Frame label={labels[flow]}>
      <Floorplan />
      <Layer />
    </Frame>
  );
}

const lanes = [
  { label: "global atomics", note: "every thread adds into one global counter", end: 0.9 },
  { label: "shared memory", note: "a tree inside each block, one atomic per block", end: 0.52 },
  { label: "warp shuffles", note: "lanes pass registers directly, no shared memory", end: 0.3 },
];

export function ReductionRaceFigure() {
  const { grow, on, reduced } = useLoop(6);
  const first = lanes[lanes.length - 1].end;
  return (
    <Frame label="The same vector summed with global atomics, shared memory and warp shuffles; warp shuffles finish first and atomics last. Relative speeds are illustrative">
      <Label x={20} y={28}>the same vector, summed three ways</Label>
      {lanes.map((lane, i) => {
        const y = 74 + i * 96;
        const best = i === lanes.length - 1;
        return (
          <g key={lane.label}>
            <Label x={20} y={y} className={best ? "fill-accent" : "fill-text"}>
              {lane.label}
            </Label>
            <rect x={20} y={y + 10} width={520} height={20} rx={3} className="fill-none stroke-border" />
            <motion.rect
              x={20}
              y={y + 10}
              width={reduced ? (520 * (first - 0.04)) / (lane.end - 0.04) : 520}
              height={20}
              rx={3}
              className={best ? "fill-accent" : "fill-baseline/60"}
              {...grow(0.04, lane.end)}
            />
            <Label x={20} y={y + 50}>
              {lane.note}
            </Label>
            <motion.g {...on(lane.end, 0.94, best)}>
              <Label x={540} y={y} anchor="end" className={best ? "fill-accent" : "fill-muted"}>
                {best ? "first to finish" : "done"}
              </Label>
            </motion.g>
          </g>
        );
      })}
      <Label x={20} y={380}>relative speeds are illustrative, not measured</Label>
    </Frame>
  );
}

const tile = 30;
const grid = [0, 1, 2];
const A = { x: 20, y: 66 };
const B = { x: 20, y: 226 };
const C = { x: 440, y: 140 };
const core: Box = { x: 210, y: 130, width: 150, height: 110 };

export function GemmTilesFigure() {
  const { on, move, pulse } = useLoop(6);
  const step = 0.095;
  return (
    <Frame label="BF16 tiles of A and B stream through the tensor cores and land in C tile by tile; cuBLAS is asked for the transposed product so nothing is copied">
      <Label x={A.x} y={A.y - 10}>A, BF16, row-major</Label>
      <Label x={B.x} y={B.y - 10}>B, BF16</Label>
      <Label x={C.x} y={C.y - 10}>C = A · B</Label>
      {grid.flatMap((r) =>
        grid.map((c) => (
          <g key={`${r}-${c}`}>
            <rect x={A.x + c * tile} y={A.y + r * tile} width={tile - 2} height={tile - 2} rx={2} className="fill-surface stroke-border" />
            <rect x={B.x + c * tile} y={B.y + r * tile} width={tile - 2} height={tile - 2} rx={2} className="fill-surface stroke-border" />
            <rect x={C.x + c * tile} y={C.y + r * tile} width={tile - 2} height={tile - 2} rx={2} className="fill-surface stroke-border" />
          </g>
        )),
      )}
      <line x1={112} y1={110} x2={core.x} y2={170} className="stroke-border" />
      <line x1={112} y1={270} x2={core.x} y2={200} className="stroke-border" />
      <line x1={core.x + core.width} y1={185} x2={C.x - 4} y2={185} className="stroke-border" />
      <rect {...core} rx={4} className="fill-surface stroke-border" />
      <motion.rect {...core} rx={4} className="fill-accent/10" {...pulse(0)} />
      <Label x={core.x + core.width / 2} y={core.y + core.height / 2 + 4} anchor="middle" className="fill-text">
        tensor cores
      </Label>
      {grid.flatMap((i) =>
        grid.map((j) => {
          const t = 0.04 + (i * 3 + j) * step;
          return (
            <g key={`${i}-${j}`}>
              <motion.rect x={A.x - 2} y={A.y + i * tile - 2} width={3 * tile + 2} height={tile + 2} rx={3} className="fill-none stroke-accent" {...on(t, t + step, false)} />
              <motion.rect x={B.x + j * tile - 2} y={B.y - 2} width={tile + 2} height={3 * tile + 2} rx={3} className="fill-none stroke-accent" {...on(t, t + step, false)} />
              <motion.rect x={A.x + 2 * tile} y={A.y + i * tile} width={tile - 2} height={tile - 2} rx={2} className="fill-accent/60" {...move(t, t + 0.05, core.x + 20 - A.x - 2 * tile, core.y + 14 - A.y - i * tile, false)} />
              <motion.rect x={B.x + j * tile} y={B.y + 2 * tile} width={tile - 2} height={tile - 2} rx={2} className="fill-accent/60" {...move(t, t + 0.05, core.x + 20 - B.x - j * tile, core.y + 68 - B.y - 2 * tile, false)} />
              <motion.rect x={core.x + core.width - tile - 10} y={core.y + 40} width={tile - 2} height={tile - 2} rx={2} className="fill-accent" {...move(t + 0.05, t + 0.09, C.x + j * tile - core.x - core.width + tile + 10, C.y + i * tile - core.y - 40, false)} />
              <motion.rect x={C.x + j * tile} y={C.y + i * tile} width={tile - 2} height={tile - 2} rx={2} className="fill-accent" {...on(t + 0.09)} />
            </g>
          );
        }),
      )}
      <Label x={20} y={352}>cuBLAS is column-major, so the engine asks for Cᵀ = Bᵀ · Aᵀ:</Label>
      <Label x={20} y={372}>the same row-major memory, read in place, nothing copied</Label>
    </Frame>
  );
}
