"use client";

import { motion } from "motion/react";
import { Frame, Label } from "../frame";
import { useLoop } from "../loop";

export const peakGbs = 323.6;

function bf16(x: number) {
  const f = new Float32Array([x]);
  const u = new Uint32Array(f.buffer);
  u[0] = (u[0] + 0x7fff + ((u[0] >>> 16) & 1)) & 0xffff0000;
  return f[0];
}

const silu = (g: number) => Math.fround(g / Math.fround(1 + Math.fround(Math.exp(-g))));

function TimeBars({ rows, y, label, digits = 1 }: { rows: { label: string; us: number; accent?: boolean }[]; y: number; label: string; digits?: number }) {
  const { grow } = useLoop(8);
  const x = 110;
  const full = 340;
  const max = Math.max(...rows.map((r) => r.us));
  return (
    <g>
      <Label x={20} y={y - 12}>{label}</Label>
      {rows.map((r, i) => {
        const yy = y + i * 24;
        return (
          <g key={r.label}>
            <Label x={20} y={yy + 10} className={r.accent ? "fill-accent" : "fill-muted"}>
              {r.label}
            </Label>
            <rect x={x} y={yy} width={full} height={12} rx={6} className="fill-border" />
            <motion.rect x={x} y={yy} width={(full * r.us) / max} height={12} rx={6} className={r.accent ? "fill-accent" : "fill-baseline"} {...grow(0.04, 0.04 + (0.7 * r.us) / max)} />
            <Label x={x + full + 8} y={yy + 10} className={r.accent ? "fill-accent" : "fill-muted"}>{`${r.us.toFixed(digits)} µs`}</Label>
          </g>
        );
      })}
    </g>
  );
}

const drawn = 16;

export function ZipFigure() {
  const { on, reduced } = useLoop(9);
  const cw = 16;
  const gx = 20;
  const ux = 284;
  const ox = 152;
  const rowY = 50;
  const outY = 160;
  const slice = 0.78 / drawn;
  const px = (g: number) => 50 + (g + 5) * 24;
  const py = (s: number) => 372 - (s + 0.5) * 22;
  const curve = Array.from({ length: 101 }, (_, k) => {
    const g = -5 + k / 10;
    return `${k ? "L" : "M"} ${px(g).toFixed(1)} ${py(silu(g)).toFixed(1)}`;
  }).join(" ");
  const sweep = Array.from({ length: drawn }, (_, j) => -4 + (8 * j) / (drawn - 1));
  return (
    <Frame label="The gate/up projection writes one row of 2 × 5,632 values; SwiGLU pairs value j of the gate half with value j of the up half and writes one output, silu of the gate times up">
      <Label x={20} y={20}>{`one token's gate/up row: 2 × 5,632 values, ${drawn} + ${drawn} drawn`}</Label>
      {Array.from({ length: drawn }, (_, j) => (
        <g key={j}>
          <rect x={gx + j * cw} y={rowY} width={cw - 2} height={18} rx={2} className="fill-accent/45" />
          <rect x={ux + j * cw} y={rowY} width={cw - 2} height={18} rx={2} className="fill-baseline/60" />
          <rect x={ox + j * cw} y={outY} width={cw - 2} height={18} rx={2} className="fill-surface stroke-border" strokeWidth={0.5} />
        </g>
      ))}
      <Label x={gx} y={42}>gate: the first 5,632</Label>
      <Label x={ux} y={42}>up: the second 5,632</Label>
      {Array.from({ length: drawn }, (_, j) => {
        const a = gx + j * cw + 7;
        const b = ux + j * cw + 7;
        const o = ox + j * cw + 7;
        const start = 0.04 + j * slice;
        return (
          <g key={j}>
            <motion.g {...on(start, start + 2.5 * slice, j === 0)}>
              <path d={`M ${a} 70 C ${a} 124, ${o} 104, ${o} 158`} className="fill-none stroke-accent" strokeWidth={1.25} />
              <path d={`M ${b} 70 C ${b} 124, ${o} 104, ${o} 158`} className="fill-none stroke-baseline" strokeWidth={1.25} />
              <rect x={gx + j * cw} y={rowY} width={cw - 2} height={18} rx={2} className="fill-accent" />
              <rect x={ux + j * cw} y={rowY} width={cw - 2} height={18} rx={2} className="fill-baseline" />
            </motion.g>
            <motion.rect x={ox + j * cw} y={outY} width={cw - 2} height={18} rx={2} className="fill-accent" {...on(start + slice, 0.94, true)} />
          </g>
        );
      })}
      <Label x={ox} y={194}>out: 5,632 values, the input of the down projection</Label>
      <Label x={20} y={222} className="fill-text">silu(g) = g / (1 + e⁻ᵍ)</Label>
      <line x1={px(-5)} x2={px(5)} y1={py(0)} y2={py(0)} className="stroke-border" />
      <line x1={px(0)} x2={px(0)} y1={py(-0.5)} y2={py(5)} className="stroke-border" />
      <path d={curve} className="fill-none stroke-accent" strokeWidth={1.5} />
      <Label x={px(5)} y={py(0) + 14} anchor="end">g</Label>
      <Label x={px(-5)} y={py(0) + 14}>−5</Label>
      {reduced ? (
        <circle cx={px(2)} cy={py(silu(2))} r={4} className="fill-accent" />
      ) : (
        <motion.circle
          cx={0}
          cy={0}
          r={4}
          className="fill-accent"
          initial={{ x: px(sweep[0]), y: py(silu(sweep[0])) }}
          animate={{ x: sweep.map(px), y: sweep.map((g) => py(silu(g))) }}
          transition={{ duration: 9 * 0.78, repeat: Infinity, repeatDelay: 9 * 0.22, ease: "linear" }}
        />
      )}
      <Label x={320} y={252}>a gate well below 0 → about 0:</Label>
      <Label x={320} y={268}>that value is switched off</Label>
      <Label x={320} y={292}>a large gate → about g:</Label>
      <Label x={320} y={308}>up passes through, scaled</Label>
      <Label x={320} y={340}>per value: read 4 bytes, write 2,</Label>
      <Label x={320} y={356}>a few dozen FLOPs: memory bound</Label>
      <Label x={320} y={372}>once there is enough work</Label>
    </Frame>
  );
}

export function ChainFigure() {
  const { on } = useLoop(10);
  const g = 2;
  const u = 2.25;
  const s = silu(g);
  const kept = bf16(s);
  const right = bf16(kept * u);
  const wrong = bf16(Math.fround(s * u));
  const steps = [
    { code: "g, u", note: "BF16 from the gate/up GEMM", value: `${g}, ${u}` },
    { code: "e = expf(−g)", note: "accurate expf, not __expf", value: Math.fround(Math.exp(-g)).toPrecision(8) },
    { code: "s = g / (1 + e)", note: "a true divide, in FP32", value: s.toPrecision(8) },
    { code: "s → BF16", note: "rounding 1: PyTorch's act output", value: String(kept), mark: true },
    { code: "s · u → BF16", note: "rounding 2: the output", value: String(right), mark: true },
  ];
  return (
    <Frame label="The order of operations per value, with a worked example: g = 2 and u = 2.25 give 3.953125 when silu is rounded to BF16 first, and 3.96875 when it is not">
      <Label x={20} y={24}>{`one value, worked through: g = ${g}, u = ${u}`}</Label>
      {steps.map((st, i) => {
        const y = 60 + i * 44;
        return (
          <motion.g key={st.code} {...on(0.03 + i * 0.09)}>
            <Label x={20} y={y} className={st.mark ? "fill-accent" : "fill-text"}>
              {st.code}
            </Label>
            <Label x={170} y={y}>{st.note}</Label>
            <Label x={540} y={y} anchor="end" className={st.mark ? "fill-accent" : "fill-text"}>
              {st.value}
            </Label>
            {i < steps.length - 1 && <line x1={28} x2={28} y1={y + 8} y2={y + 30} className="stroke-border" />}
          </motion.g>
        );
      })}
      <motion.g {...on(0.52)}>
        <line x1={20} x2={540} y1={276} y2={276} className="stroke-border" />
        <Label x={20} y={300} className="fill-warn">
          skip rounding 1 and multiply in FP32:
        </Label>
        <Label x={540} y={300} anchor="end" className="fill-warn">
          {String(wrong)}
        </Label>
        <Label x={20} y={318}>one BF16 step away from what Hugging Face computes</Label>
      </motion.g>
      <motion.g {...on(0.66)}>
        <Label x={20} y={350}>__expf, a reciprocal multiply or a reordered divide can move</Label>
        <Label x={20} y={366}>the last bit the same way, so every fast kernel keeps this chain</Label>
        <Label x={20} y={382}>and the tests compare every bit with the naive kernel</Label>
      </motion.g>
    </Frame>
  );
}

const ffBlocks = 22;

export function GridFigure({ naive, draft }: { naive: number; draft: number }) {
  const { on } = useLoop(10);
  const bx = 96;
  const bw = 19;
  const rows = ["row 0", "row 1", "row 2", "row 3"];
  return (
    <Frame label="Joe's first draft: a 2D grid with 22 blocks of 256 threads along each row and the row on blockIdx.y, capped at 65,535 rows; the fix flattens it into one 1D grid">
      <Label x={20} y={24}>{`the first draft: ${ffBlocks} blocks of 256 threads per row, the row on blockIdx.y`}</Label>
      {rows.map((r, i) => {
        const y = 40 + i * 20;
        return (
          <g key={r}>
            <Label x={20} y={y + 11}>{r}</Label>
            {Array.from({ length: ffBlocks }, (_, b) => (
              <g key={b}>
                <rect x={bx + b * bw} y={y} width={bw - 3} height={14} rx={2} className="fill-surface stroke-border" strokeWidth={0.5} />
                <motion.rect x={bx + b * bw} y={y} width={bw - 3} height={14} rx={2} className="fill-accent/70" {...on(0.03 + i * 0.03, 0.42, true)} />
              </g>
            ))}
          </g>
        );
      })}
      <Label x={20} y={136}>⋮</Label>
      <Label x={bx} y={136}>blockIdx.x: 0 … 21 along the row; each thread one value</Label>
      <motion.g {...on(0.2)}>
        <line x1={20} x2={540} y1={154} y2={154} className="stroke-warn" strokeDasharray="4 3" />
        <Label x={20} y={172} className="fill-warn">
          gridDim.y ≤ 65,535: a batch of 70,000 rows did not launch
        </Label>
      </motion.g>
      <motion.g {...on(0.46)}>
        <Label x={20} y={206}>the fix: one flat 1D grid over rows × 5,632, row = index / ff</Label>
        {Array.from({ length: 26 }, (_, b) => (
          <motion.rect key={b} x={20 + b * 20} y={218} width={17} height={14} rx={2} className={Math.floor((b * 256) / 5632) % 2 ? "fill-accent/45" : "fill-accent"} {...on(0.48 + b * 0.006, 0.94, true)} />
        ))}
        <Label x={20} y={248}>row and column come from the index; gridDim.x goes to 2³¹ − 1</Label>
      </motion.g>
      <TimeBars
        y={300}
        label="one token, µs per launch"
        rows={[
          { label: "naive", us: naive },
          { label: "draft", us: draft, accent: true },
        ]}
      />
    </Frame>
  );
}

export function VectorFigure({ bars }: { bars: { label: string; gbs: number; accent?: boolean }[] }) {
  const { on, grow } = useLoop(9);
  const lane = 16;
  const x0 = 20;
  return (
    <Frame label="A scalar thread loads 2 bytes of gate and 2 of up per output; a vector thread loads 16 bytes of each and writes 16 bytes, 8 outputs, with 3 memory instructions instead of 24; at 4,096 tokens the vector kernel reaches the measured bandwidth peak">
      <Label x={20} y={24}>one thread&apos;s gate load</Label>
      <Label x={20} y={50} className="fill-muted">scalar</Label>
      {Array.from({ length: 8 }, (_, k) => (
        <g key={k}>
          <rect x={90 + k * lane * 2} y={38} width={lane - 2} height={16} rx={2} className="fill-surface stroke-border" strokeWidth={0.5} />
          <motion.rect x={90 + k * lane * 2} y={38} width={lane - 2} height={16} rx={2} className="fill-baseline" {...on(0.04 + k * 0.06, 0.04 + (k + 1) * 0.06, false)} />
        </g>
      ))}
      <Label x={360} y={50}>2 B per load, 8 loads</Label>
      <Label x={20} y={86} className="fill-accent">vector</Label>
      <rect x={88} y={72} width={8 * lane * 2 - lane + 4} height={20} rx={3} className="fill-none stroke-border" />
      {Array.from({ length: 8 }, (_, k) => (
        <g key={k}>
          <rect x={90 + k * lane * 2} y={74} width={lane - 2} height={16} rx={2} className="fill-surface stroke-border" strokeWidth={0.5} />
          <motion.rect x={90 + k * lane * 2} y={74} width={lane - 2} height={16} rx={2} className="fill-accent" {...on(0.04, 0.52, true)} />
        </g>
      ))}
      <Label x={360} y={86} className="fill-accent">16 B, one uint4 load</Label>
      <Label x={x0} y={126}>memory instructions per 8 outputs</Label>
      {[
        { label: "scalar", n: 24, note: "8 gate + 8 up loads, 8 stores" },
        { label: "vector", n: 3, note: "2 loads, 1 store", accent: true },
      ].map((r, i) => {
        const y = 138 + i * 24;
        return (
          <g key={r.label}>
            <Label x={20} y={y + 10} className={r.accent ? "fill-accent" : "fill-muted"}>
              {r.label}
            </Label>
            {Array.from({ length: r.n }, (_, k) => (
              <motion.rect key={k} x={90 + k * 11} y={y} width={8} height={12} rx={1.5} className={r.accent ? "fill-accent" : "fill-baseline"} {...on(0.04 + k * 0.012, 0.94, true)} />
            ))}
            <Label x={90 + r.n * 11 + 8} y={y + 10}>{r.note}</Label>
          </g>
        );
      })}
      <Label x={20} y={212}>only when ff % 8 = 0 and both pointers are 16-byte aligned;</Label>
      <Label x={20} y={228}>anything else runs the scalar kernel, with the same math helper</Label>
      <Label x={20} y={268}>4,096 tokens: the same 119 MB ÷ each kernel&apos;s time</Label>
      {bars.map((b, i) => {
        const y = 280 + i * 24;
        const full = 340;
        const max = Math.max(peakGbs, ...bars.map((x) => x.gbs)) * 1.05;
        return (
          <g key={b.label}>
            <Label x={20} y={y + 10} className={b.accent ? "fill-accent" : "fill-muted"}>
              {b.label}
            </Label>
            <rect x={110} y={y} width={full} height={12} rx={6} className="fill-border" />
            <motion.rect x={110} y={y} width={(full * b.gbs) / max} height={12} rx={6} className={b.accent ? "fill-accent" : "fill-baseline"} {...grow(0.1, 0.1 + (0.6 * b.gbs) / max)} />
            <Label x={458} y={y + 10} className={b.accent ? "fill-accent" : "fill-muted"}>{`${Math.round(b.gbs)} GB/s`}</Label>
          </g>
        );
      })}
      {(() => {
        const max = Math.max(peakGbs, ...bars.map((x) => x.gbs)) * 1.05;
        const x = 110 + (340 * peakGbs) / max;
        return (
          <g>
            <line x1={x} x2={x} y1={274} y2={280 + bars.length * 24 - 8} className="stroke-text" strokeDasharray="3 3" />
            <Label x={x} y={280 + bars.length * 24 + 8} anchor="middle" className="fill-text">{`measured peak ${peakGbs}`}</Label>
          </g>
        );
      })()}
    </Frame>
  );
}

const sms = 26;

export function SmFigure({ naive, vector, dispatch }: { naive: number; vector: number; dispatch: number }) {
  const { on, grow } = useLoop(12);
  const tile = 36;
  const x0 = 46;
  const modes = [
    { blocks: 3, chains: 8, from: 0.03, to: 0.47, title: "vector kernel, one token: 704 vectors = 3 blocks", note: "23 SMs idle; every thread runs 8 exp and divide chains", warn: true },
    { blocks: 22, chains: 1, from: 0.5, to: 0.94, title: "scalar kernel, one token: 5,632 values = 22 blocks", note: "one chain per thread, 22 SMs at work" },
  ];
  return (
    <Frame label="At one token the vector kernel has 3 blocks for a GPU with 26 SMs and 8 dependent chains per thread; the scalar kernel has 22 blocks and one chain per thread, so below 5 tokens the kernel takes the scalar path">
      {modes.map((m, mi) => (
        <motion.g key={mi} {...on(m.from, m.to, mi === 1)}>
          <Label x={20} y={24} className="fill-text">
            {m.title}
          </Label>
          <Label x={20} y={150} className={m.warn ? "fill-warn" : "fill-accent"}>
            {m.note}
          </Label>
        </motion.g>
      ))}
      {Array.from({ length: sms }, (_, s) => {
        const x = x0 + (s % 13) * tile;
        const y = 42 + Math.floor(s / 13) * 48;
        return (
          <g key={s}>
            <rect x={x} y={y} width={tile - 6} height={40} rx={3} className="fill-surface stroke-border" strokeWidth={0.75} />
            {modes.map((m, mi) =>
              s < m.blocks ? (
                <motion.g key={mi} {...on(m.from, m.to, mi === 1)}>
                  <rect x={x} y={y} width={tile - 6} height={40} rx={3} className="fill-accent/25 stroke-accent" strokeWidth={0.75} />
                  {Array.from({ length: m.chains }, (_, c) => (
                    <motion.rect key={c} x={x + 4} y={y + 4 + c * 4} width={tile - 14} height={3} rx={1} className="fill-accent" {...grow(m.from + 0.02 + (c * (m.to - m.from - 0.1)) / 8, m.from + 0.02 + ((c + 1) * (m.to - m.from - 0.1)) / 8)} />
                  ))}
                </motion.g>
              ) : null,
            )}
          </g>
        );
      })}
      <Label x={20} y={20 + 42 + 8}>SM</Label>
      <Label x={20} y={188}>vector path only from 26 SMs × 256 / 2 = 3,328 vectors:</Label>
      <Label x={20} y={204}>4 tokens are 2,816 (scalar), 5 tokens 3,520 (vector)</Label>
      <TimeBars
        y={260}
        label="one token, µs per launch, same session"
        digits={2}
        rows={[
          { label: "naive", us: naive },
          { label: "vector", us: vector },
          { label: "dispatch", us: dispatch, accent: true },
        ]}
      />
    </Frame>
  );
}
