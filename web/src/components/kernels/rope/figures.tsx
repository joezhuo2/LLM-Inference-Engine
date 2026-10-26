"use client";

import { type MotionProps, motion } from "motion/react";
import { Frame, Label } from "../frame";
import { useLoop } from "../loop";

const headDim = 64;
const theta = (i: number) => Math.pow(10000, (-2 * i) / headDim);

function turnTime(seconds: number) {
  if (seconds < 90) return `${seconds.toFixed(1)} s`;
  if (seconds < 5400) return `${Math.round(seconds / 60)} min`;
  return `${Math.round(seconds / 3600)} h`;
}

const dials = [0, 1, 2, 3, 4, 8, 16, 31];
const stillPosition = 3;

export function RotateFigure() {
  const { reduced } = useLoop(1);
  const r = 34;
  return (
    <Frame label="Eight of the 32 pairs of one head drawn as 2D vectors, each turning at its own rate: pair 0 fastest, pair 31 barely at all">
      <Label x={20} y={24}>one head = 32 pairs, each a 2D vector turned by position · θᵢ</Label>
      {dials.map((i, k) => {
        const cx = 78 + (k % 4) * 134;
        const cy = 104 + Math.floor(k / 4) * 138;
        const period = (2 * Math.PI) / theta(i);
        const still = (-stillPosition * theta(i) * 180) / Math.PI;
        return (
          <g key={i} transform={`translate(${cx} ${cy})`}>
            <circle r={r} className="fill-surface stroke-border" />
            <line x1={-r} x2={r} y1={0} y2={0} className="stroke-border" strokeDasharray="2 3" />
            <line x1={0} x2={0} y1={-r} y2={r} className="stroke-border" strokeDasharray="2 3" />
            <motion.g
              initial={false}
              style={reduced ? { rotate: still } : undefined}
              animate={reduced ? undefined : { rotate: [0, -360] }}
              transition={reduced ? undefined : { duration: period, repeat: Infinity, ease: "linear" }}
            >
              <circle r={r} className="fill-transparent" />
              <line x1={0} y1={0} x2={r - 5} y2={0} className="stroke-accent" strokeWidth={2} />
              <circle cx={r - 5} cy={0} r={3.5} className="fill-accent" />
            </motion.g>
            <Label x={0} y={r + 18} anchor="middle" className="fill-text">{`pair ${i}`}</Label>
            <Label x={0} y={r + 32} anchor="middle">{`θ = ${theta(i).toPrecision(2)}`}</Label>
            <Label x={0} y={r + 46} anchor="middle">{`turn: ${turnTime(period)}`}</Label>
          </g>
        );
      })}
      <Label x={20} y={350}>{reduced ? `shown at position ${stillPosition}` : "one position per second; real rates for head_dim 64, theta 10,000"}</Label>
      <Label x={20} y={370}>pair 0 turns once every 6.3 positions, pair 31 once every 47,000</Label>
    </Frame>
  );
}

const heads = [
  ...Array.from({ length: 32 }, () => "q"),
  ...Array.from({ length: 4 }, () => "k"),
  ...Array.from({ length: 4 }, () => "v"),
];

export function LayoutFigure() {
  const { on } = useLoop(9);
  const cw = 13;
  const x0 = 20;
  const hx = 56;
  const hw = 7;
  const hy = 140;
  const pairs = headDim / 2;
  const slice = 0.78 / pairs;
  return (
    <Frame label="The fused QKV row holds 32 query, 4 key and 4 value heads; within a head, element i is paired with element i plus 32">
      <Label x={20} y={24}>one token&apos;s QKV row: 40 heads of 64 values</Label>
      {heads.map((h, i) => (
        <rect
          key={i}
          x={x0 + i * cw}
          y={40}
          width={cw - 2}
          height={18}
          rx={2}
          className={h === "q" ? "fill-accent/45" : h === "k" ? "fill-accent" : "fill-border"}
        />
      ))}
      <rect x={x0 - 1.5} y={38.5} width={cw + 1} height={21} rx={3} className="fill-none stroke-text" />
      <Label x={x0} y={76}>32 Q heads, rotated</Label>
      <Label x={x0 + 32 * cw} y={76}>4 K</Label>
      <Label x={x0 + 36 * cw} y={76}>4 V</Label>
      <Label x={x0 + 36 * cw} y={90}>untouched</Label>
      <line x1={x0} y1={60} x2={hx} y2={hy - 6} className="stroke-border" />
      <line x1={x0 + cw - 2} y1={60} x2={hx + headDim * hw} y2={hy - 6} className="stroke-border" />
      <Label x={hx} y={hy - 12}>head 0</Label>
      {Array.from({ length: headDim }, (_, i) => (
        <rect key={i} x={hx + i * hw} y={hy} width={hw - 1} height={16} className="fill-surface stroke-border" strokeWidth={0.5} />
      ))}
      {Array.from({ length: pairs }, (_, i) => {
        const a = hx + i * hw + (hw - 1) / 2;
        const b = a + pairs * hw;
        const start = 0.04 + i * slice;
        return (
          <g key={i}>
            <path d={`M ${a} ${hy + 18} Q ${(a + b) / 2} ${hy + 76} ${b} ${hy + 18}`} className="fill-none stroke-border" strokeWidth={0.75} />
            <motion.g {...on(start, start + 2.5 * slice, i === 0)}>
              <path d={`M ${a} ${hy + 18} Q ${(a + b) / 2} ${hy + 76} ${b} ${hy + 18}`} className="fill-none stroke-accent" strokeWidth={1.5} />
              <rect x={hx + i * hw} y={hy} width={hw - 1} height={16} className="fill-accent" />
              <rect x={hx + (i + pairs) * hw} y={hy} width={hw - 1} height={16} className="fill-accent" />
            </motion.g>
          </g>
        );
      })}
      <Label x={hx} y={hy + 30}>0</Label>
      <Label x={hx + pairs * hw} y={hy + 30} anchor="middle">32</Label>
      <Label x={hx + headDim * hw} y={hy + 30} anchor="end">63</Label>
      <Label x={hx} y={hy + 84}>pair i = (xᵢ, xᵢ₊₃₂), as in Hugging Face&apos;s rotate_half</Label>
      <Label x={20} y={268} className="fill-text">
        c = bf16(cos pθᵢ),  s = bf16(sin pθᵢ)
      </Label>
      <Label x={20} y={290} className="fill-text">
        x′ᵢ    = bf16( bf16(xᵢ · c) + bf16(−xᵢ₊₃₂ · s) )
      </Label>
      <Label x={20} y={312} className="fill-text">
        x′ᵢ₊₃₂ = bf16( bf16(xᵢ₊₃₂ · c) + bf16(xᵢ · s) )
      </Label>
      <Label x={20} y={350}>8 roundings to BF16 per pair, in the order PyTorch does them;</Label>
      <Label x={20} y={370}>no fused multiply-add, or the bits change</Label>
    </Frame>
  );
}

export function BitsFigure() {
  const { on, reduced } = useLoop(10);
  const settle = (dx: number): MotionProps =>
    reduced
      ? { initial: false, animate: { x: dx, opacity: 1 } }
      : {
          initial: { x: 0, opacity: 0 },
          animate: { x: [0, 0, 0, dx, dx, dx], opacity: [0, 0, 1, 1, 1, 0] },
          transition: { duration: 10, repeat: Infinity, ease: ["linear", "linear", "easeInOut", "linear", "linear"], times: [0, 0.3, 0.33, 0.44, 0.94, 1] },
        };
  const lx = 150;
  const tick = 90;
  const mid = lx + tick * 1.5;
  return (
    <Frame label="A frequency that differs in its last bit gives an angle that differs more at large positions, a sine that rounds to the next BF16 value, and a result off by many ulps after cancellation">
      <Label x={20} y={24} className="fill-text">
        θᵢ, computed twice
      </Label>
      <Label x={40} y={46}>host std::pow</Label>
      <Label x={180} y={46} className="fill-text">
        …1010 0110
      </Label>
      <motion.g {...on(0.02)}>
        <Label x={40} y={64}>device __powf</Label>
        <Label x={180} y={64} className="fill-text">
          …1010 011<tspan className="fill-warn">1</tspan>
        </Label>
        <Label x={270} y={64} className="fill-warn">
          the last bit
        </Label>
      </motion.g>

      <motion.g {...on(0.14)}>
        <Label x={20} y={102} className="fill-text">
          × position 2,047
        </Label>
        <Label x={40} y={122}>the angle carries the error 2,047 times over</Label>
      </motion.g>

      <motion.g {...on(0.26)}>
        <Label x={20} y={160} className="fill-text">
          sin, rounded to BF16
        </Label>
        <line x1={lx} y1={200} x2={lx + tick * 3} y2={200} className="stroke-border" />
        {[0, 1, 2, 3].map((k) => (
          <g key={k}>
            <line x1={lx + k * tick} y1={192} x2={lx + k * tick} y2={208} className="stroke-muted" />
          </g>
        ))}
        <line x1={mid} y1={182} x2={mid} y2={214} className="stroke-warn" strokeDasharray="3 3" />
        <Label x={mid} y={178} anchor="middle" className="fill-warn">
          rounding boundary
        </Label>
        <Label x={lx + tick} y={228} anchor="middle">
          BF16 value
        </Label>
        <Label x={lx + 2 * tick} y={228} anchor="middle">
          the next one
        </Label>
      </motion.g>
      <motion.circle cx={mid - 5} cy={200} r={4.5} className="fill-accent" {...settle(lx + tick - mid + 5)} />
      <motion.circle cx={mid + 5} cy={200} r={4.5} className="fill-warn" {...settle(lx + 2 * tick - mid - 5)} />

      <motion.g {...on(0.5)}>
        <Label x={20} y={264} className="fill-text">
          xᵢ·c − xᵢ₊₃₂·s cancels
        </Label>
        <Label x={40} y={284}>two near-equal terms leave a small result, so one ulp of s</Label>
        <Label x={40} y={300}>is many ulps of it</Label>
      </motion.g>
      <motion.g {...on(0.64)}>
        <Label x={20} y={336} className="fill-warn">
          naive with device powf: 57 of 1.4 million outputs off, up to 64 ulps
        </Label>
      </motion.g>
      <motion.g {...on(0.74)}>
        <Label x={20} y={360} className="fill-accent">
          both kernels: θ computed once on the host, passed by value → 0 off
        </Label>
      </motion.g>
      <Label x={20} y={386}>bit patterns and the number line are an illustration</Label>
    </Frame>
  );
}

function TimeBars({ rows, y, label }: { rows: { label: string; us: number; accent?: boolean }[]; y: number; label: string }) {
  const { grow } = useLoop(8);
  const x = 110;
  const full = 340;
  const max = Math.max(...rows.map((r) => r.us));
  return (
    <g>
      <Label x={20} y={y - 12}>{label}</Label>
      {rows.map((r, i) => {
        const yy = y + i * 26;
        return (
          <g key={r.label}>
            <Label x={20} y={yy + 10} className={r.accent ? "fill-accent" : "fill-muted"}>
              {r.label}
            </Label>
            <rect x={x} y={yy} width={full} height={12} rx={6} className="fill-border" />
            <motion.rect x={x} y={yy} width={(full * r.us) / max} height={12} rx={6} className={r.accent ? "fill-accent" : "fill-baseline"} {...grow(0.04, 0.04 + (0.7 * r.us) / max)} />
            <Label x={x + full + 8} y={yy + 10} className={r.accent ? "fill-accent" : "fill-muted"}>{`${r.us.toFixed(1)} µs`}</Label>
          </g>
        );
      })}
    </g>
  );
}

export function SerialFigure({ naive, draft }: { naive: number; draft: number }) {
  const { grow } = useLoop(8);
  const x0 = 20;
  const pitch = 16;
  const lane = 13;
  const lx = 110;
  return (
    <Frame label="The first draft gives each of the 36 rotated heads one thread, which walks its 32 pairs in order; 92 of the 128 threads idle, and the kernel takes longer than the naive one">
      <Label x={20} y={24}>128 threads per token: one per head, 36 heads to rotate</Label>
      {Array.from({ length: 128 }, (_, t) => (
        <rect key={t} x={x0 + (t % 32) * pitch} y={36 + Math.floor(t / 32) * pitch} width={pitch - 3} height={pitch - 3} rx={2} className={t < 36 ? "fill-accent" : "fill-border"} />
      ))}
      <Label x={20} y={118}>36 working, 92 with nothing to do</Label>
      <Label x={20} y={150}>each working thread, in order: load, sin and cos, 4 products, store</Label>
      {["thread 0", "thread 1", "thread 35"].map((name, row) => {
        const y = 164 + row * 24;
        return (
          <g key={name}>
            <Label x={20} y={y + 10}>{name}</Label>
            {Array.from({ length: 32 }, (_, p) => (
              <g key={p}>
                <rect x={lx + p * lane} y={y} width={lane - 2} height={12} rx={2} className="fill-surface stroke-border" strokeWidth={0.5} />
                <motion.rect x={lx + p * lane} y={y} width={lane - 2} height={12} rx={2} className="fill-accent/70" {...grow(0.04 + p * 0.024, 0.04 + (p + 1) * 0.024, "x", 0.9)} />
              </g>
            ))}
          </g>
        );
      })}
      <Label x={lx} y={246}>pair 0</Label>
      <Label x={lx + 32 * lane - 2} y={246} anchor="end">
        pair 31
      </Label>
      <TimeBars
        y={298}
        label="one token, µs per launch"
        rows={[
          { label: "naive", us: naive },
          { label: "draft", us: draft, accent: true },
        ]}
      />
    </Frame>
  );
}

const groups = 8;

export function ParallelFigure({ naive, fast }: { naive: number; fast: number }) {
  const { on, grow } = useLoop(9);
  const x0 = 76;
  const y0 = 46;
  const pitch = 11;
  const rowPitch = 14;
  const steps = Math.ceil(36 / groups);
  const window = 0.12;
  const first = 0.24;
  const chipX = x0 + 32 * pitch + 14;
  const chip = 22;
  const sincos = [
    { label: "naive", n: 36 * 32 },
    { label: "draft", n: 36 * 32 },
    { label: "fast", n: 32 * groups, accent: true },
  ];
  return (
    <Frame label="The fast kernel runs 32 pairs times 8 head groups; every thread computes sin and cos once, then rotates the same pair in 4 or 5 heads">
      <Label x={x0} y={30}>256 threads: 32 pairs × 8 head groups</Label>
      <Label x={chipX} y={30}>heads</Label>
      {Array.from({ length: groups }, (_, g) => {
        const y = y0 + g * rowPitch;
        const owned = Array.from({ length: steps }, (_, k) => g + k * groups).filter((h) => h < 36);
        return (
          <g key={g}>
            <Label x={20} y={y + 9}>{`group ${g}`}</Label>
            {Array.from({ length: 32 }, (_, p) => (
              <rect key={p} x={x0 + p * pitch} y={y} width={pitch - 2} height={rowPitch - 4} rx={1.5} className="fill-surface stroke-border" strokeWidth={0.5} />
            ))}
            <motion.rect x={x0} y={y} width={32 * pitch - 2} height={rowPitch - 4} rx={1.5} className="fill-baseline/60" {...on(0.04, 0.2, false)} />
            {owned.map((h, k) => {
              const s = first + k * (window + 0.01);
              return (
                <g key={h}>
                  <motion.rect x={x0} y={y} width={32 * pitch - 2} height={rowPitch - 4} rx={1.5} className="fill-accent/70" {...on(s, s + window, k === 0)} />
                  <text x={chipX + k * chip} y={y + 9} className="fill-muted text-[10px]">
                    {h}
                  </text>
                  <motion.text x={chipX + k * chip} y={y + 9} className="fill-accent text-[10px]" {...on(s, s + window, k === 0)}>
                    {h}
                  </motion.text>
                </g>
              );
            })}
          </g>
        );
      })}
      <motion.g {...on(0.04, 0.2, false)}>
        <Label x={x0} y={y0 + groups * rowPitch + 14}>
          first, every thread: cos and sin of its pair, once
        </Label>
      </motion.g>
      <motion.g {...on(first)}>
        <Label x={x0} y={y0 + groups * rowPitch + 14}>
          then the same pair in head g, g + 8, g + 16, …
        </Label>
      </motion.g>
      <Label x={20} y={214}>sin and cos per token</Label>
      {sincos.map((s, i) => {
        const y = 226 + i * 22;
        return (
          <g key={s.label}>
            <Label x={20} y={y + 10} className={s.accent ? "fill-accent" : "fill-muted"}>
              {s.label}
            </Label>
            <rect x={110} y={y} width={340} height={12} rx={6} className="fill-border" />
            <motion.rect x={110} y={y} width={(340 * s.n) / 1152} height={12} rx={6} className={s.accent ? "fill-accent" : "fill-baseline"} {...grow(0.04, 0.04 + (0.6 * s.n) / 1152)} />
            <Label x={458} y={y + 10} className={s.accent ? "fill-accent" : "fill-muted"}>
              {s.n.toLocaleString("en-US")}
            </Label>
          </g>
        );
      })}
      <TimeBars
        y={334}
        label="one token, µs per launch"
        rows={[
          { label: "naive", us: naive },
          { label: "fast", us: fast, accent: true },
        ]}
      />
    </Frame>
  );
}
