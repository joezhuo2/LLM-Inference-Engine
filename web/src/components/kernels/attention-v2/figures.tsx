"use client";

import { motion } from "motion/react";
import { Frame, Label } from "../frame";
import { useLoop } from "../loop";

export function ThreePassFigure({ us }: { us: number }) {
  const { on, grow } = useLoop(7);
  const cells = 32;
  const cw = 16.5;
  const threads = 16;
  const tw = 33;
  const passes = [
    { text: "1  a score for every token into shared memory", from: 0.02, to: 0.4, busy: threads },
    { text: "2  max over all scores, then exp and their sum", from: 0.42, to: 0.62, busy: threads },
    { text: "3  probabilities × values: one thread per dim", from: 0.64, to: 0.92, busy: 4 },
  ];
  return (
    <Frame label="v1 makes three passes over the context: every score into shared memory, a softmax over all of them, then the values with a quarter of the threads">
      <Label x={20} y={24}>one thread block = one row × one query head, 256 threads</Label>
      <Label x={20} y={50}>shared memory: one float per token of the context</Label>
      {Array.from({ length: cells }, (_, i) => (
        <g key={i}>
          <rect x={20 + i * cw} y={58} width={cw - 2} height={22} rx={2} className="fill-surface stroke-border" />
          <motion.rect x={20 + i * cw} y={58} width={cw - 2} height={22} rx={2} className="fill-accent/60" {...grow(0.02 + (i * 0.38) / cells, 0.02 + ((i + 1) * 0.38) / cells, "y")} />
        </g>
      ))}
      <Label x={20} y={98}>2,048 tokens, one cell = 64 scores</Label>
      {passes.map((p, k) => (
        <g key={p.text}>
          <Label x={20} y={140 + k * 28}>{p.text}</Label>
          <motion.text x={20} y={140 + k * 28} className="fill-accent text-[11px]" {...on(p.from, p.to, false)}>
            {p.text}
          </motion.text>
        </g>
      ))}
      <Label x={20} y={244}>threads at work (one square = 16 threads)</Label>
      {passes.map((p, k) => (
        <motion.g key={k} {...on(p.from, p.to, k === 2)}>
          {Array.from({ length: threads }, (_, t) => (
            <rect key={t} x={20 + t * tw} y={256} width={tw - 4} height={18} rx={2} className={t < p.busy ? "fill-accent" : "fill-border"} />
          ))}
        </motion.g>
      ))}
      <motion.g {...on(0.64, 0.92, true)}>
        <Label x={20} y={296} className="fill-warn">
          pass 3: 64 of 256 threads busy, the other 192 wait
        </Label>
      </motion.g>
      <Label x={20} y={336}>each thread also computes a whole 64-dim dot product alone,</Label>
      <Label x={20} y={354}>so neighbouring threads read keys 128 bytes apart</Label>
      <Label x={20} y={384} className="fill-text">{`one row at 2,048 tokens: ${us} µs`}</Label>
    </Frame>
  );
}

export type SharedBytes = { context: number; v1?: number; v2: number };

export function SharedMemoryFigure({ points, cap, limit }: { points: SharedBytes[]; cap: number; limit: number }) {
  const { on, move } = useLoop(8);
  const x0 = 70;
  const x1 = 520;
  const y0 = 300;
  const y1 = 60;
  const maxCtx = Math.max(...points.map((p) => p.context));
  const maxBytes = limit;
  const X = (c: number) => x0 + ((x1 - x0) * c) / maxCtx;
  const Y = (b: number) => y0 - ((y0 - y1) * b) / maxBytes;
  const v1 = points.filter((p) => p.v1 !== undefined);
  const capPoint = v1[v1.length - 1];
  return (
    <Frame label={`v1's shared memory grows with the context and the host code rejects anything past ${cap} tokens; v2 uses the same ${points[0].v2} bytes at every length`}>
      <Label x={20} y={24} className="fill-text">
        shared memory per thread block, bytes
      </Label>
      <line x1={x0} y1={y0} x2={x1} y2={y0} className="stroke-border" />
      <line x1={x0} y1={y0} x2={x0} y2={y1 - 10} className="stroke-border" />
      <line x1={x0} y1={Y(limit)} x2={x1} y2={Y(limit)} className="stroke-warn" strokeDasharray="4 3" />
      <Label x={x1} y={Y(limit) - 6} anchor="end" className="fill-warn">{`${(limit / 1024).toFixed(0)} KB default limit per block`}</Label>
      {[0, cap, maxCtx].map((c) => (
        <g key={c}>
          <line x1={X(c)} y1={y0} x2={X(c)} y2={y0 + 5} className="stroke-border" />
          <Label x={X(c)} y={y0 + 18} anchor="middle">
            {c.toLocaleString("en-US")}
          </Label>
        </g>
      ))}
      <Label x={x1} y={y0 + 36} anchor="end">
        context, tokens
      </Label>
      <motion.polyline
        points={v1.map((p) => `${X(p.context)},${Y(p.v1 ?? 0)}`).join(" ")}
        className="fill-none stroke-baseline"
        strokeWidth={2}
        {...on(0.04)}
      />
      <motion.line x1={X(cap)} y1={y1 + 40} x2={X(cap)} y2={y0} className="stroke-warn" strokeWidth={2} {...on(0.3)} />
      <motion.g {...on(0.34)}>
        <Label x={X(cap) + 8} y={y1 + 60} className="fill-warn">{`v1 stops here: kMaxContext = ${cap.toLocaleString("en-US")}`}</Label>
        <Label x={X(cap) + 8} y={y1 + 78}>{`${(capPoint.v1 ?? 0).toLocaleString("en-US")} bytes, far below the limit`}</Label>
      </motion.g>
      <motion.polyline
        points={points.map((p) => `${X(p.context)},${Y(p.v2)}`).join(" ")}
        className="fill-none stroke-accent"
        strokeWidth={2}
        {...on(0.55)}
      />
      <motion.g {...on(0.6)}>
        <Label x={x1} y={Y(points[0].v2) - 8} anchor="end" className="fill-accent">{`v2: ${points[0].v2.toLocaleString("en-US")} bytes at any length`}</Label>
      </motion.g>
      <motion.circle cx={X(0)} cy={Y(points[0].v2)} r={4} className="fill-accent" {...move(0.62, 0.9, X(maxCtx) - X(0), 0, true, 0.94)} />
      {v1.map((p) => (
        <motion.circle key={p.context} cx={X(p.context)} cy={Y(p.v1 ?? 0)} r={3} className="fill-baseline" {...on(0.04)} />
      ))}
      <Label x={20} y={372}>the cap is a check in the host code, sized for TinyLlama&apos;s 2,048 positions</Label>
    </Frame>
  );
}

const chunkScores = [0.3, 0.45, 0.2, 0.5, 0.35, 0.4, 0.25, 0.38, 0.55, 0.7, 0.4, 0.62, 0.5, 0.3, 0.66, 0.45, 0.35, 0.5, 0.6, 0.42, 0.3, 0.55, 0.48, 0.36, 0.82, 0.6, 0.7, 0.5, 0.65, 0.4, 0.58, 0.45];

const chunkMaxes = Array.from({ length: 4 }, (_, c) => ({
  prev: c === 0 ? 0 : Math.max(...chunkScores.slice(0, c * 8)),
  now: Math.max(...chunkScores.slice(0, (c + 1) * 8)),
}));

export function OnlineSoftmaxFigure() {
  const { on } = useLoop(7);
  const chunks = 4;
  const per = 8;
  const cw = 16.5;
  const slice = 0.92 / chunks;
  const base = 170;
  const h = 110;
  return (
    <Frame label="v2 reads the context one 256-token chunk at a time, keeping a running maximum and rescaling its sums when the maximum rises">
      <Label x={20} y={24}>chunks of 256 tokens, one after another (one bar = 32 tokens)</Label>
      {chunkScores.map((s, i) => {
        const c = Math.floor(i / per);
        return (
          <motion.rect
            key={i}
            x={20 + i * cw}
            y={base - s * h}
            width={cw - 3}
            height={s * h}
            rx={1}
            className="fill-baseline"
            {...on(0.02 + c * slice, 0.94)}
          />
        );
      })}
      <line x1={18} y1={base} x2={20 + 32 * cw} y2={base} className="stroke-border" />
      {chunkMaxes.map((m, c) => {
        const t = 0.02 + c * slice;
        const rose = c > 0 && m.now > m.prev;
        return (
          <g key={c}>
            <motion.rect x={18 + c * per * cw} y={40} width={per * cw} height={base - 36} rx={3} className="fill-accent/10 stroke-accent" {...on(t, t + slice, false)} />
            <motion.line
              x1={18}
              x2={20 + (c + 1) * per * cw}
              y1={base - m.now * h}
              y2={base - m.now * h}
              className="stroke-accent"
              strokeDasharray="4 3"
              {...on(t + 0.03, c === chunks - 1 ? 0.94 : t + slice, c === chunks - 1)}
            />
            {rose && (
              <motion.text x={20} y={204} className="fill-warn text-[11px]" {...on(t + 0.03, t + slice - 0.02, false)}>
                max rose: scale the old sum and acc by exp(m_old − m_new)
              </motion.text>
            )}
          </g>
        );
      })}
      <Label x={20 + 32 * cw} y={36} anchor="end" className="fill-accent">
        running max m
      </Label>
      <rect x={20} y={232} width={250} height={84} rx={3} className="fill-none stroke-accent" />
      <Label x={32} y={254} className="fill-text">
        kept between chunks
      </Label>
      <Label x={32} y={276}>m, running max</Label>
      <Label x={32} y={294}>l, running sum of exp</Label>
      <Label x={32} y={312}>acc[64], running Σ p · v</Label>
      <rect x={290} y={232} width={250} height={84} rx={3} className="fill-none stroke-border" />
      <Label x={302} y={254} className="fill-text">
        shared memory
      </Label>
      <Label x={302} y={276}>one chunk of 256 scores</Label>
      <Label x={302} y={294}>the same at 128 or 8,192 tokens</Label>
      <Label x={20} y={360}>one pass over the keys and values; nothing grows with the context</Label>
    </Frame>
  );
}

export function LanesFigure() {
  const { on } = useLoop(6);
  const cw = 26;
  const x0 = 40;
  const y0 = 64;
  const rowH = 18;
  const steps = 8;
  const slice = 0.56 / steps;
  const levels = [
    { y: 270, n: 8, t: 0.62 },
    { y: 302, n: 4, t: 0.7 },
    { y: 334, n: 2, t: 0.78 },
    { y: 366, n: 1, t: 0.86 },
  ];
  const cx = (n: number, j: number) => x0 + ((j + 0.5) * (8 * cw)) / n;
  return (
    <Frame label="Eight lanes share one key: each step they read eight neighbouring dimensions together, then three shuffles add their partial dot products">
      <Label x={x0} y={24}>one key, 64 dims, read by a group of 8 lanes</Label>
      {Array.from({ length: 8 }, (_, j) => (
        <Label key={j} x={x0 + j * cw + cw / 2} y={52} anchor="middle">
          {String(j)}
        </Label>
      ))}
      <Label x={x0 + 8 * cw + 10} y={52}>lane</Label>
      {Array.from({ length: steps }, (_, s) => (
        <g key={s}>
          <Label x={x0 - 8} y={y0 + s * rowH + 12} anchor="end">
            {String(s * 8)}
          </Label>
          {Array.from({ length: 8 }, (_, j) => (
            <g key={j}>
              <rect x={x0 + j * cw} y={y0 + s * rowH} width={cw - 3} height={rowH - 3} rx={2} className="fill-surface stroke-border" />
              <motion.rect x={x0 + j * cw} y={y0 + s * rowH} width={cw - 3} height={rowH - 3} rx={2} className="fill-accent/60" {...on(0.02 + s * slice, 0.94)} />
            </g>
          ))}
          <motion.text x={x0 + 8 * cw + 10} y={y0 + s * rowH + 12} className="fill-accent text-[11px]" {...on(0.02 + s * slice, 0.02 + (s + 1) * slice, false)}>
            {`dims ${s * 8}–${s * 8 + 7}: 16 bytes in one go`}
          </motion.text>
        </g>
      ))}
      <Label x={x0} y={252}>partial dot products, added by shuffles (xor 4, 2, 1)</Label>
      {levels.map((lv, li) => (
        <g key={lv.y}>
          {Array.from({ length: lv.n }, (_, j) => (
            <g key={j}>
              {li > 0 &&
                [0, 1].map((k) => (
                  <motion.line
                    key={k}
                    x1={cx(levels[li - 1].n, j * 2 + k)}
                    y1={levels[li - 1].y}
                    x2={cx(lv.n, j)}
                    y2={lv.y}
                    className="stroke-accent"
                    {...on(lv.t, 0.94)}
                  />
                ))}
              <motion.circle cx={cx(lv.n, j)} cy={lv.y} r={li === 3 ? 6 : 4} className={li === 3 ? "fill-accent" : "fill-baseline"} {...on(lv.t, 0.94)} />
            </g>
          ))}
        </g>
      ))}
      <Label x={x0 + 4 * cw + 12} y={370} className="fill-text">
        one score
      </Label>
      <Label x={300} y={296} className="fill-text">
        a warp = 4 groups = 4 keys at once
      </Label>
      <Label x={300} y={314}>a 256-thread block: 32 keys at once</Label>
      <Label x={300} y={346}>v1: one thread walks all 64</Label>
      <Label x={300} y={364}>dims of its key on its own</Label>
    </Frame>
  );
}

export type Pair = { label: string; v1: number; v2: number; unit: string; lowerIsBetter?: boolean };

function PairBars({ pairs, y, grow }: { pairs: Pair[]; y: number; grow: ReturnType<typeof useLoop>["grow"] }) {
  return (
    <g>
      {pairs.map((p, i) => {
        const max = Math.max(p.v1, p.v2);
        const yy = y + i * 62;
        return (
          <g key={p.label}>
            <Label x={20} y={yy} className="fill-text">
              {p.label}
            </Label>
            <Label x={20} y={yy + 20}>v1</Label>
            <rect x={60} y={yy + 10} width={360} height={12} rx={2} className="fill-border" />
            <motion.rect x={60} y={yy + 10} width={(360 * p.v1) / max} height={12} rx={2} className="fill-baseline" {...grow(0.04 + i * 0.1, 0.24 + i * 0.1)} />
            <Label x={430} y={yy + 20}>{`${p.v1.toLocaleString("en-US")} ${p.unit}`}</Label>
            <Label x={20} y={yy + 40} className="fill-accent">
              v2
            </Label>
            <rect x={60} y={yy + 30} width={360} height={12} rx={2} className="fill-border" />
            <motion.rect x={60} y={yy + 30} width={(360 * p.v2) / max} height={12} rx={2} className="fill-accent" {...grow(0.1 + i * 0.1, 0.3 + i * 0.1)} />
            <Label x={430} y={yy + 40} className="fill-accent">{`${p.v2.toLocaleString("en-US")} ${p.unit}`}</Label>
          </g>
        );
      })}
    </g>
  );
}

export function ProfileFigure({ pairs, dram }: { pairs: Pair[]; dram: number }) {
  const { grow } = useLoop(6);
  return (
    <Frame label="Nsight Compute on one launch of v1 and v2: v2 keeps more warps active, does more useful work per cycle and finishes sooner">
      <Label x={20} y={24}>Nsight Compute, one launch each</Label>
      <PairBars pairs={pairs} y={56} grow={grow} />
      <Label x={20} y={56 + pairs.length * 62 + 10}>{`DRAM throughput stays at ${dram}% or less in both:`}</Label>
      <Label x={20} y={56 + pairs.length * 62 + 28}>the win is latency and parallel work, not bandwidth</Label>
    </Frame>
  );
}

export function StepFigure({ pairs }: { pairs: Pair[] }) {
  const { grow } = useLoop(6);
  return (
    <Frame label="With v2 wired into the decode step, attention time per step drops and the batch-16 throughput rises the most">
      <Label x={20} y={24}>the decode step, before and after the switch</Label>
      <PairBars pairs={pairs} y={56} grow={grow} />
    </Frame>
  );
}

export function V2LinkFigure({ v1, v2 }: { v1: number; v2: number }) {
  const { grow } = useLoop(5);
  return (
    <Frame label={`Attention per decode step went from ${v1} to ${v2} µs when v2 replaced v1`}>
      <Label x={20} y={60} className="fill-text">
        attention per decode step (eager profile)
      </Label>
      <Label x={20} y={110}>v1: three passes, scores in shared memory</Label>
      <rect x={20} y={120} width={500} height={16} rx={3} className="fill-border" />
      <motion.rect x={20} y={120} width={500} height={16} rx={3} className="fill-baseline" {...grow(0.04, 0.6)} />
      <Label x={520} y={156} anchor="end">{`${v1} µs`}</Label>
      <Label x={20} y={200} className="fill-accent">
        v2: one pass, online softmax, 8 lanes per key
      </Label>
      <rect x={20} y={210} width={500} height={16} rx={3} className="fill-border" />
      <motion.rect x={20} y={210} width={(500 * v2) / v1} height={16} rx={3} className="fill-accent" {...grow(0.04, 0.04 + (0.56 * v2) / v1)} />
      <Label x={20 + (500 * v2) / v1} y={246} anchor="end" className="fill-accent">{`${v2} µs`}</Label>
      <Label x={20} y={320}>the context cap of v1 is gone too: v2 runs 8,192 tokens</Label>
    </Frame>
  );
}
