"use client";

import { motion } from "motion/react";
import { type DeadEndKind, kindLabel } from "@/content/dead-ends";
import { fitEstimate } from "@/content/estimates";
import { prefix } from "@/content/kernels/prefix-caching";
import { Label } from "./frame";
import { useLoop } from "./loop";
import { SharesFigure } from "./tensor-parallelism/figures";

const X = 20;
const W = 520;

function Fig({ label, h, caption, children }: { label: string; h: number; caption: string; children: React.ReactNode }) {
  return (
    <figure className="max-w-2xl rounded border border-border bg-surface p-4">
      <svg viewBox={`0 0 560 ${h}`} className="w-full tabular-nums" role="img" aria-label={label}>
        {children}
      </svg>
      <figcaption className="mt-2 text-[11px] text-muted">{caption}</figcaption>
    </figure>
  );
}

const kindStroke: Record<DeadEndKind, string> = { slower: "stroke-muted", broke: "stroke-warn", cliff: "stroke-accent", misread: "stroke-text" };
const kindFill: Record<DeadEndKind, string> = { slower: "fill-muted", broke: "fill-warn", cliff: "fill-accent", misread: "fill-text" };
const kinds = Object.keys(kindLabel) as DeadEndKind[];

export function DeadEndsFigure({ items }: { items: DeadEndKind[] }) {
  const { on, draw, move } = useLoop(16);
  const y = 80;
  const pitch = (W - 60) / items.length;
  return (
    <Fig h={176} label={`The work as one line with ${items.length} branches that end: ${kinds.map((k) => `${items.filter((i) => i === k).length} ${kindLabel[k]}`).join(", ")}. From each end something returns to the line`} caption="one branch for each entry below, in the page's order; the dot that returns is what was kept">
      <motion.line x1={X} y1={y} x2={X + W} y2={y} strokeWidth={1.5} className="stroke-text" {...draw(0.02, 0.7)} />
      {items.map((kind, i) => {
        const x = X + 24 + i * pitch;
        const dy = i % 2 ? 44 : -44;
        const t = 0.06 + (i * 0.6) / items.length;
        return (
          <g key={i}>
            <motion.line x1={x} y1={y} x2={x + 20} y2={y + dy} strokeWidth={1.5} className={kindStroke[kind]} {...draw(t, t + 0.06)} />
            <motion.circle cx={x + 20} cy={y + dy} r={4} className={kindFill[kind]} {...on(t + 0.05)} />
            <motion.circle cx={x + 20} cy={y + dy} r={3} className="fill-accent" {...move(t + 0.08, t + 0.2, 20, -dy, false)} />
          </g>
        );
      })}
      {kinds.map((k, i) => (
        <g key={k}>
          <circle cx={X + 5 + (i % 2) * 270} cy={152 + Math.floor(i / 2) * 16} r={4} className={kindFill[k]} />
          <Label x={X + 16 + (i % 2) * 270} y={156 + Math.floor(i / 2) * 16}>{`${items.filter((v) => v === k).length} ${kindLabel[k]}`}</Label>
        </g>
      ))}
    </Fig>
  );
}

export type PathStop = { number: number; title: string; minutes: number };

export function ShortPathFigure({ chapters, stops }: { chapters: number; stops: PathStop[] }) {
  const { on, draw } = useLoop(18);
  const y = 70;
  const x = (n: number) => X + 8 + ((n - 1) * (W - 16)) / (chapters - 1);
  const onPath = new Set(stops.map((s) => s.number));
  const slot = 0.92 / stops.length;
  return (
    <Fig h={130} label={`The book's ${chapters} chapters in order; the path stops at chapters ${stops.map((s) => s.number).join(", ")}`} caption={`the ${chapters} chapters in the book's order; filled: on the path; the minutes are the estimate described above`}>
      {stops.slice(1).map((s, i) => {
        const a = x(stops[i].number);
        const b = x(s.number);
        return <motion.path key={s.number} d={`M${a} ${y - 8} Q${(a + b) / 2} ${y - 14 - Math.min(40, (b - a) / 3)} ${b} ${y - 8}`} fill="none" strokeWidth={1.5} className="stroke-accent" {...draw(0.02 + (i + 0.55) * slot, 0.02 + (i + 1) * slot)} />;
      })}
      {Array.from({ length: chapters }, (_, i) => (
        <g key={i}>
          <circle cx={x(i + 1)} cy={y} r={5} className={onPath.has(i + 1) ? "fill-accent" : "fill-none stroke-baseline"} />
          <text x={x(i + 1)} y={y + 20} textAnchor="middle" className={`${onPath.has(i + 1) ? "fill-text" : "fill-muted"} text-[10px]`}>
            {i + 1}
          </text>
        </g>
      ))}
      {stops.map((s, i) => (
        <motion.g key={s.number} {...on(0.02 + i * slot, 0.02 + (i + 1) * slot, false)}>
          <circle cx={x(s.number)} cy={y} r={9} className="fill-none stroke-accent" strokeWidth={1.5} />
          <Label x={X} y={118} className="fill-text">{`chapter ${s.number}: ${s.title} · about ${s.minutes} min`}</Label>
        </motion.g>
      ))}
    </Fig>
  );
}

const exerciseSteps = ["stub and tests", "tests fail", "write it", "tests pass", "compare"];

export function ExercisesFigure({ total, cpu, numpy }: { total: number; cpu: number; numpy: number }) {
  const { on, move } = useLoop(14);
  const pitch = 106;
  const slot = 0.18;
  const tests = (at: number, pass: boolean) =>
    Array.from({ length: 6 }, (_, k) => (
      <motion.rect key={k} x={X + at * pitch + 8 + k * 14} y={88} width={10} height={10} rx={2} className={pass ? "fill-accent" : "fill-none stroke-warn"} {...on(0.05 + at * slot + k * 0.012)} />
    ));
  return (
    <Fig h={116} label="An exercise in order: check out the stub and its tests, the tests fail, write the function, the tests pass, compare with the code that shipped" caption={`the same loop for each of the ${total} exercises; ${cpu} run on a CPU and ${numpy} more have a NumPy version`}>
      {exerciseSteps.map((s, i) => (
        <g key={s}>
          <rect x={X + i * pitch} y={30} width={96} height={40} rx={4} className="fill-bg stroke-border" />
          <motion.rect x={X + i * pitch} y={30} width={96} height={40} rx={4} className="fill-accent-dim stroke-accent" {...on(0.03 + i * slot, 0.03 + (i + 1) * slot, false)} />
          <text x={X + i * pitch + 48} y={54} textAnchor="middle" className="fill-text text-[11px]">
            {s}
          </text>
          <Label x={X + i * pitch} y={20}>{`${i + 1}`}</Label>
        </g>
      ))}
      {exerciseSteps.slice(1).map((s, i) => (
        <motion.circle key={s} cx={X + i * pitch + 96} cy={50} r={4} className="fill-accent" {...move(0.03 + (i + 1) * slot - 0.07, 0.03 + (i + 1) * slot + 0.02, 10, 0, false)} />
      ))}
      {tests(1, false)}
      {tests(3, true)}
    </Fig>
  );
}

function Quantization() {
  const { on, grow } = useLoop(12);
  const f = fitEstimate;
  const bits = f.bf16BytesPerWeight * 8;
  return (
    <Fig h={112} label={`A weight in BF16 is ${bits} bits; at ${f.lowBits} bits a decode step reads ${f.fewerBytes} times fewer bytes`} caption="bytes read for one decode step, to scale (calculated); a real file is a little larger, as the text says">
      <Label x={X} y={18} className="fill-text">{`BF16, ${bits} bits per weight: every step reads all of this`}</Label>
      <motion.rect x={X} y={26} width={W} height={16} rx={2} className="fill-text/70" {...grow(0.05, 0.3)} />
      <Label x={X} y={74} className="fill-text">{`${f.lowBits} bits per weight`}</Label>
      <rect x={X} y={82} width={W} height={16} rx={2} className="fill-border" />
      <motion.rect x={X} y={82} width={(W * f.lowBits) / bits} height={16} rx={2} className="fill-accent" {...grow(0.4, 0.55)} />
      <motion.g {...on(0.58)}>
        <Label x={X + (W * f.lowBits) / bits + 8} y={94} className="fill-accent">{`${f.fewerBytes} times fewer bytes`}</Label>
      </motion.g>
    </Fig>
  );
}

function Draft() {
  const { on, grow } = useLoop(14);
  const guesses = 3;
  const big = 110;
  return (
    <Fig h={150} label={`Plain decoding reads the large model's weights once for each of ${guesses + 1} tokens. A round with a draft reads the small draft ${guesses} times and the large model once for the same ${guesses + 1} tokens`} caption={`illustrative: a draft a tenth of the large model's size and a round where all ${guesses} guesses are kept; widths are bytes read`}>
      <Label x={X} y={18} className="fill-text">{`plain decoding: ${guesses + 1} tokens, the large model read ${guesses + 1} times`}</Label>
      {Array.from({ length: guesses + 1 }, (_, i) => (
        <motion.rect key={i} x={X + i * (big + 6)} y={26} width={big} height={16} rx={2} className="fill-text/70" {...grow(0.04 + i * 0.09, 0.12 + i * 0.09)} />
      ))}
      <Label x={X} y={88} className="fill-text">{`with a draft: ${guesses} small reads to guess, one large read to check them all`}</Label>
      {Array.from({ length: guesses }, (_, i) => (
        <motion.rect key={i} x={X + i * (big / 10 + 6)} y={96} width={big / 10} height={16} rx={2} className="fill-warn" {...grow(0.46 + i * 0.05, 0.52 + i * 0.05)} />
      ))}
      <motion.rect x={X + guesses * (big / 10 + 6)} y={96} width={big} height={16} rx={2} className="fill-accent" {...grow(0.64, 0.74)} />
      <motion.g {...on(0.76)}>
        <Label x={X + guesses * (big / 10 + 6) + big + 8} y={108} className="fill-accent">{`= the same ${guesses + 1} tokens`}</Label>
      </motion.g>
    </Fig>
  );
}

function Prefix() {
  const { on, grow } = useLoop(14);
  const block = 4;
  const shared = 10;
  const total = 14;
  const cell = 26;
  const whole = Math.floor(shared / block) * block;
  const cells = (from: number, count: number, y: number, cls: string) => Array.from({ length: count }, (_, i) => <rect key={i} x={X + (from + i) * cell} y={y} width={cell - 3} height={18} rx={2} className={cls} />);
  const edges = Array.from({ length: total / block + 1 }, (_, i) => <line key={i} x1={X + i * block * cell - 1.5} y1={20} x2={X + i * block * cell - 1.5} y2={124} strokeDasharray="2 3" className="stroke-border" />);
  return (
    <Fig h={150} label={`A later request shares its first ${shared} tokens with a cached prompt. In blocks of ${block}, this engine reuses ${whole} and computes the other ${shared - whole} shared tokens again; a cache that matches any run of tokens reuses all ${shared}`} caption={`illustrative: blocks of ${block} tokens here, ${prefix.block} in the engine; one cell is one prompt token's keys and values`}>
      {edges}
      <Label x={X} y={14} className="fill-text">{`whole blocks, as in this engine: ${whole} of ${shared} shared tokens reused`}</Label>
      <motion.g {...grow(0.04, 0.24)}>
        {cells(0, whole, 26, "fill-text/35")}
        {cells(whole, shared - whole, 26, "fill-warn/70")}
        {cells(shared, total - shared, 26, "fill-accent")}
      </motion.g>
      <motion.g {...on(0.26)}>
        <Label x={X + whole * cell} y={62} className="fill-warn">{`${shared - whole} shared tokens computed again`}</Label>
      </motion.g>
      <Label x={X} y={90} className="fill-text">{`any shared run, as in a radix tree: all ${shared} reused`}</Label>
      <motion.g {...on(0.4)}>
        <motion.g {...grow(0.42, 0.62)}>
          {cells(0, shared, 100, "fill-text/35")}
          {cells(shared, total - shared, 100, "fill-accent")}
        </motion.g>
      </motion.g>
    </Fig>
  );
}

export function BeyondFigure({ id }: { id: string }) {
  if (id === "quantization") return <Quantization />;
  if (id === "speculative-decoding") return <Draft />;
  if (id === "prefix-caching") return <Prefix />;
  if (id !== "multiple-gpus") return null;
  return (
    <figure className="max-w-2xl rounded border border-border bg-surface p-4">
      <div className="aspect-[560/400]">
        <SharesFigure />
      </div>
    </figure>
  );
}
