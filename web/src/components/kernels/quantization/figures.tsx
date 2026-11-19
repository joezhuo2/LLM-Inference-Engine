"use client";

import { motion } from "motion/react";
import { arithmetic, bf16Bits, binade, e4m3, floors, kvPool, kvToken, ladder, qFmt, qGb, qMs, ridge } from "@/content/kernels/quantization";
import { Frame, Label } from "../frame";
import { useLoop } from "../loop";

const X = 20;
const W = 520;
const fmt = qFmt;

const formats = [
  { name: "BF16, today", fields: [bf16Bits.sign, bf16Bits.exponent, bf16Bits.mantissa] },
  { name: "FP8 E4M3, the plan", fields: [e4m3.sign, e4m3.exponent, e4m3.mantissa] },
  { name: "4-bit, as in a GGUF Q4 file", fields: [0, 0, 4] },
];
const fieldCls = ["fill-muted/60", "fill-text/70", "fill-accent"];

export function BitsFigure() {
  const { on } = useLoop(14);
  const cell = W / 16;
  return (
    <Frame label={`One weight in bits: BF16 is 1 sign, ${bf16Bits.exponent} exponent and ${bf16Bits.mantissa} mantissa bits; FP8 E4M3 is 1, ${e4m3.exponent} and ${e4m3.mantissa}; a 4-bit weight is 4 bits plus a share of its block's scale`}>
      <Label x={X} y={22} className="fill-text">
        one weight, bit by bit, to scale
      </Label>
      {formats.map((f, r) => {
        const y = 52 + r * 92;
        let at = 0;
        return (
          <motion.g key={f.name} {...on(0.05 + r * 0.18)}>
            <Label x={X} y={y} className="fill-text">
              {f.name}
            </Label>
            <Label x={X + W} y={y} anchor="end" className="fill-accent">{`${f.fields.reduce((a, b) => a + b, 0)} bits`}</Label>
            {f.fields.map((n, k) =>
              Array.from({ length: n }, () => {
                const x = X + at * cell;
                at += 1;
                return <rect key={`${k}-${at}`} x={x + 1} y={y + 10} width={cell - 2} height={30} rx={2} className={r === 2 ? "fill-accent/40" : fieldCls[k]} />;
              }),
            )}
          </motion.g>
        );
      })}
      <motion.g {...on(0.65)}>
        <rect x={X} y={318} width={10} height={10} rx={1} className={fieldCls[0]} />
        <Label x={X + 16} y={327}>sign</Label>
        <rect x={X + 70} y={318} width={10} height={10} rx={1} className={fieldCls[1]} />
        <Label x={X + 86} y={327}>exponent: which power of two</Label>
        <rect x={X + 270} y={318} width={10} height={10} rx={1} className={fieldCls[2]} />
        <Label x={X + 286} y={327}>mantissa: where inside it</Label>
        <Label x={X} y={352} className="fill-accent">fewer bits per weight, fewer bytes per token</Label>
      </motion.g>
      <Label x={X} y={384}>to scale in bits; the 4-bit file also stores a scale per block of weights</Label>
    </Frame>
  );
}

export function FloorFigure() {
  const { on, grow } = useLoop(14);
  const px = W / floors[0].gb;
  return (
    <Frame label={`Bytes a decoded token reads and the floor at the measured bandwidth: ${floors.map((f) => `${f.name} ${qGb(f.gb * 1e9)}, ${qMs(f.ms)}`).join("; ")}`}>
      <Label x={X} y={22} className="fill-text">
        the weights a token reads, and the floor they set
      </Label>
      {floors.map((f, i) => {
        const y = 60 + i * 96;
        return (
          <motion.g key={f.name} {...on(0.05 + i * 0.15)}>
            <Label x={X} y={y} className="fill-text">
              {f.long}
            </Label>
            <Label x={X + W} y={y} anchor="end" className="fill-accent">{`${qGb(f.gb * 1e9)} · ${qMs(f.ms)}`}</Label>
            <rect x={X} y={y + 10} width={W} height={26} rx={2} className="fill-border" />
            <motion.rect x={X} y={y + 10} width={f.gb * px} height={26} rx={2} className={i === 0 ? "fill-text/70" : "fill-accent"} {...grow(0.08 + i * 0.15, 0.3 + i * 0.15)} />
          </motion.g>
        );
      })}
      <motion.g {...on(0.6)}>
        <Label x={X} y={352} className="fill-accent">half the bytes, half the floor: the floor is bytes ÷ bandwidth</Label>
      </motion.g>
      <Label x={X} y={384}>calculated from TinyLlama&apos;s shapes and the measured bandwidth; not measured: the kernels are stubs</Label>
    </Frame>
  );
}

export function RidgeFigure() {
  const { on, grow } = useLoop(14);
  const px = W / ridge;
  return (
    <Frame label={`FLOP per byte of weights against the ridge of ${fmt(ridge)}: ${arithmetic.map((a) => `${a.name} at ${a.rows} ${a.rows === 1 ? "row" : "rows"} ${fmt(a.intensity)}`).join(", ")}`}>
      <Label x={X} y={22} className="fill-text">
        FLOP per byte of weights, against the ridge
      </Label>
      {arithmetic.map((a, i) => {
        const y = 52 + i * 46;
        return (
          <motion.g key={i} {...on(0.04 + i * 0.08)}>
            <Label x={X} y={y}>{`${a.name}, ${a.rows} ${a.rows === 1 ? "row" : "rows"}`}</Label>
            <Label x={X + W} y={y} anchor="end" className="fill-accent">{`${fmt(a.intensity)} FLOP/byte · ${fmt(a.pct, a.pct < 10 ? 1 : 0)}% of the ridge`}</Label>
            <rect x={X} y={y + 7} width={W} height={12} rx={2} className="fill-border" />
            <motion.rect x={X} y={y + 7} width={Math.max(2, a.intensity * px)} height={12} rx={2} className="fill-accent" {...grow(0.06 + i * 0.08, 0.14 + i * 0.08)} />
          </motion.g>
        );
      })}
      <motion.g {...on(0.6)}>
        <Label x={X} y={348} className="fill-accent">{`every bar ends left of the ridge, ${fmt(ridge)} FLOP per byte: the read still sets the time`}</Label>
      </motion.g>
      <Label x={X} y={384}>calculated: 2 FLOP per weight per row ÷ bytes per weight, against the measured ceilings</Label>
    </Frame>
  );
}

export function LadderFigure() {
  const { on, move } = useLoop(14);
  const cell = W / ladder.length;
  return (
    <Frame label={`E4M3's values, one power of two per cell from 448 down: ${ladder.map((l) => `${l.label}, ${l.values} values`).join("; ")}`}>
      <Label x={X} y={22} className="fill-text">
        what E4M3 can hold, one power of two per cell
      </Label>
      {ladder.map((l, i) => (
        <motion.g key={l.label} {...on(0.04 + i * 0.02)}>
          <rect x={X + i * cell + 1} y={60} width={cell - 2} height={50} rx={2} className={l.tone === "ink" ? "fill-muted/50" : "fill-accent/70"} />
          <text x={X + i * cell + cell / 2} y={90} textAnchor="middle" className="fill-text text-[10px]">
            {l.values}
          </text>
        </motion.g>
      ))}
      <Label x={X} y={128}>448</Label>
      <Label x={X + W - cell} y={128} anchor="middle">{`2^${e4m3.minNormalExp}`}</Label>
      <Label x={X + W} y={128} anchor="end">0</Label>
      <motion.g {...move(0.36, 0.6, -cell * 8, 0, true, 0.94)}>
        <rect x={X + cell * 8 + 1} y={160} width={cell * 6 - 2} height={14} rx={2} className="fill-text/60" />
      </motion.g>
      <motion.g {...on(0.6)}>
        <Label x={X} y={194} className="fill-text">one row ÷ its scale: its largest weight lands on 448</Label>
      </motion.g>
      <motion.g {...on(0.66)}>
        <Label x={X} y={240} className="fill-accent">{`scale = the row's largest |w| ÷ ${e4m3.max}, one FP32 number per row`}</Label>
        <Label x={X} y={262}>{`8 values in every power of two from 2^${e4m3.minNormalExp} to 256, then 256 to ${e4m3.max} in steps of ${e4m3.topStep}`}</Label>
        <Label x={X} y={284}>{`below 2^${e4m3.minNormalExp}: 7 coarse values, 2^${e4m3.minSubExp} apart; under 2^${e4m3.minSubExp - 1} a weight becomes 0`}</Label>
      </motion.g>
      <Label x={X} y={384}>from the E4M3 definition in the spec; cells are powers of two, not to scale in value</Label>
    </Frame>
  );
}

export function RoundFigure() {
  const { on, move } = useLoop(14);
  const x = (v: number) => X + (v - 1) * W;
  return (
    <Frame label={`Between 1 and 2, E4M3 has ${e4m3.perPower} values and BF16 ${bf16Bits.perPower}; a weight rounds to the nearer value, at most half a step away`}>
      <Label x={X} y={22} className="fill-text">
        between 1 and 2: the values each format can hold
      </Label>
      <Label x={X} y={62}>{`BF16: ${bf16Bits.perPower} steps of 1/${bf16Bits.perPower}`}</Label>
      {Array.from({ length: bf16Bits.perPower + 1 }, (_, i) => (
        <line key={i} x1={x(1 + i / bf16Bits.perPower)} x2={x(1 + i / bf16Bits.perPower)} y1={72} y2={86} className="stroke-text/50" />
      ))}
      <Label x={X} y={132}>{`FP8 E4M3: ${e4m3.perPower} steps of 1/${e4m3.perPower}`}</Label>
      <line x1={x(1)} x2={x(2)} y1={160} y2={160} className="stroke-border" />
      {[...binade.map((b) => b.lo), 2].map((v) => (
        <g key={v}>
          <line x1={x(v)} x2={x(v)} y1={146} y2={174} className="stroke-accent" strokeWidth={2} />
          <Label x={x(v)} y={192} anchor="middle">
            {fmt(v, 3).replace(/0+$/, "").replace(/\.$/, "")}
          </Label>
        </g>
      ))}
      <motion.g {...on(0.3)}>
        <rect x={x(binade[2].lo)} y={208} width={x(binade[2].hi) - x(binade[2].lo)} height={10} rx={2} className="fill-accent/30" />
      </motion.g>
      <motion.circle cx={x(1.3)} cy={160} r={6} className="fill-warn" {...move(0.4, 0.6, x(binade[2].lo) - x(1.3), 0, true, 0.9)} />
      <motion.g {...on(0.62)}>
        <Label x={X} y={256} className="fill-accent">{`a weight moves at most half a step: 1/${2 * e4m3.perPower} of its power of two in FP8`}</Label>
        <Label x={X} y={278}>{`in BF16 at most 1/${2 * bf16Bits.perPower}: FP8's bound is ${bf16Bits.perPower / e4m3.perPower} times larger`}</Label>
        <Label x={X} y={300}>what that does to the model&apos;s answers is not measured</Label>
      </motion.g>
      <Label x={X} y={384}>calculated from the formats: 3 and 7 mantissa bits</Label>
    </Frame>
  );
}

export function PoolFigure() {
  const { on, grow } = useLoop(14);
  const px = W / kvPool.fp8;
  const rows = [
    { name: `BF16, ${fmt(kvToken.bf16)} bytes a token`, tokens: kvPool.bf16, contexts: kvPool.contextsBf16, cls: "fill-text/70" },
    { name: `FP8, ${fmt(kvToken.fp8)} bytes a token`, tokens: kvPool.fp8, contexts: kvPool.contextsFp8, cls: "fill-accent" },
  ];
  return (
    <Frame label={`The same ${qGb(kvPool.bytes)} pool holds ${fmt(kvPool.bf16)} tokens in BF16 and ${fmt(kvPool.fp8)} in FP8`}>
      <Label x={X} y={22} className="fill-text">{`the same ${qGb(kvPool.bytes)} pool, two widths of cache`}</Label>
      {rows.map((r, i) => {
        const y = 70 + i * 100;
        return (
          <motion.g key={r.name} {...on(0.05 + i * 0.2)}>
            <Label x={X} y={y} className="fill-text">
              {r.name}
            </Label>
            <Label x={X + W} y={y} anchor="end" className="fill-accent">{`${fmt(r.tokens)} tokens · ${fmt(r.contexts)} contexts`}</Label>
            <rect x={X} y={y + 10} width={W} height={30} rx={2} className="fill-border" />
            <motion.rect x={X} y={y + 10} width={r.tokens * px} height={30} rx={2} className={r.cls} {...grow(0.08 + i * 0.2, 0.3 + i * 0.2)} />
          </motion.g>
        );
      })}
      <motion.g {...on(0.62)}>
        <Label x={X} y={300} className="fill-accent">half the bytes per token, twice the blocks in the same memory</Label>
        <Label x={X} y={322}>attention does not get faster: it is about room first</Label>
      </motion.g>
      <Label x={X} y={384}>calculated from the measured pool; not measured: the kernels are stubs</Label>
    </Frame>
  );
}

const pieces = [
  { name: "DType::F8E4M3", what: "one byte per element", state: "done", cls: "fill-text/70" },
  { name: "write_kv_fp8", what: "round new keys and values into the cache", state: "stub, with tests", cls: "fill-accent" },
  { name: "paged_attention_decode_fp8", what: "read them back, byte × scale", state: "stub, with tests", cls: "fill-accent" },
  { name: "gemm_fp8", what: "BF16 activations × FP8 weights, a scale per row", state: "stub, with tests", cls: "fill-accent" },
  { name: "the wiring", what: "a flag, the scales, the measurements", state: "not started", cls: "fill-border" },
];

export function PlanFigure() {
  const { on } = useLoop(14);
  return (
    <Frame label={`The plan: ${pieces.map((p) => `${p.name}, ${p.state}`).join("; ")}`}>
      <Label x={X} y={22} className="fill-text">
        the plan, piece by piece
      </Label>
      {pieces.map((p, i) => {
        const y = 44 + i * 62;
        return (
          <motion.g key={p.name} {...on(0.05 + i * 0.1)}>
            <rect x={X} y={y} width={12} height={40} rx={2} className={p.cls} />
            <Label x={X + 24} y={y + 16} className="fill-text">
              {p.name}
            </Label>
            <Label x={X + W} y={y + 16} anchor="end" className={p.state === "done" ? "fill-text" : "fill-accent"}>
              {p.state}
            </Label>
            <Label x={X + 24} y={y + 34}>{p.what}</Label>
          </motion.g>
        );
      })}
      <Label x={X} y={384}>from the specs in src/kernels/gemm_fp8.cu and src/kernels/fp8_kv.cu</Label>
    </Frame>
  );
}
