"use client";

import { motion } from "motion/react";
import { attentionMatrices, loadHistory, measuredRanks, mib, mlpMatrices, mostRanks, rankChoices, rankShare, tpFmt, tpShape, type TpMatrix } from "@/content/kernels/tensor-parallelism";
import { Frame, Label } from "../frame";
import { useLoop } from "../loop";

const fmt = tpFmt;
const X = 20;
const W = 520;
const full = rankShare(1).total;
const parts = (n: number) => {
  const s = rankShare(n);
  return [
    { label: "embedding", bytes: s.embed, cls: "fill-text/70" },
    { label: `layers ÷ ${n}`, bytes: s.layers, cls: "fill-accent" },
    { label: "LM head", bytes: s.head, cls: "fill-text/70" },
    { label: "norms", bytes: s.norms, cls: "fill-text/70" },
  ];
};

export function RankBar({ n, x, y, h, px }: { n: number; x: number; y: number; h: number; px: number }) {
  const list = parts(n);
  const starts = list.map((_, i) => list.slice(0, i).reduce((a, p) => a + p.bytes * px, x));
  return (
    <>
      {list.map((p, i) => (
        <rect key={p.label} x={starts[i]} y={y} width={Math.max(1, p.bytes * px - 1)} height={h} rx={2} className={p.cls} />
      ))}
    </>
  );
}

export function SharesFigure() {
  const { on } = useLoop(14);
  const px = W / full;
  const groups = rankChoices.map((n, k) => ({ n, top: 44 + rankChoices.slice(0, k).reduce((a, m) => a + 26 + m * 22, 0), s: rankShare(n) }));
  return (
    <Frame label={`TinyLlama on ${rankChoices.join(", ")} GPUs: ${groups.map((g) => `with ${g.n}, each holds ${fmt(g.s.pct)}% of the model`).join("; ")}`}>
      <Label x={X} y={22} className="fill-text">
        one model, cut across GPUs: what each GPU holds, to scale
      </Label>
      {groups.map((g, k) => (
        <motion.g key={g.n} {...on(0.05 + k * 0.2)}>
          <Label x={X} y={g.top + 12} className="fill-text">{g.n === 1 ? "1 GPU: the whole model" : `${g.n} GPUs: each holds ${fmt(g.s.pct)}% of the model`}</Label>
          {Array.from({ length: g.n }, (_, r) => (
            <RankBar key={r} n={g.n} x={X} y={g.top + 20 + r * 22} h={16} px={px} />
          ))}
        </motion.g>
      ))}
      <rect x={X} y={354} width={10} height={10} rx={1} className="fill-accent" />
      <Label x={X + 16} y={363}>every layer, cut in N</Label>
      <rect x={X + 170} y={354} width={10} height={10} rx={1} className="fill-text/70" />
      <Label x={X + 186} y={363}>embedding, LM head and norms: whole in every GPU</Label>
      <Label x={X} y={388}>calculated from TinyLlama&apos;s shapes at 2 bytes per weight</Label>
    </Frame>
  );
}

export function Matrix({ m, x, y, w, h, n, on }: { m: TpMatrix; x: number; y: number; w: number; h: number; n: number; on: ReturnType<typeof useLoop>["on"] }) {
  const pieces: { x: number; y: number; w: number; h: number; r: number }[] = [];
  if (m.cut === "rows") {
    let top = y;
    for (const b of m.bands) {
      const bh = (b.rows / m.rows) * h;
      for (let r = 0; r < n; r++) pieces.push({ x, y: top + (r * bh) / n, w, h: bh / n, r });
      top += bh;
    }
  } else for (let r = 0; r < n; r++) pieces.push({ x: x + (r * w) / n, y, w: w / n, h, r });
  return (
    <g>
      <Label x={x} y={y - 22} className="fill-text">{m.label}</Label>
      <Label x={x} y={y - 8}>{`${fmt(m.rows)} rows × ${fmt(m.cols)} columns, cut by ${m.cut === "rows" ? "rows" : "columns"}`}</Label>
      {pieces.map((p, i) => (
        <rect key={i} x={p.x} y={p.y} width={Math.max(1, p.w - 1)} height={Math.max(1, p.h - 1)} className={p.r % 2 ? "fill-text/25" : "fill-text/40"} />
      ))}
      {Array.from({ length: n }, (_, r) => (
        <motion.g key={r} {...on(0.08 + (r * 0.84) / n, 0.08 + ((r + 1) * 0.84) / n, false)}>
          {pieces.filter((p) => p.r === r).map((p, i) => (
            <rect key={i} x={p.x} y={p.y} width={Math.max(1, p.w - 1)} height={Math.max(1, p.h - 1)} className="fill-accent" />
          ))}
        </motion.g>
      ))}
      {m.bands.length > 1 &&
        m.bands.map((b, i) => {
          const top = y + (m.bands.slice(0, i).reduce((a, c) => a + c.rows, 0) / m.rows) * h;
          return (
            <Label key={b.label} x={x + w + 6} y={top + Math.max(10, ((b.rows / m.rows) * h) / 2 + 4)}>
              {b.label}
            </Label>
          );
        })}
    </g>
  );
}

export function MatrixFigure({ part }: { part: "attention" | "mlp" }) {
  const { on } = useLoop(16);
  const [a, b] = part === "attention" ? attentionMatrices : mlpMatrices;
  const n = mostRanks;
  const what = part === "attention" ? `${tpShape.heads / n} query heads and ${tpShape.kv_heads / n} K/V ${tpShape.kv_heads / n === 1 ? "head" : "heads"}` : `${fmt(tpShape.intermediate / n)} channels`;
  return (
    <Frame label={`${a.label} cut by rows and ${b.label} cut by columns into ${n} ranks; each rank gets ${what}`}>
      <Label x={X} y={22} className="fill-text">{`one layer's ${part === "attention" ? "attention" : "MLP"} weights, cut for ${n} ranks`}</Label>
      <Matrix m={a} x={X} y={78} w={210} h={250} n={n} on={on} />
      <Matrix m={b} x={X + 290} y={78} w={210} h={250} n={n} on={on} />
      <Label x={X} y={360} className="fill-accent">{`in blue, one rank after another: ${what} each`}</Label>
      <Label x={X} y={388}>rows and columns from plan_shard_layout() in src/loader/tensor_parallel.cpp; boxes not to scale</Label>
    </Frame>
  );
}

export function SumsFigure() {
  const { on } = useLoop(14);
  const lanes = [0, 1];
  const ys = [92, 212];
  return (
    <Frame label="Each rank multiplies the whole input by its rows, gets its own slice, multiplies that by its columns, and gets a full-size partial result; the partial results add up to the layer's output">
      <Label x={X} y={22} className="fill-text">
        rows first, then columns: the partial results just add
      </Label>
      <rect x={X} y={70} width={16} height={180} rx={2} className="fill-text/70" />
      <Label x={X} y={270}>x, whole</Label>
      {lanes.map((r) => (
        <motion.g key={r} {...on(0.05 + r * 0.2)}>
          <Label x={X + 40} y={ys[r] - 26} className="fill-text">{`rank ${r}`}</Label>
          <rect x={X + 40} y={ys[r] - 16} width={80} height={56} rx={2} className="fill-text/30" />
          <Label x={X + 46} y={ys[r] + 18}>its rows</Label>
          <rect x={X + 140} y={ys[r] - 4} width={16} height={32} rx={2} className="fill-accent/60" />
          <Label x={X + 128} y={ys[r] + 50}>its slice</Label>
          <rect x={X + 176} y={ys[r] - 16} width={56} height={56} rx={2} className="fill-text/30" />
          <Label x={X + 180} y={ys[r] + 18}>its columns</Label>
          <rect x={X + 260} y={ys[r] - 30} width={16} height={84} rx={2} className="fill-accent" />
          <Label x={X + 284} y={ys[r] + 18}>a partial result, full size</Label>
        </motion.g>
      ))}
      <motion.g {...on(0.5)}>
        <text x={X + 268} y={160} textAnchor="middle" className="fill-text text-[18px]">
          +
        </text>
        <rect x={X + 440} y={70} width={16} height={180} rx={2} className="fill-accent" />
        <Label x={X + 400} y={270} className="fill-accent">the sum = the whole layer</Label>
      </motion.g>
      <Label x={X} y={316} className="fill-text">{`twice in every layer: after attention, and after the MLP (${tpShape.sums} sums per token)`}</Label>
      <Label x={X} y={336}>RMSNorm, the embedding and the LM head are not cut: they run on the summed vector</Label>
      <Label x={X} y={388}>the order of a layer from src/loader/tensor_parallel.cpp; the check in tests/unit/test_tensor_parallel.cpp</Label>
    </Frame>
  );
}

const forwardRows = [
  { title: "attention", ops: ["QKV", "RoPE", "K/V", "attn", "wo"] },
  { title: "MLP", ops: ["gate·up", "SwiGLU", "down"] },
];

export function ForwardFigure() {
  const { on } = useLoop(16);
  const n = 2;
  const left = X + 64;
  return (
    <Frame label={`One layer with ${n} ranks on one GPU: a shared RMSNorm, then each rank in turn runs its part into a partial result, then a sum into the residual stream; once for attention and once for the MLP`}>
      <Label x={X} y={22} className="fill-text">{`one layer of the forward pass, ${n} ranks on one GPU`}</Label>
      {forwardRows.map((row, k) => {
        const top = 52 + k * 150;
        const boxes = 2 + n * row.ops.length;
        const pitch = (W - 64) / boxes;
        const box = (i: number, lane: number, label: string, cls: string, delay: number) => (
          <motion.g key={`${lane}-${i}`} {...on(delay)}>
            <rect x={left + i * pitch} y={top + 18 + lane * 34} width={pitch - 3} height={26} rx={2} className={cls} />
            <text x={left + i * pitch + (pitch - 3) / 2} y={top + 35 + lane * 34} textAnchor="middle" className={`${cls === "fill-accent" ? "fill-bg" : "fill-text"} text-[10px]`}>
              {label}
            </text>
          </motion.g>
        );
        return (
          <g key={row.title}>
            <Label x={X} y={top + 6} className="fill-text">{row.title}</Label>
            {["shared", ...Array.from({ length: n }, (_, r) => `rank ${r}`)].map((lane, l) => (
              <Label key={lane} x={X} y={top + 35 + l * 34}>{lane}</Label>
            ))}
            {box(0, 0, "norm", "fill-text/30", 0.04 + k * 0.4)}
            {Array.from({ length: n }, (_, r) => row.ops.map((op, i) => box(1 + r * row.ops.length + i, 1 + r, op, "fill-accent/50", 0.08 + k * 0.4 + (r * row.ops.length + i) * 0.02)))}
            {box(boxes - 1, 0, "sum", "fill-accent", 0.3 + k * 0.4)}
          </g>
        );
      })}
      <Label x={X} y={352} className="fill-text">each rank keeps its own K/V cache and scratch buffers; the sums add into the residual stream</Label>
      <Label x={X} y={388}>the ranks take turns on the one GPU, so the pass cannot be faster than the whole model</Label>
    </Frame>
  );
}

export function RankBytesFigure() {
  const { on } = useLoop(14);
  const px = W / full;
  const rows = measuredRanks.filter((r) => r.n !== 1);
  return (
    <Frame label={rows.map((r) => `${r.n === 0 ? "the whole model" : `a rank of ${r.n}`}: ${fmt(r.mib, 1)} MiB`).join("; ")}>
      <Label x={X} y={22} className="fill-text">
        what one rank holds, to scale
      </Label>
      {rows.map((r, i) => (
        <motion.g key={r.n} {...on(0.05 + i * 0.15)}>
          <Label x={X} y={62 + i * 78} className="fill-text">{r.n === 0 ? "the whole model" : `a rank of ${r.n}: ${fmt(r.calc.pct)}%, not ${fmt(100 / r.n)}%`}</Label>
          <Label x={X + W} y={62 + i * 78} anchor="end" className="fill-accent">{`${fmt(r.mib, 1)} MiB`}</Label>
          <RankBar n={Math.max(1, r.n)} x={X} y={72 + i * 78} h={26} px={px} />
          {r.n > 1 && <line x1={X + W / r.n} x2={X + W / r.n} y1={68 + i * 78} y2={104 + i * 78} strokeDasharray="3 3" className="stroke-text" />}
        </motion.g>
      ))}
      <Label x={X} y={300} className="fill-text">{`the dashed line is 1/N of the model; the embedding and the LM head, ${mib(rankShare(1).embed)} each, stay whole`}</Label>
      <rect x={X} y={318} width={10} height={10} rx={1} className="fill-accent" />
      <Label x={X + 16} y={327}>layer matrices, divided by N</Label>
      <rect x={X + 200} y={318} width={10} height={10} rx={1} className="fill-text/70" />
      <Label x={X + 216} y={327}>embedding, LM head, norms</Label>
      <Label x={X} y={388}>MiB on the device, results.md, Tensor parallelism (measured); the parts calculated from the shapes</Label>
    </Frame>
  );
}

export function LoadFigure() {
  const { grow, on } = useLoop(14);
  const most = Math.max(...loadHistory.rows.map((r) => r.perCopy));
  const px = (W - 170 - 70) / most;
  return (
    <Frame label={loadHistory.rows.map((r) => `${r.layout}: ${fmt(r.perCopy, 1)} ms with one call per copy, ${fmt(r.packed, 1)} ms with one chunk per tensor`).join("; ")}>
      <Label x={X} y={22} className="fill-text">
        the load of one rank, before and after packing the copies
      </Label>
      {loadHistory.rows.map((r, i) => (
        <g key={r.layout}>
          <Label x={X} y={60 + i * 70} className="fill-text">{r.layout}</Label>
          <Label x={X} y={76 + i * 70}>{`${fmt(r.copies)} copies`}</Label>
          <motion.rect x={X + 170} y={50 + i * 70} width={Math.max(2, r.perCopy * px)} height={14} rx={2} className="fill-text/40" {...grow(0.05 + i * 0.05, 0.3 + i * 0.05)} />
          <motion.rect x={X + 170} y={68 + i * 70} width={Math.max(2, r.packed * px)} height={14} rx={2} className="fill-accent" {...grow(0.4 + i * 0.05, 0.6 + i * 0.05)} />
          <motion.g {...on(0.3 + i * 0.05)}>
            <Label x={X + 176 + r.perCopy * px} y={61 + i * 70}>{`${fmt(r.perCopy)} ms`}</Label>
          </motion.g>
          <motion.g {...on(0.6 + i * 0.05)}>
            <Label x={X + 176 + Math.max(2, r.packed * px)} y={79 + i * 70} className="fill-accent">{`${fmt(r.packed)} ms`}</Label>
          </motion.g>
        </g>
      ))}
      <rect x={X} y={340} width={10} height={10} rx={1} className="fill-text/40" />
      <Label x={X + 16} y={349}>{`one call per copy (${loadHistory.before})`}</Label>
      <rect x={X + 220} y={340} width={10} height={10} rx={1} className="fill-accent" />
      <Label x={X + 236} y={349}>{`the copies of one tensor in one chunk (${loadHistory.after})`}</Label>
      <Label x={X} y={388}>upload ms, median of 21 uploads, results.md, Tensor parallelism (measured)</Label>
    </Frame>
  );
}

export function WireFigure() {
  const { on } = useLoop(14);
  const n = mostRanks;
  const ring = Array.from({ length: n }, (_, r) => {
    const a = (r / n) * Math.PI * 2 - Math.PI / 2;
    return { x: 420 + Math.cos(a) * 80, y: 190 + Math.sin(a) * 80 };
  });
  return (
    <Frame label={`Here every rank runs on the one GPU, in turn, and a sum is an add in its memory. Across ${n} GPUs each of the ${tpShape.sums} sums of a token would cross the link between them; that is not measured here`}>
      <Label x={X} y={22} className="fill-text">
        where the sums happen
      </Label>
      <Label x={X} y={62} className="fill-text">this laptop: one GPU</Label>
      <rect x={X} y={74} width={220} height={230} rx={6} className="fill-text/10 stroke-text/40" />
      {Array.from({ length: n }, (_, r) => (
        <motion.g key={r} {...on(0.05 + r * 0.08)}>
          <rect x={X + 20} y={92 + r * 46} width={180} height={34} rx={3} className="fill-accent/50" />
          <Label x={X + 30} y={113 + r * 46} className="fill-text">{`rank ${r}, its turn`}</Label>
        </motion.g>
      ))}
      <Label x={X} y={326}>a sum is an add in the one GPU&apos;s memory</Label>
      <Label x={330} y={62} className="fill-text">{`${n} real GPUs`}</Label>
      {ring.map((p, r) => {
        const q = ring[(r + 1) % n];
        return <line key={`l${r}`} x1={p.x} y1={p.y} x2={q.x} y2={q.y} strokeDasharray="4 4" className="stroke-text/50" />;
      })}
      {ring.map((p, r) => (
        <g key={r}>
          <rect x={p.x - 30} y={p.y - 16} width={60} height={32} rx={4} className="fill-text/30" />
          <text x={p.x} y={p.y + 4} textAnchor="middle" className="fill-text text-[11px]">{`GPU ${r}`}</text>
        </g>
      ))}
      <Label x={330} y={306}>{`${tpShape.sums} sums per token cross the link:`}</Label>
      <Label x={330} y={322}>not measured here, no NCCL</Label>
      <Label x={X} y={388}>{`${tpShape.layers} layers × 2 sums, calculated; this engine has no second GPU`}</Label>
    </Frame>
  );
}
