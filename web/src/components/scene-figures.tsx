"use client";

import { animate, motion, type MotionValue, useMotionValue, useTransform } from "motion/react";
import { useEffect } from "react";
import { metric } from "@/components/gpu-floorplan";
import { Frame, Label } from "@/components/kernels/frame";
import { useLoop } from "@/components/kernels/loop";
import { Chip, short } from "@/components/kernels/speculative-decoding/figures";
import { Matrix, RankBar } from "@/components/kernels/tensor-parallelism/figures";
import { bucketScene, rankScene, sweepScene } from "@/content/build-log";
import { spec, specFmt } from "@/content/kernels/speculative-decoding";
import { attentionMatrices, rankShare, shardedForward, tpFmt, tpShape } from "@/content/kernels/tensor-parallelism";
import { gemmOps, roofline } from "@/content/kernels/gemm";
import { ceilings } from "@/lib/data";

function useClock(duration: number) {
  const { reduced } = useLoop(duration);
  const clock = useMotionValue(reduced ? 1 : 0);
  useEffect(() => {
    if (reduced) {
      clock.set(1);
      return;
    }
    const run = animate(clock, [0, 1], { duration, ease: "linear", repeat: Infinity });
    return () => run.stop();
  }, [reduced, clock, duration]);
  return clock;
}

function Typed({ clock, text, start, end, x, y, className = "fill-text" }: { clock: MotionValue<number>; text: string; start: number; end: number; x: number; y: number; className?: string }) {
  const shown = useTransform(clock, (v) => text.slice(0, Math.round(Math.min(1, Math.max(0, (v - start) / (end - start))) * text.length)));
  return (
    <motion.text x={x} y={y} className={`${className} text-[13px]`}>
      {shown}
    </motion.text>
  );
}

const template = [
  { piece: "<|user|>", id: 29966 },
  { piece: "▁What", id: 1724 },
  { piece: "▁is", id: 338 },
  { piece: "▁the", id: 278 },
  { piece: "▁capital", id: 7483 },
  { piece: "▁of", id: 310 },
  { piece: "▁France", id: 3444 },
  { piece: "?", id: 29973 },
  { piece: "</s>", id: 2 },
  { piece: "<|assistant|>", id: 29989 },
];

export function TokenizeFigure() {
  const { on } = useLoop(10);
  const clock = useClock(10);
  return (
    <Frame label="A user types a prompt into a chat box, the prompt is wrapped in TinyLlama's three-line chat template, then split into SentencePiece pieces that each map to an integer token id. The ids are illustrative">
      <Label x={20} y={28}>user types a prompt</Label>
      <rect x={20} y={36} width={520} height={30} rx={3} className="fill-surface stroke-border" />
      <Typed clock={clock} text={prompt} start={0.02} end={0.22} x={32} y={56} />
      <motion.g {...on(0.27)}>
        <Label x={20} y={88}>{"wrapped in TinyLlama's chat template"}</Label>
        <rect x={20} y={96} width={520} height={56} rx={3} className="fill-surface stroke-border" />
      </motion.g>
      {["<|user|>", `${prompt}</s>`, "<|assistant|>"].map((line, i) => (
        <motion.g key={line} {...on(0.3 + i * 0.05)}>
          <text x={32} y={112 + i * 15} className="fill-text text-[13px]">
            {line}
          </text>
        </motion.g>
      ))}
      <motion.g {...on(0.46)}>
        <Label x={20} y={174}>SentencePiece pieces (▁ marks a word start), then ids</Label>
      </motion.g>
      {template.map(({ piece, id }, k) => {
        const x = 20 + (k % 5) * 106;
        const y = 182 + Math.floor(k / 5) * 80;
        const t = 0.48 + k * 0.025;
        const u = 0.66 + k * 0.02;
        return (
          <g key={piece}>
            <motion.g {...on(0.46)}>
              <rect x={x} y={y} width={100} height={68} rx={3} className="fill-none stroke-border" />
            </motion.g>
            <motion.g {...on(t)}>
              <rect x={x} y={y} width={100} height={34} rx={3} className="fill-accent/10 stroke-accent" />
              <text x={x + 50} y={y + 21} textAnchor="middle" className="fill-text text-[12px]">
                {piece}
              </text>
            </motion.g>
            <motion.g {...on(u)}>
              <text x={x + 50} y={y + 55} textAnchor="middle" className="fill-accent text-[15px] tabular-nums">
                {id}
              </text>
            </motion.g>
          </g>
        );
      })}
      <motion.g {...on(0.86)}>
        <Label x={20} y={356}>each id picks one row of the embedding table</Label>
      </motion.g>
      <Label x={20} y={380}>{"token ids are illustrative, not TinyLlama's real ones"}</Label>
    </Frame>
  );
}

const prompt = "What is the capital of France?";
const reply = "The capital of France is Paris.";

const decode: { tok: string; alts: [string, number][] }[] = [
  { tok: "The", alts: [["A", 0.14], ["Paris", 0.08], ["In", 0.06], ["I", 0.04]] },
  { tok: " capital", alts: [[" city", 0.12], [" answer", 0.07], [" name", 0.05], [" French", 0.03]] },
  { tok: " of", alts: [[" city", 0.08], [" is", 0.05], [" for", 0.03], [",", 0.02]] },
  { tok: " France", alts: [[" the", 0.1], [" this", 0.04], [" a", 0.03], [" Paris", 0.02]] },
  { tok: " is", alts: [[" was", 0.09], [",", 0.05], [" city", 0.03], [":", 0.02]] },
  { tok: " Paris", alts: [[" Lyon", 0.05], [" a", 0.04], [" the", 0.03], [" Nice", 0.02]] },
  { tok: ".", alts: [["!", 0.1], [",", 0.06], ["↵", 0.04], [" </s>", 0.03]] },
];
const probs = [0.62, 0.71, 0.83, 0.78, 0.74, 0.81, 0.69];

export function GenerateFigure() {
  const { on } = useLoop(12);
  const clock = useClock(12);
  const first = 0.26;
  const step = 0.065;
  return (
    <Frame label="A prompt is typed into a chat box; the engine then runs decode steps, each choosing the highest-probability next token from a small top-5 bar chart and appending it to a token strip; finally the reply appears. Greedy decoding, probabilities illustrative">
      <Label x={20} y={22}>chat input</Label>
      <rect x={20} y={30} width={520} height={30} rx={3} className="fill-surface stroke-border" />
      <Typed clock={clock} text={prompt} start={0.02} end={0.22} x={32} y={50} />
      <Label x={20} y={84}>top-5 next-token probabilities</Label>
      <line x1={20} y1={210} x2={540} y2={210} className="stroke-border" />
      {decode.map(({ tok, alts }, k) => {
        const t = first + k * step;
        const last = k === decode.length - 1;
        const bars: [string, number][] = [[tok, probs[k]], ...alts];
        return (
          <motion.g key={k} {...on(t, t + step, last)}>
            <Label x={540} y={84} anchor="end" className="fill-text">{`decode step ${k + 1}`}</Label>
            {bars.map(([name, p], j) => (
              <g key={j}>
                <rect x={40 + j * 100} y={210 - p * 100} width={70} height={p * 100} rx={2} className={j === 0 ? "fill-accent" : "fill-baseline/60"} />
                <Label x={75 + j * 100} y={204 - p * 100} anchor="middle" className={j === 0 ? "fill-accent" : "fill-muted"}>
                  {p.toFixed(2)}
                </Label>
                <Label x={75 + j * 100} y={226} anchor="middle" className={j === 0 ? "fill-accent" : "fill-muted"}>
                  {name.trim() === "" ? "·" : name.trim()}
                </Label>
              </g>
            ))}
          </motion.g>
        );
      })}
      <Label x={20} y={254}>tokens so far</Label>
      {decode.map(({ tok }, k) => {
        const t = first + k * step;
        return (
          <motion.g key={k} {...on(t + step * 0.6)}>
            <rect x={20 + k * 74} y={262} width={68} height={24} rx={3} className="fill-accent/10 stroke-accent" />
            <text x={54 + k * 74} y={278} textAnchor="middle" className="fill-text text-[11px]">
              {tok.trim()}
            </text>
          </motion.g>
        );
      })}
      <Label x={20} y={312}>reply</Label>
      <rect x={20} y={320} width={520} height={30} rx={3} className="fill-surface stroke-border" />
      <Typed clock={clock} text={reply} start={0.74} end={0.92} x={32} y={340} className="fill-accent" />
      <Label x={20} y={380}>greedy decoding: the argmax every step; probabilities are illustrative</Label>
    </Frame>
  );
}

const tokens = ["Paris", "Lyon", "Nice", "Rome", "a", "the"];
const logits = [3.1, 2.6, 2.2, 1.4, 0.9, 0.3];
const noise = [0.2, 1.6, 0.5, 0.1, 0.9, 1.2];
const temp = 0.7;
const unit = 26;
const base = 250;
const picked = logits.map((l, i) => l / temp + noise[i]).reduce((best, v, i, all) => (v > all[best] ? i : best), 0);

export function SampleFigure() {
  const { on } = useLoop(9);
  const bx = (i: number) => 40 + i * 85;
  return (
    <Frame label="Logits are drawn as bars, temperature 0.7 sharpens them, Gumbel noise is added to each bar and the largest logit plus noise is the picked token; the same seed 42 gives the same pick every run. Values are illustrative">
      <Label x={20} y={28}>next-token logits</Label>
      <line x1={20} y1={base} x2={540} y2={base} className="stroke-border" />
      {tokens.map((name, i) => (
        <Label key={name} x={bx(i) + 25} y={base + 16} anchor="middle" className={i === picked ? "fill-accent" : "fill-muted"}>
          {name}
        </Label>
      ))}
      <motion.g {...on(0.04, 0.3, false)}>
        <Label x={540} y={28} anchor="end" className="fill-text">temperature T = 1</Label>
        {logits.map((l, i) => (
          <rect key={i} x={bx(i)} y={base - l * unit} width={50} height={l * unit} rx={2} className="fill-baseline/60" />
        ))}
      </motion.g>
      <motion.g {...on(0.32)}>
        <Label x={540} y={28} anchor="end" className="fill-text">logits / T, T = 0.7</Label>
        {logits.map((l, i) => (
          <rect key={i} x={bx(i)} y={base - (l / temp) * unit} width={50} height={(l / temp) * unit} rx={2} className="fill-baseline/60" />
        ))}
      </motion.g>
      <motion.g {...on(0.5)}>
        <Label x={20} y={48} className="fill-warn">+ Gumbel noise, one draw per bar</Label>
        {logits.map((l, i) => (
          <rect key={i} x={bx(i)} y={base - (l / temp + noise[i]) * unit} width={50} height={noise[i] * unit} rx={2} className="fill-warn/60" />
        ))}
      </motion.g>
      <motion.g {...on(0.7)}>
        <rect x={bx(picked) - 3} y={base - (logits[picked] / temp + noise[picked]) * unit - 3} width={56} height={(logits[picked] / temp + noise[picked]) * unit + 6} rx={3} className="fill-accent/20 stroke-accent" />
        <Label x={20} y={68} className="fill-accent">{`max of logit + noise → ${tokens[picked]}`}</Label>
      </motion.g>
      <motion.g {...on(0.82)}>
        <rect x={20} y={318} width={250} height={34} rx={4} className="fill-accent/10 stroke-accent" />
        <text x={32} y={340} className="fill-accent text-[15px]">seed 42 → same pick</text>
        <Label x={286} y={340}>Philox: a seed replays the run</Label>
      </motion.g>
      <Label x={20} y={380}>logits and noise are illustrative, not measured</Label>
    </Frame>
  );
}

const roofBandwidth = (ai: number) => Math.min((ceilings().gbs / 1000) * ai, ceilings().tflops);
const aiMin = 0.5;
const aiMax = 1000;
const tMin = 0.03;
const tMax = 100;
const px = (ai: number) => 60 + (Math.log10(ai / aiMin) / Math.log10(aiMax / aiMin)) * 470;
const py = (t: number) => 320 - (Math.log10(t / tMin) / Math.log10(tMax / tMin)) * 280;

const points = (rows: string) =>
  gemmOps.map((op) => ({ x: px(roofline(op, rows, "FLOP/byte")), y: py(roofline(op, rows, "TFLOP/s")), pct: roofline(op, rows, "% of roof") }));
const batch1 = points("1");
const batch16 = points("16");
const pctRange = (pts: typeof batch1) => `${Math.min(...pts.map((p) => p.pct))}–${Math.max(...pts.map((p) => p.pct))}%`;
const stepRoof = metric("tools/roofline");

export function RooflineFigure() {
  const { on, move } = useLoop(8);
  const { gbs, tflops } = ceilings();
  const ridge = (tflops * 1000) / gbs;
  const slant = `${px(aiMin)},${py(roofBandwidth(aiMin))} ${px(ridge)},${py(tflops)}`;
  const dot = (pts: typeof batch1, start: number) =>
    pts.map((p, i) => <motion.circle key={i} cx={p.x} cy={320} r={4} className="fill-accent" {...move(start + i * 0.015, start + 0.14 + i * 0.015, 0, p.y - 320, true, 0.94)} />);
  return (
    <Frame label={`Log-log roofline: the ${gbs} GB/s bandwidth roof meets the ${tflops} TFLOP/s compute roof at about ${Math.round(ridge)} FLOP/byte. The batch-1 matrix-vector products sit at ${pctRange(batch1)} of the bandwidth roof, and the whole batch-1 step reaches ${stepRoof}`}>
      <Label x={12} y={28}>TFLOP/s</Label>
      <rect x={60} y={40} width={470} height={280} className="fill-none stroke-border" />
      {[1, 10, 100, 1000].map((ai) => (
        <g key={ai}>
          <line x1={px(ai)} y1={320} x2={px(ai)} y2={325} className="stroke-border" />
          <Label x={px(ai)} y={338} anchor="middle">{ai}</Label>
        </g>
      ))}
      {[0.1, 1, 10, 100].map((t) => (
        <g key={t}>
          <line x1={55} y1={py(t)} x2={60} y2={py(t)} className="stroke-border" />
          <Label x={52} y={py(t) + 4} anchor="end">{t}</Label>
        </g>
      ))}
      <Label x={295} y={360} anchor="middle">arithmetic intensity (FLOP/byte)</Label>
      <motion.polyline points={slant} fill="none" strokeWidth={2} className="stroke-accent" {...on(0.04)} />
      <motion.line x1={px(ridge)} y1={py(tflops)} x2={530} y2={py(tflops)} strokeWidth={2} className="stroke-accent" {...on(0.2)} />
      <motion.g {...on(0.2)}>
        <Label x={528} y={py(tflops) - 6} anchor="end" className="fill-accent">{`compute roof ${tflops} TFLOP/s`}</Label>
        <Label x={px(ridge) + 6} y={py(tflops) + 16}>{`ridge ~${Math.round(ridge)}`}</Label>
      </motion.g>
      <motion.g {...on(0.1)}>
        <Label x={px(1.5)} y={py(roofBandwidth(1.5)) + 28} className="fill-accent">{`${gbs} GB/s bandwidth roof`}</Label>
      </motion.g>
      {dot(batch1, 0.34)}
      <motion.g {...on(0.5)}>
        <Label x={batch1[0].x + 12} y={batch1[0].y + 34} className="fill-text">{`batch 1 GEMVs: ${pctRange(batch1)} of roof`}</Label>
      </motion.g>
      {dot(batch16, 0.42)}
      <motion.g {...on(0.56)}>
        <Label x={batch16[0].x + 12} y={batch16[0].y + 34} className="fill-text">{`batch 16: ${pctRange(batch16)}`}</Label>
      </motion.g>
      <motion.g {...on(0.66)}>
        <rect x={72} y={52} width={210} height={44} rx={4} className="fill-accent/10 stroke-accent" />
        <text x={82} y={82} className="fill-accent text-[22px] tabular-nums">{stepRoof}</text>
        <Label x={136} y={70} className="fill-text">of the bandwidth roof,</Label>
        <Label x={136} y={86} className="fill-text">whole batch-1 step</Label>
      </motion.g>
      <Label x={20} y={384}>dots are measured per-op points from the profiled decode step</Label>
    </Frame>
  );
}

function Key({ x, y, children, anchor, className = "fill-text", size = 14 }: { x: number; y: number; children: React.ReactNode; anchor?: "end" | "middle"; className?: string; size?: 12 | 14 }) {
  return (
    <text x={x} y={y} textAnchor={anchor} className={`${className} ${size === 14 ? "text-[14px]" : "text-[12px]"}`}>
      {children}
    </text>
  );
}

const laneLeft = 24;
const cellPitch = 16;
const cellX = (row: number) => laneLeft + row * cellPitch;
const barLeft = 150;
const barWidth = 300;

export function BucketsFigure() {
  const { on, grow } = useLoop(14);
  const { rows, maxBatch, ladders, costPct } = bucketScene;
  const slowest = Math.max(...ladders.map((l) => l.ms));
  return (
    <Frame label={`${rows} rows at max batch ${maxBatch}. ${ladders.map((l) => `With ${l.steps} ${l.steps === 1 ? "graph" : "graphs"} per power of two the graphs are ${l.sizes.join(", ")}, the step replays the graph for ${l.size} with ${l.size - rows} padded rows and takes ${l.ms} ms per token`).join(". ")}. Padding to ${ladders[0].size} costs about ${costPct.toFixed(0)}% per token`}>
      {ladders.map((l, k) => {
        const y = 36 + k * 124;
        const t = 0.03 + k * 0.27;
        return (
          <g key={l.steps}>
            <Key x={laneLeft} y={y - 10}>{`${l.steps} ${l.steps === 1 ? "graph" : "graphs"} per power of two: ${l.sizes.length} graphs`}</Key>
            {Array.from({ length: maxBatch }, (_, i) => (
              <rect key={i} x={cellX(i) + 1} y={y} width={cellPitch - 2} height={22} rx={2} className="fill-baseline/25" />
            ))}
            {l.sizes.map((b, i) => (
              <motion.g key={b} {...on(t + i * 0.006)}>
                <line x1={cellX(b)} y1={y + 24} x2={cellX(b)} y2={y + 30} className={b === l.size ? "stroke-accent" : "stroke-muted"} />
                <Key size={12} x={cellX(b) - (b === 1 ? 6 : 0)} y={y + 44} anchor={b === 1 ? undefined : "end"} className={b === l.size ? "fill-accent" : "fill-muted"}>{b}</Key>
              </motion.g>
            ))}
            {Array.from({ length: rows }, (_, i) => (
              <motion.rect key={i} x={cellX(i) + 1} y={y} width={cellPitch - 2} height={22} rx={2} className="fill-accent" {...on(t + 0.05 + i * 0.006)} />
            ))}
            <motion.rect x={cellX(0) - 2} y={y - 3} width={l.size * cellPitch + 4} height={28} rx={4} className="fill-none stroke-accent" strokeDasharray="4 3" {...on(t + 0.15)} />
            {Array.from({ length: l.size - rows }, (_, i) => (
              <motion.rect key={i} x={cellX(rows + i) + 1} y={y} width={cellPitch - 2} height={22} rx={2} className="fill-series-2" {...on(t + 0.17 + i * 0.006)} />
            ))}
            <motion.g {...on(t + 0.22)}>
              <Key x={laneLeft} y={y + 68}>{`${rows} rows replay the graph for ${l.size}: ${l.size - rows} padded rows thrown away`}</Key>
            </motion.g>
          </g>
        );
      })}
      <Label x={laneLeft} y={292}>measured ms per token</Label>
      {ladders.map((l, k) => {
        const y = 300 + k * 22;
        const t = 0.58 + k * 0.05;
        return (
          <g key={l.steps}>
            <Key x={laneLeft} y={y + 11} className={k ? "fill-accent" : "fill-text"}>{`graph for ${l.size}`}</Key>
            <motion.rect x={barLeft} y={y + 1} width={(barWidth * l.ms) / slowest} height={10} rx={3} className={k ? "fill-accent" : "fill-baseline"} {...grow(t, t + 0.08)} />
            <motion.g {...on(t + 0.06)}>
              <Key x={barLeft + (barWidth * l.ms) / slowest + 8} y={y + 11} className={k ? "fill-accent" : "fill-text"}>{`${l.ms.toFixed(2)} ms`}</Key>
            </motion.g>
          </g>
        );
      })}
      <motion.g {...on(0.72)}>
        <Key x={laneLeft} y={362}>{`padding to ${ladders[0].size} costs about ${costPct.toFixed(0)}% per token against ${ladders[1].size} (calculated)`}</Key>
      </motion.g>
      <rect x={laneLeft} y={376} width={10} height={10} rx={2} className="fill-accent" />
      <Label x={laneLeft + 16} y={385}>{`${rows} decode rows`}</Label>
      <rect x={laneLeft + 120} y={376} width={10} height={10} rx={2} className="fill-series-2" />
      <Label x={laneLeft + 136} y={385}>padded rows</Label>
      <Label x={536} y={385} anchor="end">{`max batch ${maxBatch}`}</Label>
    </Frame>
  );
}

const sweepTop = Math.ceil(Math.max(...sweepScene.points.map((p) => p.tokens)) / 500) * 500;
const sweepX = (i: number) => 96 + i * 104;
const sweepY = (tokens: number) => 232 - (tokens / sweepTop) * 180;
const shareBase = 348;
const shareHeight = (share: number) => (share / 100) * 84;
const count = (n: number) => n.toLocaleString("en-US");

export function SweepFigure() {
  const { on, grow, reduced } = useLoop(14);
  const clock = useClock(14);
  const drawn = useTransform(clock, [0, 0.06, 0.4, 1], [0, 0, 1, 1]);
  const ink = useTransform(clock, [0, 0.04, 0.06, 0.94, 0.97, 1], [0, 0, 1, 1, 0, 0]);
  const { points, knee, gains, toKnee } = sweepScene;
  const kneeAt = points.findIndex((p) => p.batch === knee);
  const after = gains.slice(kneeAt + 1).map((g) => `+${((g - 1) * 100).toFixed(0)}%`);
  const path = points.map((p, i) => `${i ? "L" : "M"}${sweepX(i)},${sweepY(p.tokens)}`).join(" ");
  const profiled = points.filter((p) => p.share !== undefined);
  return (
    <Frame label={`Output tokens per second with graphs against max batch: ${points.map((p) => `${count(p.tokens)} at ${p.batch}`).join(", ")}. The curve flattens after ${knee}. Attention's share of a step's GPU time: ${profiled.map((p) => `${p.share}% at ${p.batch}`).join(", ")}`}>
      <Label x={12} y={28}>tok/s, with graphs, measured</Label>
      <line x1={60} y1={232} x2={536} y2={232} className="stroke-border" />
      <line x1={60} y1={44} x2={60} y2={232} className="stroke-border" />
      {Array.from({ length: Math.floor(sweepTop / 1000) + 1 }, (_, i) => i * 1000).map((v) => (
        <g key={v}>
          <line x1={55} y1={sweepY(v)} x2={60} y2={sweepY(v)} className="stroke-border" />
          <Label x={52} y={sweepY(v) + 4} anchor="end">{count(v)}</Label>
        </g>
      ))}
      <motion.g {...on(0.36)}>
        <line x1={sweepX(kneeAt)} y1={sweepY(points[kneeAt].tokens) + 8} x2={sweepX(kneeAt)} y2={shareBase - shareHeight(points[kneeAt].share ?? 0) - 22} strokeDasharray="3 4" className="stroke-accent" />
      </motion.g>
      <motion.path d={path} fill="none" strokeWidth={2} strokeLinejoin="round" className="stroke-accent" style={{ pathLength: drawn, opacity: reduced ? 1 : ink }} />
      {points.map((p, i) => (
        <motion.g key={p.batch} {...on(0.06 + i * 0.085)}>
          <circle cx={sweepX(i)} cy={sweepY(p.tokens)} r={4} className="fill-accent" />
          <Key x={sweepX(i)} y={sweepY(p.tokens) + (i < kneeAt ? 22 : -10)} anchor={i < kneeAt ? undefined : "middle"}>{count(p.tokens)}</Key>
        </motion.g>
      ))}
      <motion.g {...on(0.46)}>
        <rect x={72} y={46} width={196} height={46} rx={4} className="fill-accent/10 stroke-accent" />
        <Key size={12} x={82} y={65}>{`${toKnee.toFixed(2)}x from ${points[0].batch} to ${knee},`}</Key>
        <Key size={12} x={82} y={82}>{`then ${after.join(" and ")} per doubling`}</Key>
      </motion.g>
      <Label x={12} y={262}>{"attention, share of a step's GPU time"}</Label>
      <line x1={60} y1={shareBase} x2={536} y2={shareBase} className="stroke-border" />
      {points.map((p, i) => (
        <g key={p.batch}>
          <Key size={12} x={sweepX(i)} y={shareBase + 16} anchor="middle" className={p.batch === knee ? "fill-accent" : "fill-muted"}>{p.batch}</Key>
          {p.share !== undefined && (
            <>
              <motion.rect x={sweepX(i) - 20} y={shareBase - shareHeight(p.share)} width={40} height={shareHeight(p.share)} rx={2} className="fill-series-2" {...grow(0.5 + i * 0.04, 0.6 + i * 0.04, "y")} />
              <motion.g {...on(0.58 + i * 0.04)}>
                <Key x={sweepX(i)} y={shareBase - shareHeight(p.share) - 6} anchor="middle">{`${p.share}%`}</Key>
              </motion.g>
            </>
          )}
        </g>
      ))}
      <Label x={12} y={shareBase + 16}>max batch</Label>
      <Label x={12} y={388}>{`share measured in eager profiles, max batch ${profiled[0].batch} and up`}</Label>
    </Frame>
  );
}

const specTokens = [...spec.example.prompt_tokens, ...spec.example.output_tokens];
const specTop = 100;
const specPitch = 25;

export function SpeculateFigure() {
  const { on, move } = useLoop(14);
  const s = spec.missStep;
  const lead = Math.min(2, s.match);
  const found = specTokens.slice(s.match - lead, s.match + s.ngram + s.draft.length);
  const cw = Math.min(43, 520 / found.length);
  const rows = 1 + s.draft.length;
  const rowY = (i: number) => specTop + i * specPitch;
  const guessX = (i: number) => 20 + (lead + s.ngram + i) * cw;
  return (
    <Frame label={`One decode step of one request with ${spec.maxBatch} rows. The last ${s.ngram} tokens appeared earlier in the context, so the ${s.draft.length} tokens after them fill rows 1 to ${s.draft.length} as guesses. The model agrees with ${s.kept}; the first wrong guess and the rows after it are dropped, and the step writes ${s.kept + 1} tokens`}>
      <Label x={20} y={22} className="fill-text">one decode step: guesses fill the spare rows</Label>
      <Label x={20} y={40}>earlier in the context</Label>
      {found.map((t, i) => (
        <motion.g key={i} {...on(0.03)}>
          <Chip x={20 + i * cw} y={46} w={cw} text={short(t)} tone={i < lead ? "plain" : i < lead + s.ngram ? "pattern" : "draft"} />
        </motion.g>
      ))}
      <motion.g {...on(0.08)}>
        <Label x={20 + lead * cw} y={86} className="fill-accent">{`the last ${s.ngram} tokens`}</Label>
        <Label x={540} y={86} anchor="end" className="fill-warn">{`the ${s.draft.length} after them: copied as guesses`}</Label>
      </motion.g>
      {Array.from({ length: rows }, (_, i) => (
        <Label key={i} x={20} y={rowY(i) + 16}>{`row ${i}`}</Label>
      ))}
      <motion.g {...on(0.12)}>
        <Chip x={64} y={rowY(0)} w={cw} text={short(specTokens[s.at - 1])} tone="real" />
      </motion.g>
      {s.draft.map((t, i) => (
        <motion.g key={i} {...move(0.18 + i * 0.025, 0.3 + i * 0.025, 64 - guessX(i), rowY(i + 1) - 46, false, 0.62)}>
          <Chip x={guessX(i)} y={46} w={cw} text={short(t)} tone="draft" />
        </motion.g>
      ))}
      {s.draft.map((t, i) => (
        <motion.g key={i} {...on(0.62)}>
          <Chip x={64} y={rowY(i + 1)} w={cw} text={short(t)} tone={i < s.kept ? "pattern" : "dropped"} />
        </motion.g>
      ))}
      {Array.from({ length: rows }, (_, i) => {
        const kept = i <= s.kept;
        const t = kept ? 0.44 + i * 0.05 : 0.62;
        return (
          <motion.g key={i} {...on(t)}>
            {kept && (
              <text x={118} y={rowY(i) + 16} className="fill-muted text-[11px]">
                →
              </text>
            )}
            {kept && <Chip x={134} y={rowY(i)} w={cw} text={short(specTokens[s.at + i])} tone="kept" />}
            <Label x={186} y={rowY(i) + 16} className={kept && i < s.kept ? "fill-accent" : kept && i === s.draft.length ? "fill-muted" : "fill-warn"}>
              {!kept ? "dropped: built on the wrong guess" : i === s.draft.length ? "no guess left to check" : i < s.kept ? `same as guess ${i + 1}: keep it` : `guess ${i + 1} was “${short(s.draft[i])}”: stop`}
            </Label>
          </motion.g>
        );
      })}
      {rows < spec.maxBatch && (
        <motion.g {...on(0.12)}>
          <rect x={20} y={rowY(rows) + 4} width={520} height={20} rx={3} className="fill-none stroke-border" strokeDasharray="3 2" />
          <Label x={30} y={rowY(rows) + 18}>{`rows ${rows} to ${spec.maxBatch - 1}: left empty, at most ${spec.maxDraft} guesses a step`}</Label>
        </motion.g>
      )}
      <motion.g {...on(0.74)}>
        <Label x={20} y={372} className="fill-accent">{`${s.kept + 1} tokens out of one step; ${specFmt(spec.best.tokensPerStep, 2)} a step on prompts that repeat (measured)`}</Label>
      </motion.g>
      <Label x={20} y={392}>{`step ${s.index} of a run recorded ${spec.date}, one request`}</Label>
    </Frame>
  );
}

export function RankFigure() {
  const { on, grow } = useLoop(16);
  const { n, pct } = rankScene;
  const [qkv, wo] = attentionMatrices;
  const px = 520 / rankShare(1).total;
  const s = rankShare(n);
  return (
    <Frame label={`One layer's fused QKV weight is cut by rows and the output projection by columns for ${n} ranks, so each rank gets ${s.qHeads} of ${tpShape.heads} query heads. The embedding and the LM head stay whole in every rank, so a rank of ${n} holds ${pct}% of the model, not half`}>
      <Label x={20} y={22} className="fill-text">{`one layer's attention weights, cut for ${n} ranks`}</Label>
      {Array.from({ length: n }, (_, r) => (
        <motion.g key={r} {...on(0.08 + (r * 0.84) / n, 0.08 + ((r + 1) * 0.84) / n, false)}>
          <Label x={540} y={22} anchor="end" className="fill-accent">{`rank ${r} of ${n}: ${s.qHeads} heads`}</Label>
        </motion.g>
      ))}
      <Matrix m={qkv} x={20} y={74} w={180} h={120} n={n} on={on} />
      <Matrix m={wo} x={290} y={74} w={180} h={120} n={n} on={on} />
      <Label x={20} y={218}>rows of one matrix, then the columns of the next</Label>
      <motion.g {...on(0.04)}>
        <Label x={20} y={248} className="fill-text">the whole model</Label>
        <RankBar n={1} x={20} y={254} h={18} px={px} />
      </motion.g>
      <motion.g {...grow(0.2, 0.36)}>
        <RankBar n={n} x={20} y={306} h={18} px={px} />
      </motion.g>
      <motion.g {...on(0.36)}>
        <Label x={20} y={300} className="fill-accent">{`a rank of ${n}: ${pct}% of the model, not half`}</Label>
        <line x1={280} y1={286} x2={280} y2={330} strokeDasharray="3 3" className="stroke-text" />
        <Label x={284} y={340}>half</Label>
      </motion.g>
      <rect x={20} y={354} width={10} height={10} rx={1} className="fill-accent" />
      <Label x={36} y={363}>{`layers, cut in ${n}`}</Label>
      <rect x={140} y={354} width={10} height={10} rx={1} className="fill-text/70" />
      <Label x={156} y={363}>embedding, LM head, norms: whole in every rank</Label>
      <Label x={20} y={388}>bars to scale from the shapes; the percent from the MiB on the device</Label>
    </Frame>
  );
}

const shardBlocks = [
  { name: "attention", what: `${rankShare(2).qHeads} of ${tpShape.heads} heads`, x: 20, t: 0.06 },
  { name: "MLP", what: `${tpFmt(rankShare(2).channels)} channels`, x: 290, t: 0.44 },
];
const shardLane = [96, 96, 28];

export function ShardsFigure() {
  const { on, move } = useLoop(16);
  const f = shardedForward();
  const two = f.ranks.indexOf(2);
  return (
    <Frame label={`One layer with 2 ranks on one GPU: each rank in turn computes a partial result with its own weights, and the two partials are summed into the residual stream, after attention and after the MLP. ${tpShape.sums} sums in a token. A token took ${tpFmt(f.msPerToken[two], 2)} ms with 2 ranks against ${tpFmt(f.wholeMsPerToken, 2)} ms for the whole model (measured)`}>
      <Label x={20} y={22} className="fill-text">one layer, 2 ranks taking turns on one GPU</Label>
      <Label x={20} y={56}>residual stream</Label>
      <line x1={20} y1={70} x2={540} y2={70} strokeWidth={2} className="stroke-border" />
      {shardBlocks.map((b) => {
        const node = b.x + 214;
        const sum = b.t + 0.26;
        return (
          <g key={b.name}>
            <Label x={b.x} y={100} className="fill-text">{b.name}</Label>
            {[0, 1].map((r) => {
              const y = 110 + r * 46;
              return (
                <g key={r}>
                  <rect x={b.x} y={y} width={150} height={34} rx={3} className="fill-text/10 stroke-border" />
                  <motion.rect x={b.x} y={y} width={150} height={34} rx={3} strokeWidth={1.5} className="fill-accent/25 stroke-accent" {...on(b.t + r * 0.1)} />
                  <Label x={b.x + 10} y={y + 21} className="fill-text">{`rank ${r}: ${b.what}`}</Label>
                  <motion.rect x={b.x + 164} y={y} width={10} height={34} rx={2} className="fill-accent" {...on(b.t + r * 0.1 + 0.06)} />
                  <motion.rect x={b.x + 164} y={y} width={10} height={34} rx={2} className="fill-accent" {...move(sum, sum + 0.07, node - 5 - b.x - 164, 53 - y, false, sum + 0.1)} />
                </g>
              );
            })}
            <motion.g {...on(b.t + 0.16)}>
              <Label x={b.x + 169} y={216} anchor="middle">partials</Label>
            </motion.g>
            <circle cx={node} cy={70} r={11} className="fill-surface stroke-border" />
            <motion.circle cx={node} cy={70} r={11} strokeWidth={1.5} className="fill-accent/25 stroke-accent" {...on(sum + 0.08)} />
            <motion.line x1={node + 11} y1={70} x2={b.x === 20 ? shardBlocks[1].x + 203 : 540} y2={70} strokeWidth={2} className="stroke-accent" {...on(sum + 0.08)} />
            <text x={node} y={75} textAnchor="middle" className="fill-text text-[14px]">
              +
            </text>
          </g>
        );
      })}
      <Label x={20} y={250}>the one GPU, in time</Label>
      {shardBlocks.map((b, k) =>
        shardLane.map((w, i) => {
          const x = 20 + k * 236 + shardLane.slice(0, i).reduce((a, c) => a + c + 4, 0);
          const sum = i === 2;
          return (
            <g key={`${k}-${i}`}>
              <rect x={x} y={258} width={w} height={24} rx={3} className="fill-text/10" />
              <motion.rect x={x} y={258} width={w} height={24} rx={3} strokeWidth={1.5} className="fill-accent/25 stroke-accent" {...on(sum ? b.t + 0.34 : b.t + i * 0.1)} />
              <text x={x + w / 2} y={274} textAnchor="middle" className="fill-text text-[11px]">
                {sum ? "+" : `rank ${i}`}
              </text>
            </g>
          );
        }),
      )}
      {shardBlocks.map((b, k) => (
        <Label key={b.name} x={20 + k * 236 + 112} y={298} anchor="middle">{b.name}</Label>
      ))}
      <Label x={20} y={328} className="fill-text">{`${tpShape.sums} sums in a token: 2 in each of ${tpShape.layers} layers`}</Label>
      <Label x={20} y={348} className="fill-accent">{`a token: ${tpFmt(f.msPerToken[two], 2)} ms with 2 ranks in turn, ${tpFmt(f.wholeMsPerToken, 2)} ms whole (measured)`}</Label>
      <Label x={20} y={368}>{`logits within ${f.maxLogitDiff}, every greedy token the same (measured)`}</Label>
      <Label x={20} y={390}>one GPU: the ranks take turns, so it cannot be faster; the lane is not to scale</Label>
    </Frame>
  );
}
