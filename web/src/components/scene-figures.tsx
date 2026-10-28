"use client";

import { animate, motion, type MotionValue, useMotionValue, useTransform } from "motion/react";
import { useEffect } from "react";
import { metric } from "@/components/gpu-floorplan";
import { Frame, Label } from "@/components/kernels/frame";
import { useLoop } from "@/components/kernels/loop";
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
