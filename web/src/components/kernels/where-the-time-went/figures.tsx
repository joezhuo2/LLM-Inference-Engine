"use client";

import { motion } from "motion/react";
import { eagerStage, load, maxSpan, next, nextItems, prefill, prefillLengths, prefillPaths, prefillPerToken, smallOf, stepStages, timeFmt, token, tokenParts } from "@/content/kernels/where-the-time-went";
import { Frame, Label } from "../frame";
import { useLoop } from "../loop";

const fmt = timeFmt;
const before = (values: number[]) => values.map((_, i) => values.slice(0, i).reduce((a, b) => a + b, 0));

export function ProfileFigure() {
  const { on, grow } = useLoop(14);
  const s = eagerStage;
  const tpotUs = (s.tpot ?? 0) * 1e3;
  const x0 = 20;
  const w = 520;
  const px = w / tpotUs;
  const boxes = [44, 26, 9, 5, 13, 22, 4, 60, 5, 34];
  const laneStarts = before(boxes.map((b) => b + 5));
  const lane = boxes.map((b, i) => ({ x: x0 + laneStarts[i] + 5, w: b }));
  const cursor = x0 + boxes.reduce((a, b) => a + b + 5, 0);
  const rows = [
    { label: "time per token", value: `${fmt(s.tpot ?? 0, 2)} ms`, us: tpotUs, cls: "fill-muted/50", text: "fill-text" },
    { label: "span: first kernel start to last kernel end", value: `${fmt(s.span)} µs`, us: s.span, cls: "fill-warn/70", text: "fill-warn" },
    { label: "busy: some kernel is running", value: `${fmt(s.busy)} µs`, us: s.busy, cls: "fill-accent", text: "fill-accent" },
  ];
  const parts = [
    { label: "GEMMs", us: s.groups.gemm, cls: "fill-accent" },
    { label: "attention", us: s.groups.attention, cls: "fill-text/70" },
    { label: "the small kernels", us: smallOf(s), cls: "fill-warn" },
  ];
  const kpx = w / s.kernelSum;
  const partStarts = before(parts.map((p) => p.us * kpx));
  return (
    <Frame label={`One batch-1 decode step launched one by one: ${fmt(s.tpot ?? 0, 2)} ms per token, a span of ${fmt(s.span)} µs from the first kernel to the last, ${fmt(s.busy)} µs of it with a kernel running, and the kernel time split by operation: GEMMs ${fmt(s.groups.gemm)} µs, attention ${fmt(s.groups.attention)} µs, the small kernels ${fmt(smallOf(s))} µs`}>
      <Label x={20} y={24} className="fill-text">
        one decode step on the GPU: kernels, and a gap before each
      </Label>
      <line x1={x0} y1={66} x2={x0 + w} y2={66} className="stroke-border" />
      {lane.map((b, i) => (
        <motion.rect key={i} x={b.x} y={42} width={b.w} height={22} rx={2} className={i % 10 === 4 ? "fill-text/70" : i === 2 || i === 3 || i === 6 || i === 8 ? "fill-warn" : "fill-accent"} {...on(0.02 + i * 0.03)} />
      ))}
      <Label x={cursor + 10} y={58}>{`… ${s.kernels} kernels`}</Label>
      <Label x={20} y={84}>{`a sketch of one layer; ${s.idlePct}% of the span is gaps (measured)`}</Label>
      {rows.map((r, i) => (
        <g key={r.label}>
          <Label x={20} y={122 + i * 46} className={r.text}>
            {r.label}
          </Label>
          <Label x={540} y={122 + i * 46} anchor="end" className={r.text}>
            {r.value}
          </Label>
          <rect x={x0} y={130 + i * 46} width={w} height={12} rx={2} className="fill-border" />
          <motion.rect x={x0} y={130 + i * 46} width={r.us * px} height={12} rx={2} className={r.cls} {...grow(0.3 + i * 0.08, 0.3 + i * 0.08 + 0.18)} />
        </g>
      ))}
      <Label x={20} y={270} className="fill-text">
        kernel time by the operation that launched it
      </Label>
      <motion.g {...grow(0.6, 0.82)}>
        {parts.map((p, i) => (
          <rect key={p.label} x={x0 + partStarts[i]} y={280} width={Math.max(1, p.us * kpx - 1)} height={16} className={p.cls} />
        ))}
      </motion.g>
      {parts.map((p, i) => (
        <motion.g key={p.label} {...on(0.62 + i * 0.05)}>
          <rect x={20 + i * 176} y={312} width={10} height={10} rx={2} className={p.cls} />
          <Label x={36 + i * 176} y={321}>{`${p.label} ${fmt(p.us)} µs`}</Label>
        </motion.g>
      ))}
      <Label x={20} y={352}>{`time per op is a mean over the steps, span and busy are medians:`}</Label>
      <Label x={20} y={368}>{`the ops add up to ${fmt(s.kernelSum)} µs against ${fmt(s.busy)} µs busy`}</Label>
    </Frame>
  );
}

export function StagesFigure() {
  const { on, grow } = useLoop(16);
  const x0 = 20;
  const w = 300;
  const px = w / maxSpan;
  return (
    <Frame label={`The batch-1 decode step after each change: ${stepStages.map((s) => `${s.title} (${s.date}): GEMMs ${fmt(s.groups.gemm)}, attention ${fmt(s.groups.attention)}, small kernels ${fmt(smallOf(s))}, gaps ${fmt(s.idle)} µs`).join("; ")}`}>
      <Label x={20} y={24} className="fill-text">
        the same step after each change, µs (measured)
      </Label>
      <Label x={366} y={44} anchor="end" className="fill-accent">
        GEMMs
      </Label>
      <Label x={424} y={44} anchor="end" className="fill-text">
        attention
      </Label>
      <Label x={478} y={44} anchor="end" className="fill-warn">
        small
      </Label>
      <Label x={540} y={44} anchor="end">
        gaps
      </Label>
      {stepStages.map((s, i) => {
        const y = 62 + i * 52;
        const parts = [
          { us: s.groups.gemm, cls: "fill-accent" },
          { us: s.groups.attention, cls: "fill-text/70" },
          { us: smallOf(s), cls: "fill-warn" },
          { us: s.idle, cls: "fill-none stroke-muted" },
        ];
        const starts = before(parts.map((p) => p.us * px));
        return (
          <g key={s.id}>
            <motion.g {...on(0.02 + i * 0.1)}>
              <Label x={20} y={y} className={s.replay ? "fill-accent" : "fill-text"}>{`${s.replay ? "" : `${s.date} · `}${s.title}`}</Label>
              <Label x={366} y={y + 20} anchor="end" className="fill-accent">
                {fmt(s.groups.gemm)}
              </Label>
              <Label x={424} y={y + 20} anchor="end" className="fill-text">
                {fmt(s.groups.attention)}
              </Label>
              <Label x={478} y={y + 20} anchor="end" className="fill-warn">
                {fmt(smallOf(s))}
              </Label>
              <Label x={540} y={y + 20} anchor="end">
                {fmt(s.idle)}
              </Label>
            </motion.g>
            <motion.g {...grow(0.04 + i * 0.1, 0.04 + i * 0.1 + 0.12)}>
              {parts.map((p, k) => (
                <rect key={k} x={x0 + starts[k]} y={y + 9} width={Math.max(1, p.us * px - 1)} height={13} strokeDasharray={k === 3 ? "2 2" : undefined} className={p.cls} />
              ))}
            </motion.g>
          </g>
        );
      })}
      <Label x={20} y={384}>{`dashed: launch gaps; the bar is the ops plus the gaps, ${fmt(maxSpan)} µs at its longest`}</Label>
    </Frame>
  );
}

const partTone: Record<string, { fill: string; text: string }> = {
  floor: { fill: "fill-accent", text: "fill-accent" },
  above: { fill: "fill-text/70", text: "fill-text" },
  gaps: { fill: "fill-muted/50", text: "fill-muted" },
  host: { fill: "fill-warn", text: "fill-warn" },
};

export function FloorFigure() {
  const { on, grow } = useLoop(13);
  const x0 = 20;
  const w = 520;
  const px = w / token.ms;
  const starts = before(tokenParts.map((p) => p.ms * px));
  return (
    <Frame label={`One batch-1 token today takes ${fmt(token.ms, 2)} ms: ${tokenParts.map((p) => `${p.label} ${fmt(p.ms, 2)} ms`).join(", ")}. The floor is ${fmt(token.weightGb, 2)} GB of weights divided by ${token.peakGbs} GB/s, ${fmt(token.floorPct)}% of the token`}>
      <Label x={20} y={24} className="fill-text">
        the floor: every weight is read once per token
      </Label>
      <Label x={20} y={52}>{`${fmt(token.weightGb, 2)} GB of weights ÷ ${token.peakGbs} GB/s measured peak`}</Label>
      <Label x={20} y={72} className="fill-accent">{`= ${fmt(token.floorMs, 2)} ms, whatever the kernels do (calculated)`}</Label>
      <Label x={20} y={118} className="fill-text">{`one token today, replayed from a graph: ${fmt(token.ms, 2)} ms (measured)`}</Label>
      <rect x={x0} y={130} width={w} height={26} rx={3} className="fill-border" />
      <motion.g {...grow(0.1, 0.5)}>
        {tokenParts.map((p, i) => (
          <rect key={p.id} x={x0 + starts[i]} y={130} width={Math.max(1, p.ms * px - 1)} height={26} className={partTone[p.id].fill} />
        ))}
      </motion.g>
      {tokenParts.map((p, i) => (
        <motion.g key={p.id} {...on(0.16 + i * 0.1)}>
          <rect x={20} y={182 + i * 34} width={12} height={12} rx={2} className={partTone[p.id].fill} />
          <Label x={42} y={192 + i * 34} className={partTone[p.id].text}>
            {p.label}
          </Label>
          <Label x={440} y={192 + i * 34} anchor="end" className={partTone[p.id].text}>{`${fmt(p.ms, 2)} ms`}</Label>
          <Label x={540} y={192 + i * 34} anchor="end" className={partTone[p.id].text}>{`${fmt((p.ms / token.ms) * 100, p.ms / token.ms < 0.02 ? 1 : 0)}%`}</Label>
        </motion.g>
      ))}
      <line x1={20} y1={322} x2={540} y2={322} className="stroke-border" />
      <Label x={20} y={346}>{`the first profile's token took ${token.firstMs === null ? "longer" : `${fmt(token.firstMs, 2)} ms`}; the floor was the same`}</Label>
      <Label x={20} y={364}>only fewer bytes or more bandwidth move the blue part</Label>
    </Frame>
  );
}

const pathTone: Record<string, { fill: string; text: string }> = {
  paged: { fill: "fill-baseline", text: "fill-muted" },
  first: { fill: "fill-warn", text: "fill-warn" },
  tiled: { fill: "fill-accent", text: "fill-accent" },
  llama: { fill: "fill-text/70", text: "fill-text" },
};

export function PrefillFigure() {
  const { on, grow } = useLoop(13);
  const last = prefillLengths.length - 1;
  const max = Math.max(...prefillPaths.map((p) => p.ms[last]));
  const maxPer = Math.max(...prefillPerToken.flat());
  const shown = ["paged", "tiled", "llama"];
  return (
    <Frame label={`Time to first token of one ${prefill.long}-token prompt: ${prefillPaths.map((p) => `${p.label} ${fmt(p.ms[last], 1)} ms`).join(", ")}`}>
      <Label x={20} y={24} className="fill-text">{`one ${prefill.long}-token prompt alone, ms to the first token (measured)`}</Label>
      {prefillPaths.map((p, i) => (
        <g key={p.id}>
          <Label x={20} y={56 + i * 30} className={pathTone[p.id].text}>
            {p.label}
          </Label>
          <rect x={210} y={46 + i * 30} width={250} height={12} rx={2} className="fill-border" />
          <motion.rect x={210} y={46 + i * 30} width={(250 * p.ms[last]) / max} height={12} rx={2} className={pathTone[p.id].fill} {...grow(0.05 + i * 0.07, 0.05 + i * 0.07 + (0.3 * p.ms[last]) / max)} />
          <Label x={540} y={56 + i * 30} anchor="end" className={pathTone[p.id].text}>{`${fmt(p.ms[last], 1)} ms`}</Label>
        </g>
      ))}
      <line x1={20} y1={176} x2={540} y2={176} className="stroke-border" />
      <Label x={20} y={200} className="fill-text">
        ms per prompt token against prompt length (calculated from the medians)
      </Label>
      {prefillLengths.map((n, k) => (
        <g key={n}>
          <Label x={96 + k * 122} y={372} anchor="middle">{`${n} tokens`}</Label>
          {shown.map((id, j) => {
            const i = prefillPaths.findIndex((p) => p.id === id);
            const h = (130 * prefillPerToken[i][k]) / maxPer;
            return (
              <motion.g key={id} {...on(0.4 + k * 0.08)}>
                <rect x={62 + k * 122 + j * 24} y={354 - h} width={20} height={h} rx={2} className={pathTone[id].fill} />
              </motion.g>
            );
          })}
        </g>
      ))}
      <line x1={40} y1={354} x2={540} y2={354} className="stroke-border" />
      <Label x={540} y={222} anchor="end" className="fill-muted">{`one walk per row: ${prefill.pagedPerTokenShort} to ${prefill.pagedPerTokenLong} ms`}</Label>
      <Label x={540} y={238} anchor="end" className="fill-accent">{`tiled: ${prefill.perTokenLow} to ${prefill.perTokenHigh} ms`}</Label>
    </Frame>
  );
}

const seriesTone: Record<string, { stroke: string; fill: string; text: string; dash?: string; width: number }> = {
  engine: { stroke: "stroke-baseline", fill: "fill-baseline", text: "fill-muted", width: 1.8 },
  chunked: { stroke: "stroke-accent", fill: "fill-accent", text: "fill-accent", width: 2.5 },
  vllm: { stroke: "stroke-warn", fill: "fill-warn", text: "fill-warn", dash: "5 4", width: 2 },
};

export function LoadFigure() {
  const { on } = useLoop(14);
  const rates = load.rates;
  const top = load.topRate;
  const gx0 = 44;
  const gx1 = 268;
  const y0 = 300;
  const y1 = 70;
  const gmax = Math.max(top, ...load.series.flatMap((s) => s.goodput));
  const GX = (q: number) => gx0 + ((gx1 - gx0) * q) / top;
  const GY = (v: number) => y0 - ((y0 - y1) * v) / gmax;
  const tx0 = 330;
  const tx1 = 540;
  const tails = load.series.flatMap((s) => s.ttftP99);
  const lo = Math.min(...tails, load.slo.ttft_ms) / 1.6;
  const hi = Math.max(...tails, load.slo.ttft_ms) * 1.4;
  const TX = (q: number) => tx0 + ((tx1 - tx0) * (q - rates[0])) / (top - rates[0]);
  const TY = (ms: number) => y0 - ((y0 - y1) * Math.log(ms / lo)) / Math.log(hi / lo);
  return (
    <Frame label={`Goodput at the SLO and TTFT p99 against offered load, measured ${load.date} at commit ${load.commit}: ${load.series.map((s) => `${s.label} goodput ${s.goodput.map((v) => fmt(v, 2)).join(", ")} and TTFT p99 ${s.ttftP99.map((v) => fmt(v)).join(", ")} ms`).join("; ")} at ${rates.join(", ")} requests per second`}>
      <Label x={20} y={24} className="fill-text">{`under load, ${load.date}, commit ${load.commit} (measured)`}</Label>
      <Label x={gx0} y={52}>goodput, requests/s within the SLO</Label>
      <Label x={tx0} y={52}>TTFT p99, ms (log)</Label>
      <line x1={gx0} y1={y0} x2={gx1} y2={y0} className="stroke-border" />
      <line x1={gx0} y1={y0} x2={gx0} y2={y1} className="stroke-border" />
      <line x1={GX(0)} y1={GY(0)} x2={GX(top)} y2={GY(top)} className="stroke-border" strokeDasharray="2 4" />
      <Label x={GX(top) - 4} y={GY(top) + 16} anchor="end">
        every request
      </Label>
      {rates.map((q) => (
        <g key={q}>
          <Label x={GX(q)} y={y0 + 16} anchor="middle">
            {q}
          </Label>
          <Label x={TX(q)} y={y0 + 16} anchor="middle">
            {q}
          </Label>
        </g>
      ))}
      <Label x={gx0 - 6} y={GY(gmax) + 4} anchor="end">
        {fmt(gmax, 0)}
      </Label>
      <Label x={gx0 - 6} y={y0 + 4} anchor="end">
        0
      </Label>
      <line x1={tx0} y1={y0} x2={tx1} y2={y0} className="stroke-border" />
      <line x1={tx0} y1={y0} x2={tx0} y2={y1} className="stroke-border" />
      <line x1={tx0} y1={TY(load.slo.ttft_ms)} x2={tx1} y2={TY(load.slo.ttft_ms)} className="stroke-text" strokeDasharray="2 4" />
      <Label x={tx1} y={TY(load.slo.ttft_ms) - 5} anchor="end" className="fill-text">{`SLO ${fmt(load.slo.ttft_ms)} ms`}</Label>
      {load.series.map((s, i) => {
        const t = seriesTone[s.id] ?? seriesTone.engine;
        return (
          <motion.g key={s.id} {...on(0.05 + i * 0.16)}>
            <polyline points={s.goodput.map((v, k) => `${GX(rates[k])},${GY(v)}`).join(" ")} className={`fill-none ${t.stroke}`} strokeWidth={t.width} strokeDasharray={t.dash} />
            {s.goodput.map((v, k) => (
              <circle key={k} cx={GX(rates[k])} cy={GY(v)} r={2.6} className={t.fill} />
            ))}
            <polyline points={s.ttftP99.map((v, k) => `${TX(rates[k])},${TY(v)}`).join(" ")} className={`fill-none ${t.stroke}`} strokeWidth={t.width} strokeDasharray={t.dash} />
            {s.ttftP99.map((v, k) => (
              <circle key={k} cx={TX(rates[k])} cy={TY(v)} r={2.6} className={t.fill} />
            ))}
            <line x1={20 + i * 178} y1={350} x2={42 + i * 178} y2={350} className={t.stroke} strokeWidth={t.width} strokeDasharray={t.dash} />
            <Label x={48 + i * 178} y={354} className={t.text}>
              {s.label}
            </Label>
            <Label x={48 + i * 178} y={370} className={t.text}>{`best ${fmt(Math.max(...s.goodput), 2)} req/s`}</Label>
          </motion.g>
        );
      })}
      <Label x={(gx0 + gx1) / 2} y={y0 + 32} anchor="middle">
        offered requests/s
      </Label>
      <Label x={(tx0 + tx1) / 2} y={y0 + 32} anchor="middle">
        offered requests/s
      </Label>
    </Frame>
  );
}

export function NextFigure() {
  const { on, grow } = useLoop(14);
  return (
    <Frame label={`Three next steps for this engine: ${nextItems.map((n, i) => `${i + 1}. ${n.title}: ${n.evidence}; it would move ${n.moves}`).join(". ")}. Smaller: a budget of ${next.chunkBudget} costs ${next.chunkCostPct}% on a lone long prompt, and float Gumbel noise takes a sampled draw from ${fmt(next.gumbelUs)} µs to ${fmt(next.gumbelFloatLow)} to ${fmt(next.gumbelFloatHigh)} µs`}>
      <Label x={20} y={24} className="fill-text">
        three next steps, by what the measurements say they are worth
      </Label>
      {nextItems.map((n, i) => (
        <g key={n.id}>
          <motion.g {...on(0.04 + i * 0.16)}>
            <text x={20} y={64 + i * 96} className="fill-accent font-serif text-[22px]">
              {i + 1}
            </text>
            <Label x={44} y={58 + i * 96} className="fill-text">
              {n.title}
            </Label>
            <Label x={44} y={76 + i * 96}>{n.evidence}</Label>
            <Label x={44} y={112 + i * 96}>{`moves: ${n.moves}`}</Label>
            <Label x={540} y={96 + i * 96} anchor="end" className="fill-accent">
              {n.value}
            </Label>
          </motion.g>
          <rect x={44} y={86 + i * 96} width={440} height={10} rx={2} className="fill-border" />
          <motion.rect x={44} y={86 + i * 96} width={440 * n.bar} height={10} rx={2} className="fill-accent" {...grow(0.08 + i * 0.16, 0.08 + i * 0.16 + 0.12)} />
        </g>
      ))}
      <line x1={20} y1={336} x2={540} y2={336} className="stroke-border" />
      <Label x={20} y={358}>{`smaller: a budget of ${next.chunkBudget} costs a lone ${prefill.long}-token prompt ${next.chunkCostPct}%;`}</Label>
      <Label x={20} y={374}>{`float Gumbel noise: a sampled draw from ${fmt(next.gumbelUs)} µs to ${fmt(next.gumbelFloatLow)} to ${fmt(next.gumbelFloatHigh)} µs`}</Label>
    </Frame>
  );
}
