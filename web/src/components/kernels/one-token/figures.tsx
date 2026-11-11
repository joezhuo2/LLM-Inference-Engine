"use client";

import { motion } from "motion/react";
import { hostTimes, journey, layerOps, layerUs, layersUs, shareText, stage, stations, stepProfile, tokenFmt, tokenUs, trip } from "@/content/kernels/one-token";
import { Frame, Label } from "../frame";
import { useLoop } from "../loop";

const fmt = tokenFmt;
const X = 20;
const W = 520;
const us = (v: number) => `${fmt(v, v < 100 ? 1 : 0)} µs`;
const before = (values: number[]) => values.map((_, i) => values.slice(0, i).reduce((a, b) => a + b, 0));

export function PipelineFigure() {
  const { on, move } = useLoop(16);
  const row = 44;
  return (
    <Frame label={`The path of one token: ${stations.map((s) => `${s.label} (${s.where}) takes ${s.takes} and gives ${s.gives} in ${s.time}`).join("; ")}`}>
      <Label x={X} y={22} className="fill-text">{`the token "${journey.chosenText.trim()}", from a question to an ID and back to text`}</Label>
      <line x1={X + 6} y1={52} x2={X + 6} y2={52 + row * (stations.length - 1)} className="stroke-border" />
      <motion.circle cx={X + 6} cy={52} r={5} className="fill-warn" {...move(0.04, 0.9, 0, row * (stations.length - 1))} />
      {stations.map((s, i) => (
        <motion.g key={s.id} {...on(0.04 + i * 0.11)}>
          <rect x={X + 20} y={38 + i * row} width={118} height={28} rx={3} className={s.where === "host" ? "fill-border" : "fill-accent/25"} />
          <Label x={X + 28} y={56 + i * row} className="fill-text">
            {s.label}
          </Label>
          <Label x={X + 150} y={50 + i * row}>{`${s.takes} → ${s.gives}`}</Label>
          <Label x={X + 150} y={64 + i * row} className={s.where === "host" ? "fill-muted" : "fill-accent"}>{`${s.where} · ${s.time}`}</Label>
        </motion.g>
      ))}
      <Label x={X} y={372}>{`grey runs on the host, blue on the GPU; one decoded token takes ${fmt(stepProfile.tpotMs, 2)} ms (measured)`}</Label>
    </Frame>
  );
}

export function TokenizeFigure() {
  const { on } = useLoop(14);
  const perRow = 6;
  const chipW = 84;
  return (
    <Frame label={`The question becomes ${journey.promptCount} token IDs: ${journey.promptTokens.join(" ")}. Encoding takes ${us(hostTimes.encode)}, once per request.`}>
      <Label x={X} y={22} className="fill-text">{`"${journey.user}"`}</Label>
      <Label x={X} y={40}>{`the chat template wraps it, then SentencePiece cuts it into ${journey.promptCount} pieces`}</Label>
      {journey.promptTokens.map((t, i) => {
        const x = X + (i % perRow) * (chipW + 3);
        const y = 58 + Math.floor(i / perRow) * 34;
        return (
          <motion.g key={i} {...on(0.05 + i * 0.025)}>
            <rect x={x} y={y} width={chipW} height={26} rx={3} className="fill-accent/20" />
            <text x={x + chipW / 2} y={y + 17} textAnchor="middle" className="fill-text font-mono text-[11px]">
              {t === "\n" ? "\\n" : t}
            </text>
          </motion.g>
        );
      })}
      <motion.g {...on(0.66)}>
        <Label x={X} y={222} className="fill-text">
          measured on the host, CPU alone
        </Label>
        {[
          { label: "chat template", v: hostTimes.template },
          { label: `encode, ${journey.promptCount} tokens, once per request`, v: hostTimes.encode },
          { label: "one ID back to text, once per token", v: hostTimes.push },
          { label: "one decoded token, for scale", v: tokenUs },
        ].map((r, i) => (
          <g key={r.label}>
            <Label x={X} y={246 + i * 32}>{r.label}</Label>
            <Label x={X + W} y={246 + i * 32} anchor="end" className={i === 3 ? "fill-text" : "fill-accent"}>
              {us(r.v)}
            </Label>
            <rect x={X} y={252 + i * 32} width={W} height={6} rx={2} className="fill-border" />
            <rect x={X} y={252 + i * 32} width={Math.max(1.5, (r.v / tokenUs) * W)} height={6} rx={2} className={i === 3 ? "fill-muted/60" : "fill-accent"} />
          </g>
        ))}
      </motion.g>
      <Label x={X} y={384}>{`▁ marks a space; the last pieces are the template asking for the assistant's turn`}</Label>
    </Frame>
  );
}

export function LayerFigure() {
  const { on, grow } = useLoop(14);
  const px = W / layerUs;
  const starts = before(layerOps.map((o) => o.layer * px));
  return (
    <Frame label={`One layer of the decode step takes ${us(layerUs)}: ${layerOps.map((o) => `${o.label} ${us(o.layer)}`).join(", ")}. ${journey.layers} layers take ${us(layersUs)}.`}>
      <Label x={X} y={22} className="fill-text">{`one layer, to scale: ${us(layerUs)} of kernel time, ${journey.layers} times over`}</Label>
      <motion.g {...grow(0.05, 0.3)}>
        {layerOps.map((o, i) => (
          <rect key={i} x={X + starts[i]} y={34} width={Math.max(1, o.layer * px - 1)} height={26} rx={2} className={o.slug === "gemm" ? "fill-accent" : o.slug === "paged-attention" ? "fill-text/70" : "fill-warn"} />
        ))}
      </motion.g>
      {layerOps.map((o, i) => (
        <motion.g key={i} {...on(0.3 + i * 0.04)}>
          <rect x={X} y={82 + i * 27} width={8} height={8} rx={1} className={o.slug === "gemm" ? "fill-accent" : o.slug === "paged-attention" ? "fill-text/70" : "fill-warn"} />
          <Label x={X + 16} y={90 + i * 27} className="fill-text">
            {o.label}
          </Label>
          <Label x={X + 150} y={90 + i * 27}>{o.does}</Label>
          <Label x={X + W} y={90 + i * 27} anchor="end" className="fill-accent">
            {us(o.layer)}
          </Label>
        </motion.g>
      ))}
      <Label x={X} y={372}>{`blue: matrix products (cuBLAS), ${fmt((layerOps.filter((o) => o.slug === "gemm").reduce((a, o) => a + o.layer, 0) / layerUs) * 100)}% of the layer; times are the step's mean ÷ ${journey.layers} (measured)`}</Label>
    </Frame>
  );
}

export function HeadFigure() {
  const { on, grow } = useLoop(14);
  const max = journey.top[0].logit;
  const rows = [stage("final-norm"), stage("lm-head"), stage("sample")];
  return (
    <Frame label={`After "${journey.soFar}" the five largest logits are ${journey.top.map((t) => `${t.token} ${t.logit}`).join(", ")}; greedy sampling keeps ${journey.chosen}, ID ${journey.chosenId}. Final norm ${us(rows[0].us)}, LM head ${us(rows[1].us)}, sampler ${us(rows[2].us)}.`}>
      <Label x={X} y={22} className="fill-text">{`after "${journey.soFar}": the 5 largest of ${fmt(journey.vocab)} logits`}</Label>
      {journey.top.map((t, i) => (
        <g key={t.token}>
          <text x={X} y={52 + i * 30} className={`${i === 0 ? "fill-accent" : "fill-text"} font-mono text-[11px]`}>
            {t.token}
          </text>
          <rect x={X + 100} y={42 + i * 30} width={W - 150} height={12} rx={2} className="fill-border" />
          <motion.rect x={X + 100} y={42 + i * 30} width={(t.logit / max) * (W - 150)} height={12} rx={2} className={i === 0 ? "fill-accent" : "fill-muted/60"} {...grow(0.08 + i * 0.05, 0.3 + i * 0.05)} />
          <Label x={X + W} y={52 + i * 30} anchor="end" className={i === 0 ? "fill-accent" : "fill-muted"}>
            {t.logit.toFixed(2)}
          </Label>
        </g>
      ))}
      <motion.g {...on(0.45)}>
        <Label x={X} y={212} className="fill-accent">{`the largest wins by ${journey.lead.toFixed(2)}: ID ${journey.chosenId}, the text "${journey.chosenText}"`}</Label>
      </motion.g>
      <motion.g {...on(0.55)}>
        <Label x={X} y={246} className="fill-text">
          the three kernels at the end of the step
        </Label>
        {rows.map((r, i) => (
          <g key={r.id}>
            <Label x={X} y={272 + i * 30}>{`${r.label}: ${r.does}`}</Label>
            <Label x={X + W} y={272 + i * 30} anchor="end" className="fill-accent">
              {us(r.us)}
            </Label>
            <rect x={X} y={278 + i * 30} width={W} height={5} rx={2} className="fill-border" />
            <rect x={X} y={278 + i * 30} width={Math.max(1.5, (r.us / rows[1].us) * W)} height={5} rx={2} className="fill-accent" />
          </g>
        ))}
      </motion.g>
      <Label x={X} y={384}>logits from the Hugging Face reference in BF16; times from the engine&apos;s profile (measured)</Label>
    </Frame>
  );
}

export function TripFigure() {
  const { on, grow } = useLoop(14);
  const px = W / tokenUs;
  const starts = before(trip.map((t) => t.us * px));
  const cls = (id: string) => (id === "layers" ? "fill-accent" : id === "host" ? "fill-muted/60" : id === "gaps" ? "fill-warn/60" : "fill-text/70");
  return (
    <Frame label={`One batch-1 token, ${fmt(stepProfile.tpotMs, 2)} ms: ${trip.map((t) => `${t.label} ${us(t.us)}, ${shareText(t.us)}`).join("; ")}. Once per request: encode ${us(hostTimes.encode)}.`}>
      <Label x={X} y={22} className="fill-text">{`one token, to scale: ${fmt(stepProfile.tpotMs, 2)} ms`}</Label>
      <motion.g {...grow(0.05, 0.32)}>
        {trip.map((t, i) => (
          <rect key={t.id} x={X + starts[i]} y={34} width={Math.max(1, t.us * px - 1)} height={26} rx={2} className={cls(t.id)} />
        ))}
      </motion.g>
      {trip.map((t, i) => (
        <motion.g key={t.id} {...on(0.32 + i * 0.05)}>
          <rect x={X} y={84 + i * 34} width={8} height={8} rx={1} className={cls(t.id)} />
          <Label x={X + 16} y={92 + i * 34} className="fill-text">
            {t.label}
          </Label>
          <Label x={X + 16} y={106 + i * 34}>{t.does}</Label>
          <Label x={X + W} y={92 + i * 34} anchor="end" className="fill-accent">{`${us(t.us)} · ${shareText(t.us)}`}</Label>
        </motion.g>
      ))}
      <Label x={X} y={340} className="fill-text">{`once per request, before the first step: encode, ${us(hostTimes.encode)}`}</Label>
      <rect x={X} y={348} width={Math.max(1.5, hostTimes.encode * px)} height={8} rx={1} className="fill-warn" />
      <Label x={X} y={384}>{`batch 1, kernels launched one by one, profile of ${stepProfile.date} (measured)`}</Label>
    </Frame>
  );
}
