"use client";

import { motion } from "motion/react";
import { bigSide, engineSide, type ScaleId, scaleQuantities } from "@/content/scale";
import { Frame, Label } from "./frame";
import { useLoop } from "./loop";

const X = 20;
const BAR = 196;
const SPAN = 252;
const sides = [
  { side: engineSide, label: `${engineSide.model}, ${engineSide.gpu.replace(" GPU", "")}`, fill: "fill-accent", text: "fill-accent" },
  { side: bigSide, label: `${bigSide.model}, ${bigSide.gpu}`, fill: "fill-warn", text: "fill-warn" },
];

export type ScaleMark = { value: number; label: string };

export function ScaleFigure({ quantities, marks = {}, title }: { quantities: ScaleId[]; marks?: Partial<Record<ScaleId, ScaleMark>>; title: string }) {
  const { on, grow } = useLoop(14);
  const pitch = Math.min(112, 316 / quantities.length);
  const said = quantities.map((id) => {
    const q = scaleQuantities[id];
    return `${q.label}: ${sides.map((s) => `${s.label} ${q.text(q.value(s.side))}`).join(", ")}`;
  });
  return (
    <Frame label={`${title}. ${said.join("; ")}`}>
      <Label x={X} y={22} className="fill-text">
        {title}
      </Label>
      {quantities.map((id, k) => {
        const q = scaleQuantities[id];
        const values = sides.map((s) => q.value(s.side));
        const mark = marks[id];
        const max = Math.max(...values, mark?.value ?? 0);
        const px = SPAN / max;
        const y = 52 + k * pitch;
        const ratio = values[1] / values[0];
        return (
          <motion.g key={id} {...on(0.04 + k * 0.12)}>
            <Label x={X} y={y} className="fill-text">
              {q.label}
            </Label>
            <Label x={X + 520} y={y} anchor="end">
              {`${ratio.toFixed(ratio < 1 ? 2 : 1)}x`}
            </Label>
            {sides.map((s, i) => {
              const yy = y + 10 + i * 24;
              return (
                <g key={s.label}>
                  <Label x={X} y={yy + 13}>
                    {s.label}
                  </Label>
                  <rect x={BAR} y={yy} width={SPAN} height={18} rx={2} className="fill-border" />
                  <motion.rect x={BAR} y={yy} width={Math.max(2, values[i] * px)} height={18} rx={2} className={s.fill} {...grow(0.06 + k * 0.12 + i * 0.03, 0.16 + k * 0.12 + i * 0.03)} />
                  <Label x={X + 520} y={yy + 13} anchor="end" className={s.text}>
                    {q.text(values[i])}
                  </Label>
                </g>
              );
            })}
            {mark && (
              <g>
                <line x1={BAR + mark.value * px} x2={BAR + mark.value * px} y1={y + 6} y2={y + 62} className="stroke-text" strokeDasharray="3 2" strokeWidth={1} />
                <Label x={BAR + mark.value * px + 4} y={y + 74} className="fill-text">
                  {mark.label}
                </Label>
              </g>
            )}
          </motion.g>
        );
      })}
      <Label x={X} y={372}>{`${engineSide.model}: pool and ceilings measured on this GPU, the rest calculated`}</Label>
      <Label x={X} y={388}>{`${bigSide.model} on an ${bigSide.gpu}: calculated from published specs, not measured`}</Label>
    </Frame>
  );
}
