"use client";

import { motion, useReducedMotion } from "motion/react";

export type Step = { id: string; title: string; text: string; stat: string };

export const steps: Step[] = [
  {
    id: "v1",
    title: "v1: scores in shared memory",
    text: "One thread block per row and query head, 32 blocks for one sequence. Each block writes a score for every token of the context into shared memory, takes the softmax, then sums the values. The score array grows with the context, so v1 stops at 2,048 tokens.",
    stat: "600 us for one row at 2,048 tokens",
  },
  {
    id: "v2",
    title: "v2: online softmax",
    text: "v2 streams through the KV blocks keeping a running maximum and sum and rescaling what it has so far, so nothing grows with the context. Eight lanes share each key, so the loads coalesce, and every thread accumulates values.",
    stat: "600 to 165 us (3.6x), and 8,192-token contexts run",
  },
  {
    id: "split",
    title: "split: one context, many blocks",
    text: "With one row, v2 still launches only 32 blocks, far fewer than the GPU can run at once. The split kernel cuts the context into up to 8 ranges, each its own block writing a partial maximum, sum and weighted values; a merge kernel rescales each partial by exp(m - M) and adds them in order.",
    stat: "167.6 to 60.7 us at 2,048 tokens (2.8x); 14.0 to 9.1 ms per token at a 1,984-token prompt with CUDA Graphs",
  },
];

const cells = 16;
const x0 = 40;
const cellW = 35;
const width = cells * cellW;
const ease = [0.77, 0, 0.175, 1] as const;

export function AttentionFigure({ step }: { step: number }) {
  const reduced = useReducedMotion();
  const split = step === 2;
  const transition = { duration: reduced ? 0 : 0.45, ease };
  return (
    <svg viewBox="0 0 640 310" className="h-auto w-full font-mono" role="img" aria-label={steps[step].title}>
      <text x={x0} y={24} className="fill-muted text-[11px]">
        context: 2,048 tokens in KV blocks
      </text>
      {Array.from({ length: cells }, (_, i) => (
        <rect
          key={i}
          x={x0 + i * cellW + 1}
          y={34}
          width={cellW - 2}
          height={26}
          rx={2}
          className="fill-surface stroke-border"
        />
      ))}
      <motion.rect
        x={x0}
        y={78}
        width={width}
        height={14}
        rx={3}
        className="fill-accent"
        initial={false}
        animate={{ opacity: split ? 0 : 1 }}
        transition={transition}
      />
      {[0, 1, 2, 3].map((i) => (
        <motion.rect
          key={`bar${i}`}
          x={x0 + (i * width) / 4 + 2}
          y={78}
          width={width / 4 - 4}
          height={14}
          rx={3}
          className="fill-accent"
          initial={false}
          animate={{ opacity: split ? 1 : 0 }}
          transition={transition}
        />
      ))}
      {step > 0 &&
        !reduced &&
        (split ? [0, 1, 2, 3] : [0]).map((i) => {
          const range = split ? width / 4 : width;
          return (
            <rect
              key={`scan${step}${i}`}
              x={x0 + (split ? (i * width) / 4 : 0)}
              y={32}
              width={cellW - 2}
              height={30}
              rx={2}
              className="fill-none stroke-accent"
            >
              <animateTransform
                attributeName="transform"
                type="translate"
                from="0 0"
                to={`${range - cellW} 0`}
                dur={split ? "0.8s" : "1.6s"}
                repeatCount="indefinite"
              />
            </rect>
          );
        })}
      <text x={x0} y={112} className="fill-muted text-[11px]">
        {split ? "4 of up to 8 blocks per head, each over its own range" : "1 block per head, over the whole context"}
      </text>
      <motion.g initial={false} animate={{ opacity: step === 0 ? 1 : 0 }} transition={transition}>
        <rect x={x0} y={140} width={width} height={34} rx={3} className="fill-none stroke-warn" />
        <rect x={x0 + 4} y={146} width={width - 8} height={22} rx={2} className="fill-warn/30" />
        <text x={x0 + 10} y={161} className="fill-text text-[11px]">
          shared memory: 2,048 scores, full at the cap
        </text>
      </motion.g>
      <motion.g initial={false} animate={{ opacity: step === 1 ? 1 : 0 }} transition={transition}>
        <rect x={x0} y={140} width={150} height={34} rx={3} className="fill-none stroke-accent" />
        <text x={x0 + 10} y={161} className="fill-text text-[11px]">
          registers: m, l, acc
        </text>
      </motion.g>
      <motion.g initial={false} animate={{ opacity: split ? 1 : 0 }} transition={transition}>
        {[0, 1, 2, 3].map((i) => (
          <g key={`partial${i}`}>
            <line
              x1={x0 + (i + 0.5) * (width / 4)}
              y1={92}
              x2={x0 + width / 2}
              y2={230}
              className="stroke-border"
            />
            <rect x={x0 + i * (width / 4) + 20} y={140} width={width / 4 - 40} height={24} rx={3} className="fill-bg stroke-accent" />
            <text x={x0 + i * (width / 4) + 30} y={156} className="fill-text text-[10px]">
              m, l, acc
            </text>
          </g>
        ))}
        <circle cx={x0 + width / 2} cy={240} r={12} className="fill-accent" />
        <text x={x0 + width / 2 + 22} y={244} className="fill-text text-[11px]">
          merge: scale by exp(m - M), add in order
        </text>
      </motion.g>
      {steps[step].stat.split("; ").map((line, i) => (
        <text key={line} x={x0} y={276 + i * 18} className="fill-accent text-[12px]">
          {line}
        </text>
      ))}
    </svg>
  );
}
