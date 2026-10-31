"use client";

import clsx from "clsx";
import { motion } from "motion/react";
import Link from "next/link";
import { useId, useState } from "react";
import { repoUrl } from "@/content/build-log";
import { cellNumber, decodeSteps } from "@/lib/data";
import { Label } from "./frame";
import { useLoop } from "./loop";

const vocab = 32000;
const hidden = 2048;

function stepCell(batch: "b1" | "b16", op: string, column: string) {
  const table = decodeSteps[batch];
  if (!table) return undefined;
  return cellNumber(table.rows.find((r) => r[0] === op)?.[table.header.indexOf(column)]);
}

const ids = [
  { id: 1, row: 0 },
  { id: 15043, row: 4 },
  { id: 3186, row: 2 },
];

function GatherFigure() {
  const { on, move } = useLoop(6);
  const rows = 6;
  const rowH = 16;
  const tx = 150;
  const ox = 400;
  return (
    <svg viewBox="0 0 560 180" className="w-full font-mono [font-variant-ligatures:none]" role="img" aria-label="Each token id picks one row of the embedding table, copied into the hidden state">
      <Label x={20} y={18}>token ids</Label>
      {ids.map((t, i) => (
        <motion.g key={t.id} {...on(0.04 + i * 0.28, 0.94)}>
          <rect x={20} y={30 + i * 26} width={70} height={18} rx={3} className="fill-accent/15 stroke-accent" />
          <Label x={55} y={43 + i * 26} anchor="middle" className="fill-text">
            {t.id.toLocaleString("en-US")}
          </Label>
        </motion.g>
      ))}
      <Label x={tx} y={18}>{`table E: ${vocab.toLocaleString("en-US")} × ${hidden.toLocaleString("en-US")}`}</Label>
      {Array.from({ length: rows }, (_, r) => (
        <rect key={r} x={tx} y={30 + r * (rowH + 2)} width={200} height={rowH} rx={2} className="fill-baseline/40" />
      ))}
      <Label x={tx + 100} y={30 + rows * (rowH + 2) + 12} anchor="middle">
        ⋮
      </Label>
      <Label x={ox} y={18}>{`x: tokens × ${hidden.toLocaleString("en-US")}`}</Label>
      {ids.map((t, i) => {
        const y = 30 + t.row * (rowH + 2);
        const yo = 30 + i * 26;
        const t0 = 0.08 + i * 0.28;
        return (
          <g key={t.id}>
            <motion.rect x={tx} y={y} width={200} height={rowH} rx={2} className="fill-accent/50" {...on(t0, t0 + 0.2, false)} />
            <motion.rect x={tx} y={y} width={140} height={rowH} rx={2} className="fill-accent" {...move(t0 + 0.04, t0 + 0.2, ox - tx, yo - y + 1, true, 0.94)} />
          </g>
        );
      })}
      <Label x={20} y={172}>one block per token; 256 threads copy its 2,048 BF16 values</Label>
    </svg>
  );
}

function HeadFigure() {
  const { on, grow } = useLoop(6);
  const bars = [0.3, 0.55, 0.2, 0.9, 0.4, 0.25, 0.6, 0.35, 0.15, 0.5, 0.3, 0.45, 0.2, 0.38];
  return (
    <svg viewBox="0 0 560 180" className="w-full font-mono [font-variant-ligatures:none]" role="img" aria-label="The last token's hidden state is normalized, multiplied by the LM head into 32,000 logits, and handed to the sampler">
      <Label x={20} y={18}>last token</Label>
      <rect x={20} y={30} width={90} height={16} rx={2} className="fill-baseline/60" />
      <Label x={20} y={62}>{hidden.toLocaleString("en-US")} values</Label>
      <motion.g {...on(0.12)}>
        <Label x={140} y={18}>RMSNorm</Label>
        <rect x={140} y={30} width={90} height={16} rx={2} className="fill-accent/60" />
      </motion.g>
      <motion.g {...on(0.28)}>
        <Label x={260} y={18}>{`× Wᵀ (${hidden.toLocaleString("en-US")} × ${vocab.toLocaleString("en-US")})`}</Label>
        <Label x={260} y={62}>a GEMV that reads 131 MB of weights</Label>
      </motion.g>
      <Label x={20} y={96}>{`${vocab.toLocaleString("en-US")} logits`}</Label>
      {bars.map((b, i) => (
        <motion.rect key={i} x={20 + i * 24} y={160 - b * 50} width={18} height={b * 50} rx={1} className={i === 3 ? "fill-accent" : "fill-baseline"} {...grow(0.4 + i * 0.015, 0.5 + i * 0.015, "y")} />
      ))}
      <motion.g {...on(0.7)}>
        <Label x={380} y={120} className="fill-text">
          → the sampler picks one id
        </Label>
        <Label x={380} y={138}>greedy, temperature, top-k, top-p</Label>
      </motion.g>
    </svg>
  );
}

type End = "in" | "out";

const ends: Record<End, { title: string; note: string; ops: string[]; formula: string; sources: { path: string; label: string }[]; text: string; link?: { href: string; label: string } }> = {
  in: {
    title: "embedding lookup",
    note: "before the first layer",
    ops: ["embedding"],
    formula: "x[t] = E[id[t]]",
    sources: [{ path: "src/kernels/embedding.cu", label: "embedding.cu" }],
    text: "The smallest kernel in the engine and the only one that does no arithmetic: each token id selects one row of the embedding table, and one thread block copies that row into the hidden state. It is still the naive version, because at about 3 µs a step there is nothing to gain.",
  },
  out: {
    title: "final norm + LM head",
    note: "after the last layer",
    ops: ["final_norm", "lm_head"],
    formula: "logits = RMSNorm(x) · Wᵀ",
    sources: [
      { path: "src/kernels/rmsnorm.cu", label: "rmsnorm.cu" },
      { path: "src/kernels/gemm.cpp", label: "gemm.cpp" },
    ],
    text: "Only the last token of each sequence needs logits. It goes through one more RMSNorm, then a cuBLAS GEMV against the 32,000-row output matrix, the largest single weight matrix in the model, and the logits go to the sampler.",
    link: { href: "/kernels/gemm", label: "how the GEMV runs" },
  },
};

export function LayerEnd({ end, batch }: { end: End; batch: "b1" | "b16" }) {
  const [open, setOpen] = useState(false);
  const [seen, setSeen] = useState(false);
  const panel = useId();
  const e = ends[end];
  const us = e.ops.reduce((a, op) => a + (stepCell(batch, op, "GPU us/step") ?? 0), 0);
  const share = e.ops.reduce((a, op) => a + (stepCell(batch, op, "% of kernel time") ?? 0), 0);
  return (
    <div className={clsx("rounded border border-dashed border-border font-mono text-xs", open && "border-solid")}>
      <button type="button" aria-expanded={open} aria-controls={panel} onClick={() => {
          setSeen(true);
          setOpen((o) => !o);
        }} className="pressable flex w-full items-center justify-between gap-3 px-3 py-2 text-left">
        <span className="min-w-0 truncate">
          <span className="text-muted">{end === "in" ? "in " : "out "}</span>
          <span className="text-text">{e.title}</span>
          <span className="text-muted">{` · ${e.note}`}</span>
        </span>
        <span className="shrink-0 tabular-nums text-muted">
          {`${us.toFixed(1)} µs · ${share.toFixed(1)}%`}
          <span aria-hidden className={clsx("ml-2 inline-block transition-transform duration-700 ease-in-out motion-reduce:transition-none", open && "rotate-90")}>
            ›
          </span>
        </span>
      </button>
      <div
        id={panel}
        className={clsx(
          "grid transition-[grid-template-rows,opacity] duration-700 ease-in-out motion-reduce:transition-none",
          open ? "grid-rows-[1fr] opacity-100" : "grid-rows-[0fr] opacity-0",
        )}
      >
        <div className="overflow-hidden" inert={!open}>
          <div className="space-y-3 border-t border-border px-3 py-3">
            <div className="text-text">{e.formula}</div>
            {seen && (end === "in" ? <GatherFigure /> : <HeadFigure />)}
            <p className="font-sans text-sm leading-relaxed text-muted">{e.text}</p>
            <div className="flex flex-wrap gap-x-4 gap-y-1">
              {e.sources.map((s) => (
                <a key={s.path} href={`${repoUrl}/blob/main/${s.path}`} className="link-underline text-accent" target="_blank" rel="noreferrer">
                  {s.label}
                </a>
              ))}
              {e.link && (
                <Link href={e.link.href} transitionTypes={["nav-forward"]} className="link-underline text-accent">
                  {e.link.label}
                </Link>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
