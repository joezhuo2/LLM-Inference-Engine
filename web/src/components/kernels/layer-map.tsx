"use client";

import NumberFlow from "@number-flow/react";
import clsx from "clsx";
import Link from "next/link";
import { useState } from "react";
import { kernelCards, layerOps } from "@/content/kernels";
import { cellNumber, decodeSteps } from "@/lib/data";

const choices = [
  { key: "b1", label: "batch 1" },
  { key: "b16", label: "batch 16" },
] as const;

export function LayerMap() {
  const [batch, setBatch] = useState<"b1" | "b16">("b1");
  const table = decodeSteps[batch];
  if (!table) return null;
  const col = (name: string) => table.header.indexOf(name);
  const time = (op: string) => cellNumber(table.rows.find((r) => r[0] === op)?.[col("GPU us/step")]) ?? 0;
  const share = (op: string) => cellNumber(table.rows.find((r) => r[0] === op)?.[col("% of kernel time")]) ?? 0;
  const max = Math.max(...layerOps.map((l) => time(l.op)));
  const ready = new Set(kernelCards.filter((k) => k.ready).map((k) => k.slug));
  return (
    <figure className="rounded border border-border bg-surface p-4 sm:p-5">
      <div className="flex flex-wrap items-baseline justify-between gap-3">
        <figcaption className="max-w-xl text-sm text-muted">
          One of TinyLlama&apos;s 22 layers, in the order its kernels run in a decode step. Bars are each kernel&apos;s GPU
          time per step over all layers.
        </figcaption>
        <div role="radiogroup" aria-label="batch size" className="flex overflow-hidden rounded border border-border font-mono text-xs">
          {choices.map((c) => (
            <button
              key={c.key}
              type="button"
              role="radio"
              aria-checked={batch === c.key}
              onClick={() => setBatch(c.key)}
              className={clsx("pressable px-3 py-1", batch === c.key ? "bg-accent-dim text-accent" : "text-muted")}
            >
              {c.label}
            </button>
          ))}
        </div>
      </div>
      <ol className="mt-5 space-y-1.5 font-mono text-xs">
        {layerOps.map((l, i) => {
          const linked = ready.has(l.slug);
          const label = (
            <span className={clsx("truncate", linked ? "text-accent" : "text-text")}>
              <span className="text-muted">{String(i + 1).padStart(2, "0")} </span>
              {l.label}
            </span>
          );
          return (
            <li key={l.op} className="grid grid-cols-[10.5rem_1fr_4.5rem] items-center gap-3 sm:grid-cols-[13rem_1fr_6rem]">
              {linked ? (
                <Link href={`/kernels/${l.slug}`} className="link-underline truncate">
                  {label}
                </Link>
              ) : (
                label
              )}
              <div className="h-2 min-w-0">
                <div
                  className={clsx("bar-ease h-full origin-left rounded-sm", linked ? "bg-accent" : "bg-baseline/60")}
                  style={{ transform: `scaleX(${Math.max(0.006, time(l.op) / max)})` }}
                />
              </div>
              <span className="text-right text-muted tabular-nums">
                <NumberFlow value={share(l.op)} suffix="%" format={{ minimumFractionDigits: 1, maximumFractionDigits: 1 }} />
              </span>
            </li>
          );
        })}
      </ol>
      <p className="mt-4 font-mono text-[11px] text-muted">
        from {table.file}, eager decode step, Nsight Systems; the LM head and the final norm run once after the last layer
      </p>
    </figure>
  );
}
