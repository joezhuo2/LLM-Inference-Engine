"use client";

import clsx from "clsx";
import { useState } from "react";
import NumberFlow from "@/components/number-flow";
import { layerMatrices, rankChoices, rankShare, type TpMatrix } from "@/content/kernels/tensor-parallelism";
import { Toggle } from "../toggle";

const full = rankShare(1).total;
const ease = "transition-all duration-700 ease-in-out motion-reduce:transition-none";

function pieces(m: TpMatrix, n: number) {
  if (m.cut === "cols") return Array.from({ length: n }, (_, r) => ({ r, left: (r / n) * 100, top: 0, width: 100 / n, height: 100 }));
  const out: { r: number; left: number; top: number; width: number; height: number }[] = [];
  let top = 0;
  for (const b of m.bands) {
    const h = (b.rows / m.rows) * 100;
    for (let r = 0; r < n; r++) out.push({ r, left: 0, top: top + (r * h) / n, width: 100, height: h / n });
    top += h;
  }
  return out;
}

const num = (v: number) => v.toLocaleString("en-US");
const range = (count: number, n: number, rank: number) => (count / n === 1 ? num(rank) : `${num((rank * count) / n)} to ${num(((rank + 1) * count) / n - 1)}`);

function span(m: TpMatrix, n: number, rank: number) {
  if (n === 1) return "the whole matrix";
  const s = rankShare(1);
  const heads = `query heads ${range(s.qHeads, n, rank)}`;
  const channels = `channels ${range(s.channels, n, rank)}`;
  if (m.id === "wqkv") return `rows of ${heads}, K/V ${s.kvHeads / n === 1 ? "head" : "heads"} ${range(s.kvHeads, n, rank)}`;
  if (m.id === "wo") return `columns of ${heads}`;
  if (m.id === "wgu") return `rows of ${channels}, in gate and in up`;
  return `columns of ${channels}`;
}

export function TensorSandbox() {
  const [ranks, setRanks] = useState(String(rankChoices[1] ?? rankChoices[0]));
  const [picked, setPicked] = useState("0");
  const n = Number(ranks);
  const rank = Math.min(Number(picked), n - 1);
  const s = rankShare(n);
  const segments = [
    { label: "embedding", bytes: s.embed, cls: "bg-text/60" },
    { label: `layers ÷ ${n}`, bytes: s.layers, cls: "bg-accent" },
    { label: "LM head", bytes: s.head, cls: "bg-text/60" },
  ];
  const tiles = [
    { label: "query heads", value: s.qHeads },
    { label: "K/V heads", value: s.kvHeads },
    { label: "MLP channels", value: s.channels },
    { label: "MiB in one rank", value: s.total / 1048576, digits: 1 },
    { label: "% of the model", value: s.pct },
    { label: "all ranks, × the model", value: s.together, digits: 2 },
  ];
  return (
    <div className="space-y-4 text-xs tabular-nums">
      <div className="flex flex-wrap gap-x-6 gap-y-2">
        <Toggle label="ranks" choices={rankChoices.map(String)} value={ranks} onChange={setRanks} />
        <Toggle label="rank" choices={Array.from({ length: n }, (_, r) => String(r))} value={String(rank)} onChange={setPicked} />
      </div>
      <div className="grid grid-cols-2 gap-x-6 gap-y-3">
        {layerMatrices.map((m) => (
          <figure key={m.id} className="space-y-1">
            <figcaption className="flex justify-between gap-2 text-muted">
              <span className="text-text">{m.label}</span>
              <span>{`cut by ${m.cut === "rows" ? "rows" : "columns"}`}</span>
            </figcaption>
            <div role="img" aria-label={`${m.label}, ${m.rows.toLocaleString("en-US")} by ${m.cols.toLocaleString("en-US")}: rank ${rank} of ${n} holds ${m.cut === "rows" ? `1/${n} of its rows` : `1/${n} of its columns`}`} className="relative h-20 overflow-hidden rounded-sm border border-border">
              {pieces(m, n).map((p, i) => (
                <span key={`${n}-${i}`} className={clsx("absolute block border border-bg", ease, p.r === rank ? "bg-accent" : "bg-text/15")} style={{ left: `${p.left}%`, top: `${p.top}%`, width: `${p.width}%`, height: `${p.height}%` }} />
              ))}
            </div>
            <p className="min-h-4 text-muted">{span(m, n, rank)}</p>
          </figure>
        ))}
      </div>
      <dl className="grid grid-cols-3 gap-x-6 gap-y-2">
        {tiles.map((t) => (
          <div key={t.label}>
            <dt className="text-muted">{t.label}</dt>
            <dd className="font-serif text-xl text-text">
              <NumberFlow value={t.value} format={{ minimumFractionDigits: t.digits ?? 0, maximumFractionDigits: t.digits ?? 0 }} />
            </dd>
          </div>
        ))}
      </dl>
      <div className="space-y-1">
        <div className="flex justify-between text-muted">
          <span>one rank&apos;s bytes, against the whole model</span>
          <span>{`dashed: 1/${n} of the model`}</span>
        </div>
        <div role="img" aria-label={`one rank of ${n} holds ${s.pct.toFixed(0)}% of the model`} className="relative flex h-5 w-full rounded-sm border border-border">
          {segments.map((g) => (
            <span key={g.label} className={clsx("block h-full", ease, g.cls)} style={{ width: `${(g.bytes / full) * 100}%` }} />
          ))}
          {n > 1 && <span className={clsx("absolute -top-1 -bottom-1 border-l border-dashed border-text", ease)} style={{ left: `${100 / n}%` }} />}
        </div>
        <div className="flex flex-wrap gap-x-4 gap-y-1 text-muted">
          <span><span className="mr-1 inline-block size-2 bg-accent" />the layers, cut</span>
          <span><span className="mr-1 inline-block size-2 bg-text/60" />embedding and LM head, whole</span>
        </div>
      </div>
      <p className="leading-relaxed text-muted">Calculated from TinyLlama&apos;s shapes at 2 bytes per weight; the norm weights, a fraction of a MiB, are counted in the total but too thin to draw.</p>
    </div>
  );
}
