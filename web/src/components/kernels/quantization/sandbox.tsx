"use client";

import clsx from "clsx";
import { useState } from "react";
import { qFit, qFmt, qGb, qGpus, qModels, sandboxBits } from "@/content/kernels/quantization";
import { expectAnswer, type Prediction, Predict } from "../predict";
import { Toggle } from "../toggle";

const fmt = qFmt;
const ease = "transition-all duration-700 ease-in-out motion-reduce:transition-none";
const kvChoices = ["BF16", "FP8"];

const big = qModels[1];
const here = qGpus[0];
const firstFit = sandboxBits.find((b) => qFit(big, here, b, 2).fits);
const at8 = qFit(big, here, 8, 2);
const at4 = qFit(big, here, 4, 2);
const prediction: Prediction = expectAnswer(
  {
    question: `${big.name} on this GPU: at how many bits per weight do its weights first fit?`,
    choices: [...sandboxBits.map((b) => `${b} bits`), "never"],
    answer: firstFit ? `${firstFit} bits` : "never",
    why: `At 8 bits the weights are ${qGb(at8.weights)} and this GPU has ${qGb(at8.room)} once everything else is counted. At 4 bits they take ${qGb(at4.weights)}, and what is left becomes a pool of ${fmt(at4.tokens)} tokens.`,
  },
  "4 bits",
);

export function QuantSandbox() {
  const [bits, setBits] = useState("8");
  const [kv, setKv] = useState("BF16");
  const [model, setModel] = useState(qModels[0].name);
  const [gpu, setGpu] = useState(qGpus[0].name);
  const m = qModels.find((x) => x.name === model) ?? qModels[0];
  const g = qGpus.find((x) => x.name === gpu) ?? qGpus[0];
  const f = qFit(m, g, Number(bits), kv === "FP8" ? 1 : 2);
  const full = Math.max(g.memory, f.weights + g.rest);
  const bar = [
    { label: "everything else", bytes: g.rest, cls: "bg-muted/50" },
    { label: "weights", bytes: Math.min(f.weights, f.room), cls: "bg-text/70" },
    { label: "KV pool", bytes: f.pool, cls: "bg-accent" },
  ];
  const tiles = [
    { label: "weights", value: qGb(f.weights) },
    { label: "room for them", value: qGb(f.room) },
    { label: "floor per token, batch 1", value: `${fmt(f.floorMs, 2)} ms` },
    { label: "KV bytes per token", value: fmt(f.perToken) },
    { label: "tokens the pool holds", value: f.fits ? fmt(f.tokens) : "none" },
    { label: "full 2,048-token contexts", value: f.fits ? fmt(f.contexts) : "none" },
  ];
  return (
    <div className="space-y-4 text-xs tabular-nums">
      <Predict {...prediction} />
      <div className="flex flex-wrap gap-x-6 gap-y-2">
        <Toggle label="bits per weight" choices={sandboxBits.map(String)} value={bits} onChange={setBits} />
        <Toggle label="KV cache" choices={kvChoices} value={kv} onChange={setKv} />
        <Toggle label="model" choices={qModels.map((x) => x.name)} value={model} onChange={setModel} />
        <Toggle label="GPU" choices={qGpus.map((x) => x.name)} value={gpu} onChange={setGpu} />
      </div>
      <figure className="space-y-1">
        <figcaption className="flex justify-between gap-2 text-muted">
          <span className="text-text">{`${g.name}: ${qGb(g.memory)}`}</span>
          <span className={f.fits ? "text-accent" : "text-warn"}>{f.fits ? "the weights fit" : `${qGb(f.over)} too large: no pool, no request`}</span>
        </figcaption>
        <div role="img" aria-label={`${g.name}, ${qGb(g.memory)}: everything else ${qGb(g.rest)}, weights ${qGb(f.weights)}, KV pool ${qGb(f.pool)}`} className="relative flex h-8 overflow-hidden rounded-sm border border-border">
          {bar.map((s) => (
            <span key={s.label} className={clsx("block h-full border-r border-bg", ease, s.cls)} style={{ width: `${(s.bytes / full) * 100}%` }} />
          ))}
          {!f.fits && <span className={clsx("block h-full bg-warn", ease)} style={{ width: `${(f.over / full) * 100}%` }} />}
        </div>
        <p className="text-muted">grey: everything else; dark: weights; blue: the KV pool, 90% of what is left</p>
      </figure>
      <dl className="grid grid-cols-2 gap-x-6 gap-y-2 sm:grid-cols-3">
        {tiles.map((t) => (
          <div key={t.label}>
            <dt className="text-muted">{t.label}</dt>
            <dd className="text-sm text-text">{t.value}</dd>
          </div>
        ))}
      </dl>
      <p className="text-muted">{`calculated, nothing measured: ${m.source}; ${g.source}. Weights at exactly ${bits} bits, scales not counted.${m.id === "tiny" ? "" : " The floor counts every parameter as read, so it overstates it by the embedding table."}${g.id === "here" ? "" : " Nothing else is counted on this GPU's memory: its CUDA context and buffers are not known here."}`}</p>
    </div>
  );
}
