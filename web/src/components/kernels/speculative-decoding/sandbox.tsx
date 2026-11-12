"use client";

import clsx from "clsx";
import { useState } from "react";
import NumberFlow from "@/components/number-flow";
import { piece, spec } from "@/content/kernels/speculative-decoding";
import { Toggle } from "../toggle";
import { replay, stepMs, totals } from "./lookup";

const drafts = ["1", "2", "4", "8", "15"];
const ngrams = ["1", "2", "3", "4"];
const limits = ["fixed", "adaptive"];
const plainMs = stepMs(1);
const all = [...spec.example.prompt_tokens, ...spec.example.output_tokens];
const text = (tokens: string[]) => tokens.map(piece).join("").trim();

export function SpeculativeSandbox() {
  const [maxDraft, setMaxDraft] = useState(String(spec.maxDraft));
  const [maxNgram, setMaxNgram] = useState(String(spec.maxNgram));
  const [limit, setLimit] = useState("fixed");
  const [picked, setPicked] = useState(0);
  const steps = replay({ maxDraft: Number(maxDraft), maxNgram: Number(maxNgram), adaptive: limit === "adaptive" });
  const t = totals(steps);
  const tallest = spec.maxBatch;
  const at = Math.min(picked, steps.length - 1);
  const step = steps[at];
  const tiles = [
    { label: "decode steps", value: <NumberFlow value={t.steps} /> },
    { label: "guessed rows", value: <NumberFlow value={t.drafted} /> },
    { label: "kept", value: <NumberFlow value={t.kept} /> },
    { label: "dropped", value: <NumberFlow value={t.wasted} /> },
    { label: "tokens per step", value: <NumberFlow value={t.tokensPerStep} format={{ minimumFractionDigits: 2, maximumFractionDigits: 2 }} /> },
    { label: "estimated ms per token", value: <NumberFlow value={t.msPerToken} format={{ minimumFractionDigits: 2, maximumFractionDigits: 2 }} /> },
  ];
  return (
    <div className="space-y-4 text-xs tabular-nums">
      <div className="flex flex-wrap gap-x-6 gap-y-2">
        <Toggle label="guesses" choices={drafts} value={maxDraft} onChange={setMaxDraft} />
        <Toggle label="n-gram" choices={ngrams} value={maxNgram} onChange={setMaxNgram} />
        <Toggle label="limit" choices={limits} value={limit} onChange={setLimit} />
      </div>
      <dl className="grid grid-cols-2 gap-x-6 gap-y-2 sm:grid-cols-3">
        {tiles.map((tile) => (
          <div key={tile.label}>
            <dt className="text-muted">{tile.label}</dt>
            <dd className="font-serif text-xl text-text">{tile.value}</dd>
          </div>
        ))}
      </dl>
      <div role="group" aria-label={`${t.steps} decode steps; each column is one step and each cell one row of the ${spec.maxBatch} a step can carry`} className="flex h-40 items-end gap-px">
        {steps.map((s, i) => (
          <button
            key={i}
            type="button"
            aria-label={`step ${i + 1}: ${s.draft.length} guesses, ${s.kept} kept`}
            aria-pressed={i === at}
            onClick={() => setPicked(i)}
            onPointerEnter={(e) => e.pointerType === "mouse" && setPicked(i)}
            className={clsx("flex min-w-0 flex-1 flex-col-reverse gap-px transition-[height,opacity] duration-700 ease-in-out", i === at ? "opacity-100" : "opacity-60")}
            style={{ height: `${((1 + s.draft.length) / tallest) * 100}%` }}
          >
            {Array.from({ length: 1 + s.draft.length }, (_, row) => (
              <span key={row} className={clsx("block w-full flex-1 rounded-[1px]", row === 0 ? "bg-text" : row <= s.kept ? "bg-accent" : "border border-warn")} />
            ))}
          </button>
        ))}
      </div>
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-muted">
        <span><span className="mr-1 inline-block size-2 bg-text" />the row every step has</span>
        <span><span className="mr-1 inline-block size-2 bg-accent" />guess kept</span>
        <span><span className="mr-1 inline-block size-2 border border-warn" />guess dropped</span>
      </div>
      <p role="status" aria-live="polite" className="min-h-16 leading-relaxed text-muted">
        <span className="text-text">{`Step ${at + 1} of ${t.steps}. `}</span>
        {step.draft.length === 0 ? `The last tokens, “${text(all.slice(Math.max(0, step.at - Number(maxNgram)), step.at))}”, appear nowhere earlier, so the step ran one row and wrote one token.` : `Guessed “${text(step.draft)}”. The model agreed with ${step.kept} of ${step.draft.length}, so the step wrote ${step.kept + 1} ${step.kept === 0 ? "token" : "tokens"} in about ${stepMs(1 + step.draft.length).toFixed(2)} ms.`}
      </p>
      <p className="leading-relaxed text-muted">{`A plain decode step of one row takes ${plainMs.toFixed(2)} ms (measured), so without guesses this answer costs ${plainMs.toFixed(2)} ms per token. The estimate adds up the measured median time of a step by its row count; the steps and the kept guesses are computed, not measured.`}</p>
    </div>
  );
}
