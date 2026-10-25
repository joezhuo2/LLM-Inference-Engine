"use client";

import type { ReactNode } from "react";
import { format, medianOf } from "@/lib/data";

export function HeroFrame({ children, caption }: { children: ReactNode; caption: string }) {
  const graphs1 = medianOf("engine_graphs", 1, "output_throughput");
  const graphs16 = medianOf("engine_graphs", 16, "output_throughput");
  const hf16 = medianOf("hf", 16, "output_throughput");
  const stats = [
    { label: "tokens/s at batch 16", value: format(graphs16, 0) },
    { label: "batch 16 over batch 1", value: graphs1 && graphs16 ? `${format(graphs16 / graphs1)}x` : "–" },
    { label: "over HuggingFace", value: graphs16 && hf16 ? `${format(graphs16 / hf16)}x` : "–" },
  ];
  return (
    <section className="relative -mx-4 overflow-hidden border-y border-border sm:mx-0 sm:rounded sm:border">
      <div className="relative h-[72vh] min-h-[480px]">
        {children}
        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-0"
          style={{
            background:
              "linear-gradient(to top, var(--bg) 0%, color-mix(in srgb, var(--bg) 80%, transparent) 38%, transparent 70%)",
          }}
        />
        <div className="absolute inset-x-0 bottom-0 space-y-5 p-5 sm:p-8">
          <h1 className="font-mono text-2xl sm:text-4xl">llm-inference-engine</h1>
          <p className="max-w-xl text-muted">
            TinyLlama served from scratch in C++ and CUDA: paged KV cache, continuous batching, CUDA Graphs.
          </p>
          <dl className="flex flex-wrap gap-x-8 gap-y-3 font-mono">
            {stats.map((s) => (
              <div key={s.label}>
                <dt className="text-xs text-muted">{s.label}</dt>
                <dd className="text-xl text-accent tabular-nums">{s.value}</dd>
              </div>
            ))}
          </dl>
          <p className="font-mono text-xs text-muted">{caption}</p>
        </div>
      </div>
    </section>
  );
}

export function PlayButton({ onPlay }: { onPlay: () => void }) {
  return (
    <button
      type="button"
      onClick={onPlay}
      className="pressable absolute right-4 top-4 rounded border border-border bg-surface px-3 py-1 font-mono text-xs text-muted"
    >
      play the recorded run
    </button>
  );
}
