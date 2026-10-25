import { TokenRain } from "@/components/token-rain";
import { format, medianOf } from "@/lib/data";

export function Hero({ next }: { next?: { href: string; label: string } }) {
  const graphs1 = medianOf("engine_graphs", 1, "output_throughput");
  const graphs16 = medianOf("engine_graphs", 16, "output_throughput");
  const hf16 = medianOf("hf", 16, "output_throughput");
  const tpot1 = medianOf("engine_graphs", 1, "tpot_s_p50");
  const stats = [
    { label: "tokens/s at batch 16", value: format(graphs16, 0) },
    { label: "batch 16 over batch 1", value: graphs1 && graphs16 ? `${format(graphs16 / graphs1)}x` : "–" },
    { label: "over HuggingFace", value: graphs16 && hf16 ? `${format(graphs16 / hf16)}x` : "–" },
    { label: "ms per token at batch 1", value: format(tpot1 === undefined ? undefined : tpot1 * 1e3, 2) },
  ];
  return (
    <section className="relative h-[calc(100svh-3rem)] min-h-[520px] overflow-hidden">
      <TokenRain />
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-0"
        style={{
          background:
            "linear-gradient(to top, var(--bg) 0%, color-mix(in srgb, var(--bg) 82%, transparent) 40%, transparent 72%)",
        }}
      />
      <div className="absolute inset-x-0 bottom-0">
        <div className="mx-auto w-full max-w-5xl space-y-5 px-4 pb-8 sm:px-6 sm:pb-12">
          <h1 className="font-mono text-3xl sm:text-5xl">llm-inference-engine</h1>
          <p className="max-w-xl text-muted">
            TinyLlama 1.1B served from scratch in C++ and CUDA: a paged KV cache, continuous batching, hand-written
            kernels and CUDA Graphs.
          </p>
          <dl className="flex flex-wrap gap-x-8 gap-y-3 font-mono">
            {stats.map((s) => (
              <div key={s.label}>
                <dt className="text-xs text-muted">{s.label}</dt>
                <dd className="text-xl text-accent tabular-nums">{s.value}</dd>
              </div>
            ))}
          </dl>
          <div className="flex flex-wrap items-end justify-between gap-4 font-mono text-xs text-muted">
            <p className="max-w-md">
              each falling word is a token the engine generated in a recorded run; heavier rain means more tokens per
              second, amber marks a request&apos;s first token
            </p>
            {next && (
              <a href={next.href} className="pressable link-hover inline-flex items-center gap-2 text-text">
                {next.label}
                <span aria-hidden="true" className="text-accent">
                  ↓
                </span>
              </a>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
