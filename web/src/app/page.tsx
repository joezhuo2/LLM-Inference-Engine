import Link from "next/link";
import { format, medianOf } from "@/lib/data";

export default function Overview() {
  const graphs1 = medianOf("engine_graphs", 1, "output_throughput");
  const graphs16 = medianOf("engine_graphs", 16, "output_throughput");
  const hf16 = medianOf("hf", 16, "output_throughput");
  const tpot1 = medianOf("engine_graphs", 1, "tpot_s_p50");
  const stats = [
    { label: "tokens/s at batch 16", value: format(graphs16, 0) },
    { label: "batch 16 over batch 1", value: graphs1 && graphs16 ? `${format(graphs16 / graphs1)}x` : "–" },
    { label: "over HuggingFace at batch 16", value: graphs16 && hf16 ? `${format(graphs16 / hf16)}x` : "–" },
    { label: "ms per token at batch 1", value: format(tpot1 === undefined ? undefined : tpot1 * 1e3, 2) },
  ];
  return (
    <div className="space-y-12">
      <section className="space-y-4">
        <h1 className="font-mono text-2xl text-text sm:text-3xl">llm-inference-engine</h1>
        <p className="max-w-2xl text-muted">
          A TinyLlama 1.1B inference engine written from scratch in C++ and CUDA: a paged KV cache, continuous
          batching with preemption, hand-written attention and normalization kernels, and CUDA Graphs for the
          decode step.
        </p>
      </section>
      <section aria-label="Headline numbers" className="grid grid-cols-2 gap-px overflow-hidden rounded border border-border bg-border sm:grid-cols-4">
        {stats.map((s) => (
          <div key={s.label} className="bg-surface p-4">
            <div className="font-mono text-2xl text-accent tabular-nums">{s.value}</div>
            <div className="mt-1 font-mono text-xs text-muted">{s.label}</div>
          </div>
        ))}
      </section>
      <section className="grid gap-4 sm:grid-cols-3">
        {[
          { href: "/results", title: "results", text: "Throughput and latency against HuggingFace and llama.cpp." },
          { href: "/kernels", title: "kernels", text: "How each kernel got faster, with Nsight numbers." },
          { href: "/dashboard", title: "dashboard", text: "A recorded serving run, step by step." },
        ].map((card) => (
          <Link key={card.href} href={card.href} className="pressable block rounded border border-border bg-surface p-4">
            <div className="font-mono text-sm text-accent">{card.title}</div>
            <p className="mt-2 text-sm text-muted">{card.text}</p>
          </Link>
        ))}
      </section>
    </div>
  );
}
