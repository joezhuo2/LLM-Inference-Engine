import { GlitchExit } from "@/components/glitch-exit";
import { Scramble } from "@/components/scramble";
import { TokenRain } from "@/components/token-rain";
import { headlineStats } from "@/lib/data";

export function Hero({ next }: { next?: { href: string; label: string } }) {
  const stats = headlineStats();
  return (
    <GlitchExit to={next?.href}>
      <section className="relative h-[calc(100svh-3rem)] min-h-[520px] overflow-hidden bg-bg">
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
            <h1 className="font-mono text-3xl sm:text-5xl">
              <Scramble text="llm-inference-engine" durationMs={1600} />
            </h1>
            <p className="max-w-xl text-muted">
              TinyLlama 1.1B served from scratch in C++ and CUDA: a paged KV cache, continuous batching, hand-written
              kernels and CUDA Graphs.
            </p>
            <dl className="flex flex-wrap gap-x-8 gap-y-3 font-mono">
              {stats.map((s, i) => (
                <div key={s.label}>
                  <dt className="text-xs text-muted">{s.label}</dt>
                  <dd className="text-xl text-accent tabular-nums">
                    <Scramble text={s.value} numeric delayMs={300 + i * 120} durationMs={1400} />
                  </dd>
                </div>
              ))}
            </dl>
            <div className="flex flex-wrap items-end justify-between gap-4 font-mono text-xs text-muted">
              <p className="max-w-md">
                each falling word is a token the engine generated in a recorded run; heavier rain means more tokens per
                second, amber marks a request&apos;s first token
              </p>
              {next && (
                <a href={next.href} className="pressable hover-grow inline-flex origin-right items-center gap-2 text-text">
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
    </GlitchExit>
  );
}
