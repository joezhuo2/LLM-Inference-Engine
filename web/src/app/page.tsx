import Link from "next/link";
import { BuildTimeline } from "@/components/build-timeline";
import { Hero } from "@/components/hero";
import { Scramble } from "@/components/scramble";

const cards = [
  { href: "/results", title: "results", text: "Throughput and latency against HuggingFace and llama.cpp." },
  { href: "/kernels", title: "kernels", text: "How each kernel got faster, with Nsight numbers." },
  { href: "/dashboard", title: "dashboard", text: "A recorded serving run, step by step, or your own engine live." },
];

export default function Overview() {
  return (
    <>
      <Hero next={{ href: "#built", label: "how it was built" }} />
      <div className="mx-auto w-full max-w-5xl space-y-24 px-4 py-20 sm:px-6">
        <section id="built" aria-labelledby="built-heading" className="scroll-mt-16 space-y-16">
          <div className="space-y-3">
            <h2 id="built-heading" className="font-mono text-2xl sm:text-3xl">
              <Scramble text="built from scratch" trigger="view" durationMs={1000} />
            </h2>
            <p className="max-w-2xl text-muted">
              Five weeks, one pull request at a time: from summing a vector on the GPU to beating llama.cpp. Every
              step was checked against a slower version that was already known to be right.
            </p>
          </div>
          <BuildTimeline />
        </section>
        <section aria-label="More" className="grid gap-4 sm:grid-cols-3">
          {cards.map((card) =>
            card.href ? (
              <Link key={card.title} href={card.href} className="pressable hover-grow block rounded border border-border bg-surface p-4">
                <div className="font-mono text-sm text-accent">{card.title}</div>
                <p className="mt-2 text-sm text-muted">{card.text}</p>
              </Link>
            ) : (
              <div key={card.title} className="block rounded border border-border bg-surface p-4">
                <div className="font-mono text-sm text-muted">{card.title}</div>
                <p className="mt-2 text-sm text-muted">{card.text}</p>
                <div className="mt-3 font-mono text-xs text-muted">coming soon</div>
              </div>
            ),
          )}
        </section>
      </div>
    </>
  );
}
