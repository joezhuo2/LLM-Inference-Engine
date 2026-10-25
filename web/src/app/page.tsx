import Link from "next/link";
import { Hero } from "@/components/hero";

const cards = [
  { href: "/results", title: "results", text: "Throughput and latency against HuggingFace and llama.cpp." },
  { href: "/kernels", title: "kernels", text: "How each kernel got faster, with Nsight numbers." },
  { href: "/dashboard", title: "dashboard", text: "A recorded serving run, step by step." },
];

export default function Overview() {
  return (
    <>
      <Hero />
      <div className="mx-auto w-full max-w-5xl px-4 py-16 sm:px-6">
        <section className="grid gap-4 sm:grid-cols-3">
          {cards.map((card) => (
            <Link key={card.href} href={card.href} className="pressable block rounded border border-border bg-surface p-4">
              <div className="font-mono text-sm text-accent">{card.title}</div>
              <p className="mt-2 text-sm text-muted">{card.text}</p>
            </Link>
          ))}
        </section>
      </div>
    </>
  );
}
