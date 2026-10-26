import clsx from "clsx";
import type { Metadata } from "next";
import Link from "next/link";
import { LayerMap } from "@/components/kernels/layer-map";
import { PageTitle } from "@/components/page-title";
import { Scramble } from "@/components/scramble";
import { kernelCards } from "@/content/kernels";

export const metadata: Metadata = { title: "kernels" };

export default function Kernels() {
  return (
    <div className="space-y-10">
      <header className="space-y-3">
        <PageTitle text="kernels" />
        <p className="max-w-2xl text-muted">
          How each kernel of the decode step works and how it got faster, version by version, with the measurements
          behind every change.
        </p>
      </header>
      <LayerMap />
      <ol className="grid gap-4 sm:grid-cols-2">
        {kernelCards.map((k, i) => {
          const body = (
            <>
              <div className="font-mono text-sm">
                <span className="text-muted">{String(i + 1).padStart(2, "0")} </span>
                <Scramble text={k.title} trigger="view" delayMs={i * 80} durationMs={900} className={k.ready ? "text-accent" : "text-text"} />
                {!k.ready && <span className="ml-2 text-xs text-muted">soon</span>}
              </div>
              <p className="mt-2 text-sm text-muted">{k.short}</p>
            </>
          );
          return (
            <li key={k.slug}>
              {k.ready ? (
                <Link href={`/kernels/${k.slug}`} className="pressable hover-grow block h-full rounded border border-border bg-surface p-4">
                  {body}
                </Link>
              ) : (
                <div className={clsx("h-full rounded border border-dashed border-border p-4")}>{body}</div>
              )}
            </li>
          );
        })}
      </ol>
    </div>
  );
}
