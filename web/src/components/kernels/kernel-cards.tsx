"use client";

import Link from "next/link";
import type { PointerEvent } from "react";
import { Scramble } from "@/components/scramble";
import { kernelCards } from "@/content/kernels";
import { setHoveredKernel } from "@/lib/kernel-hover";

export function KernelCards() {
  const hover = (slug: string) => ({
    onPointerEnter: (e: PointerEvent) => e.pointerType === "mouse" && setHoveredKernel(slug),
    onPointerLeave: () => setHoveredKernel(null),
    onFocus: () => setHoveredKernel(slug),
    onBlur: () => setHoveredKernel(null),
  });
  return (
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
              <Link href={`/kernels/${k.slug}`} className="pressable hover-grow block h-full rounded border border-border bg-surface p-4" {...hover(k.slug)}>
                {body}
              </Link>
            ) : (
              <div className="h-full rounded border border-dashed border-border p-4" {...hover(k.slug)}>
                {body}
              </div>
            )}
          </li>
        );
      })}
    </ol>
  );
}
