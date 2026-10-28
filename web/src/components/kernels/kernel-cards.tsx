"use client";

import clsx from "clsx";
import Link from "next/link";
import { useState, type PointerEvent } from "react";
import { Scramble } from "@/components/scramble";
import { kernelCards } from "@/content/kernels";
import { setHoveredKernel, useHoveredKernel } from "@/lib/kernel-hover";

export function KernelCards() {
  const hovered = useHoveredKernel();
  const [plays, setPlays] = useState<Record<string, number>>({});
  const enter = (slug: string) => {
    setHoveredKernel(slug);
    setPlays((p) => ({ ...p, [slug]: (p[slug] ?? 0) + 1 }));
  };
  const hover = (slug: string) => ({
    onPointerEnter: (e: PointerEvent) => e.pointerType === "mouse" && enter(slug),
    onPointerLeave: () => setHoveredKernel(null),
    onFocus: () => enter(slug),
    onBlur: () => setHoveredKernel(null),
  });
  return (
    <ol className="isolate flex flex-col">
      {kernelCards.map((k, i) => {
        const open = hovered === k.slug;
        const replay = plays[k.slug] ?? 0;
        const body = (
          <>
            <div className="font-mono text-sm">
              <span className="text-muted">{String(i + 1).padStart(2, "0")} </span>
              <Scramble
                text={k.title}
                trigger="view"
                delayMs={replay ? 0 : i * 80}
                durationMs={900}
                replay={replay}
                className={clsx(
                  "underline decoration-transparent underline-offset-4 transition-[text-decoration-color] duration-200",
                  open && "decoration-current",
                  k.ready ? "text-accent" : "text-text",
                )}
              />
              {!k.ready && <span className="ml-2 text-xs text-muted">soon</span>}
            </div>
            <div
              className={clsx(
                "grid transition-[grid-template-rows,opacity] duration-300 ease-[var(--ease-out)] motion-reduce:transition-none",
                open ? "grid-rows-[1fr] opacity-100" : "grid-rows-[0fr] opacity-0",
              )}
            >
              <div className="overflow-hidden">
                <p className="pt-2 text-sm text-muted">{k.short}</p>
                {k.dates && <p className="pt-1 font-mono text-xs tabular-nums text-muted">{k.dates}</p>}
              </div>
            </div>
          </>
        );
        const card = clsx(
          "block rounded border bg-surface px-4 py-3 shadow-[0_-4px_8px_-6px_rgba(0,0,0,0.4)]",
          k.ready ? "border-border" : "border-dashed border-border",
        );
        return (
          <li key={k.slug} className={clsx("relative", i > 0 && "-mt-2", open && "z-10")}>
            {k.ready ? (
              <Link href={`/kernels/${k.slug}`} className={clsx(card, "pressable")} {...hover(k.slug)}>
                {body}
              </Link>
            ) : (
              <div className={card} {...hover(k.slug)}>
                {body}
              </div>
            )}
          </li>
        );
      })}
    </ol>
  );
}
