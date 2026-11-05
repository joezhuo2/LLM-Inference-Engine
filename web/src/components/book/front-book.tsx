"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Scramble } from "@/components/scramble";
import { kindLabel } from "@/content/dead-ends";
import { usePrefersReducedMotion } from "@/lib/reduced-motion";
import type { BookScene } from "./book-scene";
import { bookOrder, bookParts, chapterPage, chapterTitle, contentsStop, errataIntro, errataPage, errataSpreads, errataStops, errataTitle } from "./front-matter";
import { loadFonts, readFonts } from "./typeset";

function Icon({ d }: { d: string }) {
  return (
    <svg aria-hidden viewBox="0 0 16 16" className="size-4 fill-none stroke-current" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
      <path d={d} />
    </svg>
  );
}

export function FrontBook({ closed = false }: { closed?: boolean }) {
  const reduced = usePrefersReducedMotion();
  const router = useRouter();
  const host = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const scene = useRef<BookScene | null>(null);
  const stops = useMemo(() => [contentsStop(0), ...errataStops(1)], []);
  const total = stops.length;
  const [open, setOpen] = useState(!closed);
  const [stop, setStop] = useState(0);
  const [failed, setFailed] = useState(false);
  const stopRef = useRef(0);

  const go = useCallback(
    (next: number) => {
      const clamped = Math.min(total - 1, Math.max(0, next));
      stopRef.current = clamped;
      setStop(clamped);
      scene.current?.goTo(clamped, 0);
    },
    [total],
  );

  const openBook = useCallback(() => {
    if (!scene.current || scene.current.isOpen) return;
    setOpen(true);
    void scene.current.open();
  }, []);

  const closeBook = useCallback(() => {
    if (!scene.current?.isOpen) return;
    setOpen(false);
    stopRef.current = 0;
    setStop(0);
    void scene.current.close();
  }, []);

  useEffect(() => {
    const canvasEl = canvas.current;
    const hostEl = host.current;
    if (!canvasEl || !hostEl) return;
    let cancelled = false;
    let instance: BookScene | null = null;
    const start = async () => {
      if (!document.createElement("canvas").getContext("webgl2")) {
        setFailed(true);
        return;
      }
      const fonts = readFonts();
      const [{ BookScene: Scene }] = await Promise.all([import("./book-scene"), loadFonts(fonts)]);
      if (cancelled) return;
      try {
        instance = new Scene({
          canvas: canvasEl,
          host: hostEl,
          stops,
          coverTitle: "Every Byte",
          coverSubtitle: "an inference engine, kernel by kernel",
          fonts,
          reduced,
          buildModel: () => ({ exhibits: [], handles: [], update: () => undefined }),
          onOverlay: () => undefined,
          onLens: () => undefined,
          onPageClick: (side) => hostEl.dispatchEvent(new CustomEvent("bookpage", { detail: side })),
          onCoverClick: () => hostEl.dispatchEvent(new CustomEvent("bookcover")),
          onLink: (link) => hostEl.dispatchEvent(new CustomEvent("booklink", { detail: link.href })),
        });
      } catch (error) {
        console.error(error);
        setFailed(true);
        return;
      }
      scene.current = instance;
      if (closed) instance.showClosed();
      else void instance.open();
    };
    void start();
    return () => {
      cancelled = true;
      instance?.dispose();
      scene.current = null;
    };
  }, [reduced, closed, stops]);

  useEffect(() => {
    const hostEl = host.current;
    if (!hostEl) return;
    const onPage = (event: Event) => go(stopRef.current + ((event as CustomEvent<"prev" | "next">).detail === "next" ? 1 : -1));
    const onLink = (event: Event) => {
      const href = (event as CustomEvent<string>).detail;
      if (href === "#errata") go(1);
      else router.push(href);
    };
    hostEl.addEventListener("bookpage", onPage);
    hostEl.addEventListener("bookcover", openBook);
    hostEl.addEventListener("booklink", onLink);
    return () => {
      hostEl.removeEventListener("bookpage", onPage);
      hostEl.removeEventListener("bookcover", openBook);
      hostEl.removeEventListener("booklink", onLink);
    };
  }, [go, openBook, router]);

  const prev = useCallback(() => (stopRef.current === 0 ? closeBook() : go(stopRef.current - 1)), [closeBook, go]);
  const next = useCallback(() => (open ? go(stopRef.current + 1) : openBook()), [open, go, openBook]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "ArrowRight") next();
      if (event.key === "ArrowLeft" && open) prev();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [next, prev, open]);

  const caption = open ? stops[stop].beats[0].caption : "Every Byte: click the cover to open the book";
  const spread = stop > 0 ? errataSpreads[stop - 1] : null;

  if (failed) {
    return (
      <div className="mx-auto max-w-2xl space-y-3 px-4 py-20">
        <h1 className="font-serif text-4xl">This book needs WebGL</h1>
        <p className="text-muted">
          Your browser could not start 3D graphics. The{" "}
          <Link href="/kernels/list" className="link-underline text-text">
            list of kernels
          </Link>{" "}
          has every chapter.
        </p>
      </div>
    );
  }

  return (
    <div className="flex h-[calc(100dvh-3rem)] flex-col overflow-hidden">
      <div ref={host} className="relative min-h-0 flex-1 touch-none bg-[var(--book-desk)]">
        <canvas ref={canvas} role="img" aria-label={open ? `3D book "Every Byte", open at ${caption}` : "3D book \"Every Byte\", closed on the desk"} className="absolute inset-0 size-full" />
        <nav
          aria-label="book"
          className="absolute bottom-4 left-1/2 z-20 flex max-w-[calc(100%-2rem)] -translate-x-1/2 items-center gap-1.5 rounded-full border border-white/30 bg-surface/55 py-1 pl-1.5 pr-4 tabular-nums text-xs text-muted shadow-lg shadow-black/15 backdrop-blur-xl backdrop-saturate-150"
        >
          <button type="button" aria-label={stop === 0 ? "close the book" : "previous page"} disabled={!open} onClick={prev} className="pressable rounded-full p-1.5 enabled:hover:text-text disabled:opacity-30">
            <Icon d="M10 3 5 8l5 5" />
          </button>
          {open && (
            <span className="shrink-0 text-text">
              <Scramble key={stop} text={`${stop + 1} / ${total}`} numeric durationMs={700} />
            </span>
          )}
          <span className="min-w-0 max-w-[22rem] truncate">
            <Scramble key={caption} text={caption} durationMs={900} />
          </span>
          <button type="button" aria-label={open ? "next page" : "open the book"} disabled={open && stop === total - 1} onClick={next} className="pressable rounded-full p-1.5 enabled:hover:text-text disabled:opacity-30">
            <Icon d="M6 3l5 5-5 5" />
          </button>
          <Link href="/kernels/list" className="link-underline ml-1 shrink-0 border-l border-border pl-3 max-sm:hidden">
            list view
          </Link>
        </nav>
      </div>
      <div className="relative border-border bg-surface max-[720px]:border-t">
        <article aria-live="polite" className="max-h-[32dvh] overflow-y-auto px-4 py-3 text-sm leading-relaxed text-muted min-[721px]:sr-only">
          {!open ? (
            <p>
              Every Byte, an inference engine kernel by kernel.{" "}
              <button type="button" onClick={openBook} className="link-underline text-text">
                Open the book
              </button>
            </p>
          ) : !spread ? (
            <>
              <h1 className="font-serif text-2xl text-text">Contents</h1>
              {bookParts.map((part) => (
                <div key={part.numeral} className="mt-2">
                  <h2 className="text-xs">{`Part ${part.numeral} · ${part.title}`}</h2>
                  <ol>
                    {part.slugs.map((slug) => (
                      <li key={slug} className="flex gap-2">
                        <span className="w-5 tabular-nums">{bookOrder.indexOf(slug) + 1}</span>
                        <Link href={`/kernels/${slug}`} className="link-underline text-text">
                          {chapterTitle(slug)}
                        </Link>
                        <span className="ml-auto tabular-nums">{chapterPage(slug)}</span>
                      </li>
                    ))}
                  </ol>
                </div>
              ))}
              <button type="button" onClick={() => go(1)} className="link-underline mt-2 flex w-full text-text">
                {`Errata: ${errataTitle}`}
                <span className="ml-auto tabular-nums text-muted">{errataPage}</span>
              </button>
            </>
          ) : (
            <>
              {!spread.left && (
                <>
                  <h1 className="font-serif text-2xl text-text">Errata</h1>
                  <p className="mt-1">{errataIntro}</p>
                </>
              )}
              {[spread.left, spread.right].map(
                (d) =>
                  d && (
                    <section key={d.title} className="mt-3 space-y-1">
                      <div className="text-xs">
                        {`${d.date} · ${kindLabel[d.kind]} · `}
                        <Link href={`/kernels/${d.slug}`} className="link-underline text-accent">
                          {d.kernel}
                        </Link>
                      </div>
                      <h2 className="font-serif text-xl text-text">{d.title}</h2>
                      <p>{d.tried}</p>
                      <p className="text-text">{d.happened}</p>
                      <p>{d.kept}</p>
                    </section>
                  ),
              )}
            </>
          )}
        </article>
      </div>
    </div>
  );
}
