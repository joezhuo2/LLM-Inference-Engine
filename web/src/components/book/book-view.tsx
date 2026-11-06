"use client";

import clsx from "clsx";
import Link from "next/link";
import { type ReactNode, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { Scramble } from "@/components/scramble";
import { usePrefersReducedMotion } from "@/lib/reduced-motion";
import type { BookScene, OverlayRect, Rect, Stop } from "./book-scene";
import { type ChapterDef, chapterStops, chapterTitles } from "./chapter";
import { loadFonts, readFonts, readRuns } from "./typeset";

export type Lens = { x: number; y: number; source: HTMLCanvasElement; u: number; v: number };

const LENS_SIZE = 190;
const beatMs = 6500;

function Icon({ d }: { d: string }) {
  return (
    <svg aria-hidden viewBox="0 0 16 16" className="size-4 fill-none stroke-current" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
      <path d={d} />
    </svg>
  );
}

export function Magnifier({ lens }: { lens: Lens | null }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [at, setAt] = useState({ x: 0, y: 0 });
  if (lens && (lens.x !== at.x || lens.y !== at.y)) setAt({ x: lens.x, y: lens.y });
  useEffect(() => {
    const c = canvas.current;
    if (!c || !lens) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    const zoom = lens.source.width * 0.38;
    const sx = lens.u * lens.source.width - zoom / 2;
    const sy = lens.v * lens.source.height - zoom / 2;
    ctx.clearRect(0, 0, c.width, c.height);
    ctx.drawImage(lens.source, sx, sy, zoom, zoom, 0, 0, c.width, c.height);
  }, [lens]);
  return (
    <canvas
      ref={canvas}
      aria-hidden
      width={LENS_SIZE * 2}
      height={LENS_SIZE * 2}
      style={{ width: LENS_SIZE, height: LENS_SIZE, left: at.x - LENS_SIZE / 2, top: at.y - LENS_SIZE / 2, opacity: lens ? 1 : 0 }}
      className="pointer-events-none absolute z-10 rounded-full border border-[var(--book-faint)] bg-[var(--book-paper)] shadow-lg transition-opacity duration-300"
    />
  );
}

export function FitPage({ rect, width, children, onElement }: { rect: Rect; width: number; children?: ReactNode; onElement?: (el: HTMLDivElement | null) => void }) {
  const inner = useRef<HTMLDivElement | null>(null);
  const attach = useCallback(
    (el: HTMLDivElement | null) => {
      inner.current = el;
      onElement?.(el);
    },
    [onElement],
  );
  const [fit, setFit] = useState({ scale: rect.width / width, width });
  useLayoutEffect(() => {
    const el = inner.current;
    if (!el) return;
    const measure = () => {
      let w = width;
      let scale = rect.width / w;
      for (let k = 0; k < 4; k++) {
        el.style.width = `${w}px`;
        const h = el.scrollHeight;
        if (h * scale <= rect.height) break;
        scale = Math.min(rect.width / width, rect.height / h);
        w = rect.width / scale;
      }
      setFit((old) => (Math.abs(old.scale - scale) < 0.002 && Math.abs(old.width - w) < 1 ? old : { scale, width: w }));
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect();
  }, [rect.width, rect.height, width]);
  return (
    <div style={{ left: rect.left, top: rect.top, width: rect.width, height: rect.height }} className="absolute overflow-hidden">
      <div
        ref={attach}
        style={{ width: fit.width, transform: `scale(${fit.scale})`, transformOrigin: "0 0" }}
      >
        {children}
      </div>
    </div>
  );
}

export const PAGE_DESIGN_WIDTH = 460;

export function BookView({ chapter }: { chapter: ChapterDef }) {
  const reduced = usePrefersReducedMotion();
  const host = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const bodies = useRef<HTMLDivElement>(null);
  const scene = useRef<BookScene | null>(null);
  const [stops, setStops] = useState<Stop[] | null>(null);
  const [stop, setStop] = useState(0);
  const [beat, setBeat] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [overlay, setOverlay] = useState<OverlayRect | null>(null);
  const [lens, setLens] = useState<Lens | null>(null);
  const [failed, setFailed] = useState(false);
  const [pageEl, setPageEl] = useState<HTMLDivElement | null>(null);
  const [narrow, setNarrow] = useState(false);
  const barInner = useRef<HTMLDivElement>(null);
  const [barWidth, setBarWidth] = useState<number | null>(null);

  useEffect(() => {
    const el = barInner.current;
    if (!el) return;
    const observer = new ResizeObserver(() => setBarWidth(el.offsetWidth + 2));
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const query = window.matchMedia("(max-width: 720px)");
    const update = () => setNarrow(query.matches);
    update();
    query.addEventListener("change", update);
    return () => query.removeEventListener("change", update);
  }, []);


  const titles = useMemo(() => chapterTitles(chapter), [chapter]);
  const total = titles.length;
  const readHref = `/kernels/${chapter.slug}`;

  useEffect(() => {
    const canvasEl = canvas.current;
    const hostEl = host.current;
    const bodiesEl = bodies.current;
    if (!canvasEl || !hostEl || !bodiesEl) return;
    let cancelled = false;
    let instance: BookScene | null = null;
    const start = async () => {
      const probe = document.createElement("canvas").getContext("webgl2");
      if (!probe) {
        setFailed(true);
        return;
      }
      const fonts = readFonts();
      const [{ BookScene: Scene }] = await Promise.all([import("./book-scene"), loadFonts(fonts)]);
      if (cancelled) return;
      const text = [...bodiesEl.children].map((el, i) => ({ title: chapter.steps[i].title, paragraphs: readRuns(el) }));
      const builtStops = chapterStops(chapter, text);
      setStops(builtStops);
      try {
        instance = new Scene({
          canvas: canvasEl,
          host: hostEl,
          stops: builtStops,
          coverTitle: "Every Byte",
          coverSubtitle: "an inference engine, kernel by kernel",
          fonts,
          reduced,
          buildModel: (kit) => chapter.buildModel(kit, fonts),
          onOverlay: setOverlay,
          onLens: setLens,
          onPageClick: (side) => hostEl.dispatchEvent(new CustomEvent("bookpage", { detail: side })),
        });
      } catch (error) {
        console.error(error);
        setFailed(true);
        return;
      }
      scene.current = instance;
      void instance.open();
    };
    void start();
    return () => {
      cancelled = true;
      instance?.dispose();
      scene.current = null;
    };
  }, [reduced, chapter]);

  const go = useCallback(
    (next: number) => {
      const clamped = Math.min(total - 1, Math.max(0, next));
      setStop(clamped);
      setBeat(0);
      setPlaying(false);
      scene.current?.goTo(clamped, 0);
    },
    [total],
  );

  useEffect(() => {
    const hostEl = host.current;
    if (!hostEl) return;
    const onPage = (event: Event) => go(stop + ((event as CustomEvent<"prev" | "next">).detail === "next" ? 1 : -1));
    hostEl.addEventListener("bookpage", onPage);
    return () => hostEl.removeEventListener("bookpage", onPage);
  }, [go, stop]);

  const pickBeat = useCallback(
    (next: number) => {
      setBeat(next);
      scene.current?.goTo(stop, next);
    },
    [stop],
  );

  const beats = stops?.[stop]?.beats ?? [];
  const beatCount = beats.length;

  const togglePlay = () => {
    if (playing) {
      setPlaying(false);
      return;
    }
    setPlaying(true);
    pickBeat((beat + 1) % beatCount);
  };

  useEffect(() => {
    if (!playing || beatCount < 2) return;
    const id = window.setTimeout(() => {
      const next = (beat + 1) % beatCount;
      setBeat(next);
      scene.current?.goTo(stop, next);
      if (next === 0) setPlaying(false);
    }, beatMs);
    return () => window.clearTimeout(id);
  }, [playing, beat, beatCount, stop]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target;
      if (target instanceof Element && target.closest("input, textarea, select, [role='radiogroup'], [role='radio']")) return;
      if (event.key === "ArrowRight") go(stop + 1);
      if (event.key === "ArrowLeft") go(stop - 1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [go, stop]);

  const caption = beats[beat]?.caption ?? titles[stop];
  const step = chapter.steps[stop];
  const bench = chapter.workbench[stop - chapter.steps.length];
  const Overlay = chapter.Overlay;

  if (failed) {
    return (
      <div className="mx-auto max-w-2xl space-y-3 px-4 py-20">
        <h1 className="font-serif text-4xl">This book needs WebGL</h1>
        <p className="text-muted">
          Your browser could not start 3D graphics. The{" "}
          <Link href={readHref} className="link-underline text-text">
            reading view
          </Link>{" "}
          has the same text, figures and sandbox.
        </p>
      </div>
    );
  }

  return (
    <div className="flex h-[calc(100dvh-3rem)] flex-col overflow-hidden">
      <div ref={bodies} hidden>
        {chapter.steps.map((s) => (
          <div key={s.id}>{s.body}</div>
        ))}
      </div>
      <div ref={host} className="relative min-h-0 flex-1 touch-none bg-[var(--book-desk)]">
        <canvas ref={canvas} role="img" aria-label={`3D book, open at ${titles[stop]}. ${caption}`} className="absolute inset-0 size-full" />
        <Magnifier lens={lens} />
          <nav
            aria-label="book"
            style={{ width: barWidth ?? undefined }}
            className="absolute bottom-4 left-1/2 z-20 max-w-[calc(100%-2rem)] -translate-x-1/2 overflow-hidden rounded-full border border-white/30 bg-surface/55 tabular-nums text-xs text-muted shadow-lg shadow-black/15 backdrop-blur-xl backdrop-saturate-150 transition-[width] duration-700 ease-[var(--ease-in-out)] motion-reduce:transition-none"
          >
            <div ref={barInner} className="flex w-max items-center gap-1.5 py-1 pl-1.5 pr-4">
            <button type="button" aria-label="previous page" disabled={stop === 0} onClick={() => go(stop - 1)} className="pressable rounded-full p-1.5 enabled:hover:text-text disabled:opacity-30">
              <Icon d="M10 3 5 8l5 5" />
            </button>
            <button type="button" aria-label={playing ? "pause the tour" : "play the tour"} disabled={beatCount < 2} onClick={togglePlay} className="pressable rounded-full p-1.5 enabled:hover:text-text disabled:opacity-30">
              <Icon d={playing ? "M5 3v10M11 3v10" : "M5 3l8 5-8 5z"} />
            </button>
            <span className="shrink-0 text-text">
              <Scramble key={stop} text={`${stop + 1} / ${total}`} numeric durationMs={700} />
            </span>
            <span className="min-w-0 max-w-[22rem] truncate">
                <Scramble key={caption} text={caption} durationMs={900} />
              </span>
            <span className="flex shrink-0 items-center gap-0.5">
              {beats.length > 1 &&
                beats.map((b, i) => (
                  <button key={b.caption} type="button" aria-label={b.caption} aria-current={i === beat} onClick={() => pickBeat(i)} className="p-1.5">
                    <span className={clsx("block size-1.5 rounded-full transition-colors duration-300", i === beat ? "bg-accent" : "bg-border")} />
                  </button>
                ))}
            </span>
            <button type="button" aria-label="next page" disabled={stop === total - 1} onClick={() => go(stop + 1)} className="pressable rounded-full p-1.5 enabled:hover:text-text disabled:opacity-30">
              <Icon d="M6 3l5 5-5 5" />
            </button>
            <Link href={readHref} className="link-underline ml-1 shrink-0 border-l border-border pl-3 max-sm:hidden">
              reading view
            </Link>
            </div>
          </nav>
        {overlay && narrow && (
          <div data-theme="light" className="book-overlay absolute inset-0 touch-auto overflow-y-auto bg-[var(--book-paper)] p-5 text-[var(--text)] [scrollbar-width:none]">
            <div ref={setPageEl} />
          </div>
        )}
        {overlay && !narrow && (
          <div data-theme="light" key={overlay.kind} className="book-overlay touch-auto text-[var(--text)]">
            <FitPage rect={overlay.rect} width={PAGE_DESIGN_WIDTH} onElement={setPageEl} />
          </div>
        )}
        {Overlay && <Overlay kind={overlay?.kind ?? null} target={overlay ? pageEl : null} narrow={narrow} />}
      </div>
      <div className="relative border-border bg-surface max-[720px]:border-t">
        <article aria-live="polite" className="max-h-[32dvh] overflow-y-auto px-4 py-3 text-sm leading-relaxed text-muted min-[721px]:sr-only">
          {step ? (
            <>
              <h1 className="font-serif text-2xl text-text">{step.title}</h1>
              <div className="mt-2 space-y-2">{step.body}</div>
            </>
          ) : (
            <>
              <h1 className="font-serif text-2xl text-text">{titles[stop]}</h1>
              <div className="mt-2 space-y-2">
                {bench?.paragraphs.map((p) => (
                  <p key={p}>{p}</p>
                ))}
              </div>
            </>
          )}
        </article>
      </div>
    </div>
  );
}
