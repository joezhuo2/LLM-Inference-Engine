"use client";

import clsx from "clsx";
import Link from "next/link";
import { type ReactNode, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { usePrefersReducedMotion } from "@/lib/reduced-motion";
import type { BookScene, OverlayRect, Rect, Stop } from "./book-scene";
import { createRopeModel } from "./rope-model";
import { ropeStops } from "./rope-chapter";
import { loadFonts, readFonts, readRuns } from "./typeset";
import { ropeCaptions, ropeNumbers, ropeSteps, TokensPicker } from "@/components/kernels/rope/story";

type Lens = { x: number; y: number; source: HTMLCanvasElement; u: number; v: number };

const LENS_SIZE = 190;
const beatMs = 6500;

function Icon({ d }: { d: string }) {
  return (
    <svg aria-hidden viewBox="0 0 16 16" className="size-4 fill-none stroke-current" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
      <path d={d} />
    </svg>
  );
}

function Magnifier({ lens }: { lens: Lens | null }) {
  const canvas = useRef<HTMLCanvasElement>(null);
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
      style={{ width: LENS_SIZE, height: LENS_SIZE, left: (lens?.x ?? 0) - LENS_SIZE / 2, top: (lens?.y ?? 0) - LENS_SIZE / 2, opacity: lens ? 1 : 0 }}
      className="pointer-events-none absolute z-10 rounded-full border border-[var(--book-faint)] bg-[var(--book-paper)] shadow-lg transition-opacity duration-300"
    />
  );
}

function FitPage({ rect, width, children, onElement }: { rect: Rect; width: number; children?: ReactNode; onElement?: (el: HTMLDivElement | null) => void }) {
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

const PAGE_DESIGN_WIDTH = 460;

export function BookView() {
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
  const [narrow, setNarrow] = useState(false);

  useEffect(() => {
    const query = window.matchMedia("(max-width: 720px)");
    const update = () => setNarrow(query.matches);
    update();
    query.addEventListener("change", update);
    return () => query.removeEventListener("change", update);
  }, []);


  const total = ropeSteps.length;
  const titles = useMemo(() => ropeSteps.map((s) => s.title), []);

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
      const text = [...bodiesEl.children].map((el, i) => ({ title: ropeSteps[i].title, paragraphs: readRuns(el) }));
      const chapterStops = ropeStops({ index: 3, title: "RoPE", formula: "x′ᵢ = xᵢ cos pθᵢ − xᵢ₊d/₂ sin pθᵢ,   θᵢ = 10000^(−2i/d)" }, text, ropeCaptions);
      setStops(chapterStops);
      try {
        instance = new Scene({
          canvas: canvasEl,
          host: hostEl,
          stops: chapterStops,
          coverTitle: "Every Byte",
          coverSubtitle: "an inference engine, kernel by kernel",
          fonts,
          reduced,
          buildModel: (kit) => createRopeModel(kit, fonts, ropeNumbers),
          onOverlay: setOverlay,
          onLens: setLens,
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
  }, [reduced]);

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
  const step = ropeSteps[stop];

  if (failed) {
    return (
      <div className="mx-auto max-w-2xl space-y-3 px-4 py-20">
        <h1 className="font-serif text-4xl">This book needs WebGL</h1>
        <p className="text-muted">
          Your browser could not start 3D graphics. The{" "}
          <Link href="/kernels/rope" className="link-underline text-text">
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
        {ropeSteps.map((s) => (
          <div key={s.id}>{s.body}</div>
        ))}
      </div>
      <div ref={host} className="relative min-h-0 flex-1 touch-none bg-[var(--book-desk)]">
        <canvas ref={canvas} role="img" aria-label={`3D book, open at ${titles[stop]}. ${caption}`} className="absolute inset-0 size-full" />
        <Magnifier lens={lens} />
        {overlay && narrow && (
          <div data-theme="light" className="book-overlay absolute inset-0 touch-auto overflow-y-auto bg-[var(--book-paper)] p-5 text-[var(--text)] [scrollbar-width:none]">
            <TokensPicker />
          </div>
        )}
        {overlay && !narrow && (
          <div data-theme="light" key={overlay.kind} className="book-overlay touch-auto text-[var(--text)]">
            <FitPage rect={overlay.rect} width={PAGE_DESIGN_WIDTH}>
              <div className="h-[34rem]">
                <TokensPicker />
              </div>
            </FitPage>
          </div>
        )}
      </div>
      <div className="relative border-t border-border bg-surface">
        <article aria-live="polite" className="max-h-[32dvh] overflow-y-auto px-4 py-3 text-sm leading-relaxed text-muted min-[721px]:sr-only">
          {step && (
            <>
              <h1 className="font-serif text-2xl text-text">{step.title}</h1>
              <div className="mt-2 space-y-2">{step.body}</div>
            </>
          )}
        </article>
        <nav aria-label="book" className="flex items-center gap-2 px-3 py-2 tabular-nums text-xs text-muted">
          <button type="button" aria-label="previous page" disabled={stop === 0} onClick={() => go(stop - 1)} className="pressable rounded p-1.5 enabled:hover:text-text disabled:opacity-30">
            <Icon d="M10 3 5 8l5 5" />
          </button>
          <button type="button" aria-label={playing ? "pause the tour" : "play the tour"} disabled={beatCount < 2} onClick={togglePlay} className="pressable rounded p-1.5 enabled:hover:text-text disabled:opacity-30">
            <Icon d={playing ? "M5 3v10M11 3v10" : "M5 3l8 5-8 5z"} />
          </button>
          <span className="shrink-0 text-text">
            {stop + 1} / {total}
          </span>
          <span className="min-w-0 flex-1 truncate">{caption}</span>
          <span className="flex shrink-0 items-center gap-0.5">
            {beats.length > 1 &&
              beats.map((b, i) => (
                <button key={b.caption} type="button" aria-label={b.caption} aria-current={i === beat} onClick={() => pickBeat(i)} className="p-1.5">
                  <span className={clsx("block size-1.5 rounded-full transition-colors duration-300", i === beat ? "bg-accent" : "bg-border")} />
                </button>
              ))}
          </span>
          <button type="button" aria-label="next page" disabled={stop === total - 1} onClick={() => go(stop + 1)} className="pressable rounded p-1.5 enabled:hover:text-text disabled:opacity-30">
            <Icon d="M6 3l5 5-5 5" />
          </button>
          <Link href="/kernels/rope" className="link-underline ml-1 shrink-0 max-sm:hidden">
            reading view
          </Link>
        </nav>
      </div>
    </div>
  );
}
