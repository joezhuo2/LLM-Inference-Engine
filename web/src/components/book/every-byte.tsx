"use client";

import clsx from "clsx";
import { motion } from "motion/react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { type MouseEvent, useCallback, useEffect, useRef, useState } from "react";
import type { Group, Object3D } from "three";
import { GlossProvider, type PointTerm, PointGloss } from "@/components/gloss";
import { gloss } from "@/components/glossed";
import { BookText } from "@/components/kernels/authorship";
import { Scramble } from "@/components/scramble";
import { kindLabel } from "@/content/dead-ends";
import { chapterExercises } from "@/content/exercises";
import { usePrefersReducedMotion } from "@/lib/reduced-motion";
import type { BookModel, BookScene, OverlayRect, Stop } from "./book-scene";
import { FitPage, type Lens, Magnifier, PAGE_DESIGN_WIDTH } from "./book-view";
import { type ChapterDef, chapterStops, stepNumber } from "./chapter";
import { bookChapters } from "./chapters";
import { bookOrder, bookParts, bookTitles, chapterPage, chapterTitle, contentsStop, errataIntro, errataSpreads, errataStops, errataTitle, exerciseStops, introStop, numberPages, startStops, startText } from "./front-matter";
import type { Kit } from "./ink";
import { type Fonts, loadFonts, readFonts, readRuns } from "./typeset";
import { BOOK, bookHref } from "./paths";

const chapterSlugs = bookOrder.filter((slug) => slug in bookChapters);
const sectionKeys = ["start", "contents", ...chapterSlugs, "errata"];
const beatMs = 6500;
let numbered: Promise<number> | undefined;
const numberBook = () =>
  (numbered ??= Promise.all(chapterSlugs.map((slug) => bookChapters[slug]().then((c) => [slug, 1 + c.steps.length + c.workbench.length + chapterExercises(slug).length] as const))).then((pairs) => numberPages(Object.fromEntries(pairs))));

type Meta = { key: string; local: number; slug?: string; exhibit: number | null };
type Place = { closed: true } | { closed: false; key: string; local: number };

const urlOf = bookHref;

function parsePlace(): Place {
  const path = window.location.pathname.replace(/\/$/, "");
  if (path === BOOK) return { closed: true };
  const slug = path.split("/")[2];
  if (slug && chapterSlugs.includes(slug)) return { closed: false, key: slug, local: 0 };
  if (slug === "start") return { closed: false, key: "start", local: 0 };
  return { closed: false, key: window.location.hash === "#errata" ? "errata" : "contents", local: 0 };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((r) => (resolve = r));
  return { promise, resolve };
}

function Icon({ d }: { d: string }) {
  return (
    <svg aria-hidden viewBox="0 0 16 16" className="size-4 fill-none stroke-current" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
      <path d={d} />
    </svg>
  );
}

function compose(raw: Record<string, Stop[]>) {
  const keys = ["start", "contents", ...chapterSlugs.filter((s) => raw[s]), "errata"];
  const stops: Stop[] = [];
  const meta: Meta[] = [];
  for (const key of keys) {
    const slug = key === "start" || key === "contents" || key === "errata" ? undefined : key;
    raw[key].forEach((s, i) => {
      const folios = slug && i > 0 && s.page ? { page: { ...s.page, folio: String((chapterPage(key) ?? 0) + i) } } : {};
      meta.push({ key, local: slug ? i - 1 : i, slug, exhibit: slug ? s.exhibit : null });
      stops.push({ ...s, ...folios, spread: stops.length, exhibit: slug && s.exhibit !== null ? chapterSlugs.indexOf(slug) : null });
    });
  }
  return { stops, meta };
}

const base: Record<string, Stop[]> = { start: startStops(), contents: [contentsStop()], errata: errataStops(0) };
const initial = compose(base);

export function EveryByte() {
  const reduced = usePrefersReducedMotion();
  const router = useRouter();
  const host = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const bodies = useRef<HTMLDivElement>(null);
  const scene = useRef<BookScene | null>(null);
  const raw = useRef<Record<string, Stop[]>>({ ...base });
  const book = useRef(initial);
  const models = useRef<Record<string, BookModel>>({});
  const containers = useRef<Group[]>([]);
  const handles = useRef<Object3D[]>([]);
  const kit = useRef<{ kit: Kit; fonts: Fonts } | null>(null);
  const queue = useRef(Promise.resolve());
  const ready = useRef<Record<string, Promise<void>>>({});
  const readyResolve = useRef<Record<string, () => void>>({});
  const sceneReady = useRef(deferred<BookScene>());
  const defsRef = useRef<Record<string, ChapterDef>>({});
  const queued = useRef(new Set<string>());
  const stopRef = useRef(0);
  const [defs, setDefs] = useState<Record<string, ChapterDef>>({});
  const [layout, setLayout] = useState(initial);
  const stops = layout.stops;
  const [open, setOpen] = useState(false);
  const [stop, setStop] = useState(0);
  const [beat, setBeat] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [overlay, setOverlay] = useState<OverlayRect | null>(null);
  const [lens, setLens] = useState<Lens | null>(null);
  const [term, setTerm] = useState<PointTerm | null>(null);
  const [failed, setFailed] = useState(false);
  const [pageEl, setPageEl] = useState<HTMLDivElement | null>(null);
  const [narrow, setNarrow] = useState(false);
  const [total, setTotal] = useState(0);

  useEffect(() => void numberBook().then(setTotal), []);

  const meta = layout.meta[stop];
  const segment = meta?.key ?? "contents";
  const page = open && meta ? (chapterPage(meta.key) ?? 0) + stop - layout.meta.findIndex((m) => m.key === meta.key) : 1;
  const def = meta?.slug ? defs[meta.slug] : undefined;

  useEffect(() => {
    const query = window.matchMedia("(max-width: 720px)");
    const update = () => setNarrow(query.matches);
    update();
    query.addEventListener("change", update);
    return () => query.removeEventListener("change", update);
  }, []);

  const attach = useCallback((slug: string) => {
    const c = defsRef.current[slug];
    if (!c || !kit.current || models.current[slug]) return;
    const model = c.buildModel(kit.current.kit, kit.current.fonts);
    models.current[slug] = model;
    handles.current.push(...model.handles);
    const container = containers.current[chapterSlugs.indexOf(slug)];
    for (const exhibit of model.exhibits) {
      exhibit.visible = false;
      container.add(exhibit);
    }
  }, []);

  const ensure = useCallback((slug: string): Promise<void> => {
    if (!chapterSlugs.includes(slug)) return Promise.resolve();
    if (!ready.current[slug]) {
      ready.current[slug] = new Promise((resolve) => (readyResolve.current[slug] = resolve));
      void bookChapters[slug]().then((c) => {
        defsRef.current = { ...defsRef.current, [slug]: c };
        setDefs(defsRef.current);
      });
    }
    return ready.current[slug];
  }, []);

  useEffect(() => {
    const el = bodies.current;
    if (!el) return;
    for (const [slug, c] of Object.entries(defs)) {
      if (raw.current[slug] || queued.current.has(slug)) continue;
      const box = el.querySelector(`[data-chapter="${slug}"]`);
      if (!box) continue;
      const text = [...box.children].map((child, i) => ({ title: c.steps[i].title, paragraphs: readRuns(child) }));
      const chapterRaw = [introStop(c), ...chapterStops(c, text), ...exerciseStops(c)];
      queued.current.add(slug);
      queue.current = queue.current.then(async () => {
        const s = await sceneReady.current.promise;
        await numberBook();
        await s.idle();
        attach(slug);
        const before = book.current.meta.map((m) => `${m.key}:${m.local}`);
        raw.current[slug] = chapterRaw;
        book.current = compose(raw.current);
        const index = new Map(book.current.meta.map((m, i) => [`${m.key}:${m.local}`, i]));
        const map = (i: number) => index.get(before[i]) ?? i;
        s.setStops(book.current.stops, map);
        stopRef.current = map(stopRef.current);
        setStop(stopRef.current);
        setLayout(book.current);
        readyResolve.current[slug]?.();
      });
    }
  }, [defs, attach]);

  const firstOf = (key: string, last = false) => {
    const all = book.current.meta.map((m, i) => (m.key === key ? i : -1)).filter((i) => i >= 0);
    return last ? all[all.length - 1] : all[0];
  };

  const setUrl = useCallback((index: number, push: boolean) => {
    const key = book.current.meta[index]?.key;
    if (!key || !push) return;
    const url = urlOf(key);
    if (window.location.pathname + window.location.hash !== url) window.history.pushState(null, "", url);
  }, []);

  const preload = useCallback(
    (index: number) => {
      const slug = book.current.meta[index]?.slug;
      const at = slug ? chapterSlugs.indexOf(slug) : book.current.meta[index]?.key === "errata" ? chapterSlugs.length : -1;
      for (const n of [chapterSlugs[at - 1], chapterSlugs[at + 1]]) if (n) void ensure(n);
    },
    [ensure],
  );

  const go = useCallback(
    (next: number, riffle = false, push = true) => {
      const clamped = Math.min(book.current.stops.length - 1, Math.max(0, next));
      stopRef.current = clamped;
      setStop(clamped);
      setBeat(0);
      setPlaying(false);
      setLens(null);
      scene.current?.goTo(clamped, 0, riffle);
      setUrl(clamped, push);
      preload(clamped);
    },
    [setUrl, preload],
  );

  const goKey = useCallback(
    async (key: string, riffle: boolean, push = true, last = false) => {
      await Promise.all([ensure(key), numberBook()]);
      const index = firstOf(key, last);
      if (scene.current && !scene.current.isOpen) {
        stopRef.current = index;
        setStop(index);
        setOpen(true);
        setUrl(index, push);
        preload(index);
        void scene.current.open(index);
        return;
      }
      go(index, riffle, push);
    },
    [ensure, go, setUrl, preload],
  );

  const closeBook = useCallback(
    (push = true) => {
      if (!scene.current?.isOpen) return;
      setOpen(false);
      setPlaying(false);
      stopRef.current = 0;
      setStop(0);
      void scene.current.close();
      if (push && window.location.pathname !== BOOK) window.history.pushState(null, "", BOOK);
    },
    [],
  );

  const step = useCallback(
    (delta: 1 | -1) => {
      const i = stopRef.current;
      const m = book.current.meta[i];
      const edge = book.current.meta[i + delta]?.key !== m.key;
      if (edge) {
        const keys = sectionKeys;
        const nextKey = keys[keys.indexOf(m.key) + delta];
        if (nextKey && chapterSlugs.includes(nextKey) && !raw.current[nextKey]) {
          void goKey(nextKey, false, true, delta < 0);
          return;
        }
      }
      if (i === 0 && delta < 0) closeBook();
      else go(i + delta);
    },
    [go, goKey, closeBook],
  );

  const jump = useCallback(
    (delta: 1 | -1) => {
      const key = book.current.meta[stopRef.current]?.key ?? "contents";
      const next = sectionKeys[sectionKeys.indexOf(key) + delta];
      if (next) void goKey(next, true);
    },
    [goKey],
  );

  const barRef = useRef<HTMLDivElement>(null);
  const [barWidth, setBarWidth] = useState<number>();
  useEffect(() => {
    const el = barRef.current;
    if (!el) return;
    const observer = new ResizeObserver(() => setBarWidth(el.offsetWidth + 2));
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const turnTo = useCallback((slug: string, k: number) => go(firstOf(slug) + 1 + k, k > 0), [go]);

  const onLinkHref = useCallback(
    (href: string) => {
      if (href === "#errata") return void goKey("errata", true);
      const [path, at] = href.split("#");
      const slug = path.startsWith(`${BOOK}/`) ? path.slice(BOOK.length + 1) : "";
      if (at && chapterSlugs.includes(slug)) return turnTo(slug, Number(at));
      if (chapterSlugs.includes(slug)) return void goKey(slug, true);
      router.push(href);
    },
    [goKey, router, turnTo],
  );

  useEffect(() => {
    const canvasEl = canvas.current;
    const hostEl = host.current;
    if (!canvasEl || !hostEl) return;
    let cancelled = false;
    let instance: BookScene | null = null;
    const place = parsePlace();
    if (!place.closed) void ensure(place.key);
    const start = async () => {
      if (!document.createElement("canvas").getContext("webgl2")) {
        setFailed(true);
        return;
      }
      const fonts = readFonts();
      const [{ BookScene: Scene }, { Group: ThreeGroup }] = await Promise.all([import("./book-scene"), import("three"), loadFonts(fonts)]);
      if (cancelled) return;
      containers.current = chapterSlugs.map(() => new ThreeGroup());
      models.current = {};
      handles.current.length = 0;
      let active: string | undefined;
      const composite: BookModel = {
        exhibits: containers.current,
        handles: handles.current,
        dragStart: (h) => (active ? models.current[active]?.dragStart?.(h) : undefined),
        drag: (x) => (active ? models.current[active]?.drag?.(x) : undefined),
        dragEnd: () => (active ? models.current[active]?.dragEnd?.() : undefined),
        enter: (i) => {
          const m = book.current.meta[i];
          active = m?.slug;
          if (!m?.slug) return;
          containers.current[chapterSlugs.indexOf(m.slug)].children.forEach((child, k) => (child.visible = k === m.exhibit));
          if (m.local >= 0) models.current[m.slug]?.enter?.(m.local);
        },
        update: (dt, _exhibit, i, isReduced) => {
          const m = book.current.meta[i];
          if (m?.slug && m.exhibit !== null) models.current[m.slug]?.update(dt, m.exhibit, m.local, isReduced);
        },
      };
      try {
        instance = new Scene({
          canvas: canvasEl,
          host: hostEl,
          stops: book.current.stops,
          coverTitle: "Every Byte",
          coverSubtitle: "an inference engine, kernel by kernel",
          fonts,
          reduced,
          buildModel: (k) => {
            kit.current = { kit: k, fonts };
            return composite;
          },
          onOverlay: setOverlay,
          onLens: setLens,
          onTerm: setTerm,
          onPageClick: (side) => hostEl.dispatchEvent(new CustomEvent("bookpage", { detail: side })),
          onCoverClick: () => hostEl.dispatchEvent(new CustomEvent("bookcover")),
          onLink: (link) => hostEl.dispatchEvent(new CustomEvent("booklink", { detail: link.href })),
        });
      } catch (error) {
        console.error(error);
        setFailed(true);
        return;
      }
      const built = instance;
      const addModels = () => {
        for (const slug of Object.keys(raw.current)) attach(slug);
      };
      addModels();
      scene.current = built;
      sceneReady.current.resolve(built);
      queue.current = queue.current.then(addModels);
      built.showClosed();
      if (!place.closed) void goKey(place.key, false, false);
    };
    void start();
    return () => {
      cancelled = true;
      instance?.dispose();
      scene.current = null;
      kit.current = null;
      sceneReady.current = deferred();
      setOpen(false);
    };
  }, [reduced, ensure, goKey, attach]);

  useEffect(() => {
    const hostEl = host.current;
    if (!hostEl) return;
    const onPage = (event: Event) => step((event as CustomEvent<"prev" | "next">).detail === "next" ? 1 : -1);
    const onCover = () => void goKey("start", false);
    const onLink = (event: Event) => onLinkHref((event as CustomEvent<string>).detail);
    hostEl.addEventListener("bookpage", onPage);
    hostEl.addEventListener("bookcover", onCover);
    hostEl.addEventListener("booklink", onLink);
    return () => {
      hostEl.removeEventListener("bookpage", onPage);
      hostEl.removeEventListener("bookcover", onCover);
      hostEl.removeEventListener("booklink", onLink);
    };
  }, [step, goKey, onLinkHref]);

  useEffect(() => {
    const onPop = () => {
      const place = parsePlace();
      if (place.closed) closeBook(false);
      else void goKey(place.key, true, false);
    };
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, [closeBook, goKey]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target;
      if (target instanceof Element && target.closest("input, textarea, select, [role='radiogroup'], [role='radio']")) return;
      if (event.key === "ArrowRight") {
        if (open) step(1);
        else void goKey("start", false);
      }
      if (event.key === "ArrowLeft" && open) step(-1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, step, goKey]);

  const beats = stops[stop]?.beats ?? [];
  const beatCount = beats.length;

  const pickBeat = useCallback(
    (next: number) => {
      setBeat(next);
      scene.current?.goTo(stopRef.current, next);
    },
    [],
  );

  useEffect(() => {
    if (!playing || beatCount < 2) return;
    const id = window.setTimeout(() => {
      const next = (beat + 1) % beatCount;
      pickBeat(next);
      if (next === 0) setPlaying(false);
    }, beatMs);
    return () => window.clearTimeout(id);
  }, [playing, beat, beatCount, pickBeat]);

  const togglePlay = () => {
    if (playing) return setPlaying(false);
    setPlaying(true);
    pickBeat((beat + 1) % beatCount);
  };

  const inBook = (event: MouseEvent, key: string) => {
    event.preventDefault();
    void goKey(key, true);
  };

  const title = stops[stop]?.page?.title ?? "";
  const caption = open ? (beats[beat]?.caption ?? title) : "Every Byte: click the cover to open the book";
  const spread = segment === "errata" && meta ? errataSpreads[meta.local] : null;
  const chapterStep = def && meta ? def.steps[meta.local] : undefined;
  const bench = def && meta ? def.workbench[meta.local - def.steps.length] : undefined;
  const exercise = def && meta ? chapterExercises(def.slug)[meta.local - def.steps.length - def.workbench.length] : undefined;
  const otherHref = def ? `/kernels/${def.slug}` : "/kernels";

  if (failed) {
    return (
      <div className="mx-auto max-w-2xl space-y-3 px-4 py-20">
        <h1 className="font-serif text-4xl">This book needs WebGL</h1>
        <p className="text-muted">
          Your browser could not start 3D graphics. The{" "}
          <Link href="/kernels" className="link-underline text-text">
            list of kernels
          </Link>{" "}
          has every chapter as a scroll page.
        </p>
      </div>
    );
  }

  return (
    <div className="flex h-[calc(100dvh-3rem)] flex-col overflow-hidden">
      <div ref={bodies} hidden>
        <BookText>
          {Object.values(defs).map((c) => (
            <div key={c.slug} data-chapter={c.slug}>
              {c.steps.map((s) => (
                <div key={s.id}>{s.body}</div>
              ))}
            </div>
          ))}
        </BookText>
      </div>
      <div ref={host} className="relative min-h-0 flex-1 touch-none bg-[var(--book-desk)]">
        <canvas ref={canvas} role="img" aria-label={open ? `3D book "Every Byte", open at ${title}. ${caption}` : "3D book \"Every Byte\", closed on the desk"} className="absolute inset-0 size-full" />
        <Magnifier lens={lens} />
        <PointGloss at={term} />
        <motion.nav
          aria-label="book"
          initial={false}
          animate={{ width: barWidth }}
          transition={{ duration: reduced ? 0 : 0.7, ease: "easeInOut" }}
          className="absolute bottom-4 left-1/2 z-20 -translate-x-1/2 overflow-hidden rounded-full border border-white/30 bg-surface/55 tabular-nums text-xs text-muted shadow-lg shadow-black/15 backdrop-blur-xl backdrop-saturate-150"
        >
          <div ref={barRef} className="flex w-max max-w-[calc(100vw-2rem)] items-center gap-1.5 py-1 pl-1.5 pr-4">
            <button type="button" aria-label="previous chapter" disabled={!open || sectionKeys.indexOf(segment) <= 0} onClick={() => jump(-1)} className="pressable rounded-full p-1.5 enabled:hover:text-text disabled:opacity-30">
              <Icon d="M8 3 3 8l5 5M13 3 8 8l5 5" />
            </button>
            <button type="button" aria-label={stop === 0 ? "close the book" : "previous page"} disabled={!open} onClick={() => step(-1)} className="pressable rounded-full p-1.5 enabled:hover:text-text disabled:opacity-30">
              <Icon d="M10 3 5 8l5 5" />
            </button>
            <button type="button" aria-label={playing ? "pause the tour" : "play the tour"} disabled={!open || beatCount < 2} onClick={togglePlay} className="pressable rounded-full p-1.5 enabled:hover:text-text disabled:opacity-30">
              <Icon d={playing ? "M5 3v10M11 3v10" : "M5 3l8 5-8 5z"} />
            </button>
            {total > 0 && (
              <span className="shrink-0 text-text">
                <Scramble key={page} text={`${page} / ${total}`} numeric durationMs={700} />
              </span>
            )}
            <span className="min-w-0 max-w-[22rem] truncate">
              <Scramble key={caption} text={caption} durationMs={900} />
            </span>
            <span className="flex shrink-0 items-center gap-0.5">
              {open &&
                beats.length > 1 &&
                beats.map((b, i) => (
                  <button key={b.caption} type="button" aria-label={b.caption} aria-current={i === beat} onClick={() => pickBeat(i)} className="p-1.5">
                    <span className={clsx("block size-1.5 rounded-full transition-colors duration-300", i === beat ? "bg-accent" : "bg-border")} />
                  </button>
                ))}
            </span>
            <button type="button" aria-label={open ? "next page" : "open the book"} disabled={open && stop === stops.length - 1} onClick={() => (open ? step(1) : void goKey("start", false))} className="pressable rounded-full p-1.5 enabled:hover:text-text disabled:opacity-30">
              <Icon d="M6 3l5 5-5 5" />
            </button>
            <button type="button" aria-label="next chapter" disabled={!open || sectionKeys.indexOf(segment) >= sectionKeys.length - 1} onClick={() => jump(1)} className="pressable rounded-full p-1.5 enabled:hover:text-text disabled:opacity-30">
              <Icon d="M3 3l5 5-5 5M8 3l5 5-5 5" />
            </button>
            <Link href={otherHref} className="link-underline ml-1 shrink-0 border-l border-border pl-3 max-sm:hidden">
              {def ? "reading view" : "list view"}
            </Link>
          </div>
        </motion.nav>
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
        {Object.values(defs).map(
          (c) => c.Overlay && <c.Overlay key={c.slug} kind={c.slug === def?.slug ? (overlay?.kind ?? null) : null} target={c.slug === def?.slug && overlay ? pageEl : null} narrow={narrow} />,
        )}
      </div>
      <div className="relative border-border bg-surface max-[720px]:border-t">
        <GlossProvider key={stop} off={!narrow}>
        <article aria-live="polite" className="max-h-[32dvh] overflow-y-auto px-4 py-3 text-sm leading-relaxed text-muted min-[721px]:sr-only">
          {!open ? (
            <p>
              Every Byte, an inference engine kernel by kernel.{" "}
              <button type="button" onClick={() => void goKey("start", false)} className="link-underline text-text">
                Open the book
              </button>
            </p>
          ) : def && meta && meta.local < 0 ? (
            <>
              <h2 className="text-xs">{bookParts.filter((p) => p.slugs.includes(def.slug)).map((p) => `Part ${p.numeral} · ${p.title}`)}</h2>
              <h1 className="font-serif text-2xl text-text">{`Chapter ${def.number}: ${chapterTitle(def.slug)}`}</h1>
              <p className="mt-1">{stops[stop]?.page?.paragraphs[0]?.[0]?.text}</p>
              <h2 className="mt-2 text-xs">in this chapter</h2>
              <ol>
                {bookTitles(def).map((t, k) => (
                  <li key={`${k}-${t}`} className="flex gap-2">
                    <span className="w-5 tabular-nums">{stepNumber(def, k)}</span>
                    <button type="button" onClick={() => turnTo(def.slug, k)} className="link-underline text-left text-text">
                      {t}
                    </button>
                    <span className="ml-auto tabular-nums">{(chapterPage(def.slug) ?? 0) + k + 1}</span>
                  </li>
                ))}
              </ol>
            </>
          ) : chapterStep ? (
            <>
              <h1 className="font-serif text-2xl text-text">{chapterStep.title}</h1>
              <div className="mt-2 space-y-2">
                <BookText>{gloss(chapterStep.body)}</BookText>
              </div>
            </>
          ) : bench ? (
            <>
              <h1 className="font-serif text-2xl text-text">{bench.title}</h1>
              <div className="mt-2 space-y-2">
                {bench.paragraphs.map((p) => (
                  <p key={p}>{gloss(p)}</p>
                ))}
              </div>
            </>
          ) : exercise ? (
            <>
              <h2 className="text-xs">build it yourself</h2>
              <h1 className="font-serif text-2xl text-text">{exercise.title}</h1>
              <p className="mt-2">{gloss(exercise.task)}</p>
              <Link href={`/exercises#${exercise.id}`} className="link-underline mt-2 inline-block text-text">
                The steps, the spec and the commands to copy →
              </Link>
            </>
          ) : segment === "start" && meta ? (
            meta.local === 0 ? (
              <>
                <h1 className="font-serif text-2xl text-text">start here</h1>
                <div className="mt-2 space-y-2">
                  {startText.forWho.map((p) => (
                    <p key={p}>{gloss(p)}</p>
                  ))}
                </div>
                <h2 className="mt-3 font-serif text-xl text-text">what you need first</h2>
                <ul className="mt-1 list-disc space-y-1 pl-4">
                  {startText.needFirst.map((p) => (
                    <li key={p}>{gloss(p)}</li>
                  ))}
                </ul>
              </>
            ) : (
              <>
                <h1 className="font-serif text-2xl text-text">the life of a token</h1>
                <div className="mt-2 space-y-2">
                  {startText.tokenLife.map((p) => (
                    <p key={p}>{gloss(p)}</p>
                  ))}
                </div>
                <h2 className="mt-3 font-serif text-xl text-text">one decode step</h2>
                <ol className="mt-1">
                  {startText.map().map((r) => (
                    <li key={r.y} className={clsx("flex gap-2", r.inLayer && "pl-3")}>
                      {r.slugs.length ? (
                        <Link href={bookHref(r.slugs[0])} onClick={(e) => inBook(e, r.slugs[0])} className="link-underline text-text">
                          {r.label}
                        </Link>
                      ) : (
                        <span>{r.label}</span>
                      )}
                      {r.slugs.length > 0 && <span className="ml-auto tabular-nums">{`ch ${r.slugs.map((x) => bookOrder.indexOf(x) + 1).join(", ")}`}</span>}
                    </li>
                  ))}
                </ol>
                <p className="mt-1 text-xs">Every layer repeats the indented rows, 22 times.</p>
              </>
            )
          ) : spread ? (
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
                        <Link href={chapterSlugs.includes(d.slug) ? bookHref(d.slug) : `/kernels/${d.slug}`} onClick={chapterSlugs.includes(d.slug) ? (e) => inBook(e, d.slug) : undefined} className="link-underline text-accent">
                          {d.kernel}
                        </Link>
                      </div>
                      <h2 className="font-serif text-xl text-text">{d.title}</h2>
                      <p>{gloss(d.tried)}</p>
                      <p className="text-text">{gloss(d.happened)}</p>
                      <p>{gloss(d.kept)}</p>
                    </section>
                  ),
              )}
            </>
          ) : (
            <>
              <h1 className="font-serif text-2xl text-text">Contents</h1>
              {bookParts.map((part) => (
                <div key={part.numeral} className="mt-2">
                  <h2 className="text-xs">{`Part ${part.numeral} · ${part.title}`}</h2>
                  <ol>
                    {part.slugs.map((slug) => (
                      <li key={slug} className="flex gap-2">
                        <span className="w-5 tabular-nums">{bookOrder.indexOf(slug) + 1}</span>
                        <Link href={bookHref(slug)} onClick={chapterSlugs.includes(slug) ? (e) => inBook(e, slug) : undefined} className="link-underline text-text">
                          {chapterTitle(slug)}
                        </Link>
                        <span className="ml-auto tabular-nums">{chapterPage(slug)}</span>
                      </li>
                    ))}
                  </ol>
                </div>
              ))}
              <button type="button" onClick={() => void goKey("errata", true)} className="link-underline mt-2 flex w-full text-text">
                {`Errata: ${errataTitle}`}
                <span className="ml-auto tabular-nums text-muted">{chapterPage("errata")}</span>
              </button>
            </>
          )}
        </article>
        </GlossProvider>
      </div>
    </div>
  );
}
