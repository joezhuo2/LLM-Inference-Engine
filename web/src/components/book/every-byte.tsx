"use client";

import clsx from "clsx";
import { AnimatePresence, motion } from "motion/react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { type MouseEvent, useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import type { Group, Object3D } from "three";
import { GlossProvider, type PointTerm, PointGloss } from "@/components/gloss";
import { gloss } from "@/components/glossed";
import { BookText } from "@/components/kernels/authorship";
import { EnginesList } from "@/components/kernels/engines-map";
import { Scramble } from "@/components/scramble";
import { beyondIntro, beyondTitle, beyondTopics } from "@/content/beyond";
import { kindLabel } from "@/content/dead-ends";
import { estimates, estimatesIntro, estimatesTitle } from "@/content/estimates";
import { chapterExercises, numpyCommand, numpyOf, numpyUrl, runsLabel } from "@/content/exercises";
import { chapterReview } from "@/content/reviews";
import { shortPathName, shortPathQuery } from "@/content/short-path";
import { usePrefersReducedMotion } from "@/lib/reduced-motion";
import type { BookModel, BookScene, OverlayRect, Stop } from "./book-scene";
import { ReviewPage } from "./review-page";
import { GlossaryArticle } from "./glossary-article";
import { FitPage, type Lens, Magnifier, PAGE_DESIGN_WIDTH } from "./book-view";
import { type ChapterDef, chapterStops, stepNumber } from "./chapter";
import { bookChapters } from "./chapters";
import { beyondSpreadOf, beyondSpreads, beyondStops, beyondUnitsNote, beyondUnitsTitle, bookOrder, bookRoute, routeIndex, routeLine, ideasRoute, isDeep, setIdeasView, deepGroup, routePage, routeSection, reviewOverlay, routeStep, setBookRoute, bookParts, bookTitles, chapterPage, chapterTitle, leftPage, contentsStop, errataIntro, estimateOf, errataSpreads, errataStops, errataTitle, estimateStops, exerciseStops, introStop, numberPages, reviewCount, reviewStops, startStops, startText, glossarySpreads, glossaryStops, glossaryTitle, engineStops } from "./front-matter";
import type { Kit } from "./ink";
import { type Fonts, loadFonts, readFonts, readRuns } from "./typeset";
import { BOOK, bookHref } from "./paths";

const chapterSlugs = bookOrder.filter((slug) => slug in bookChapters);
const backKeys = ["beyond", "engines", "estimates", "errata", "glossary"];
const sectionKeys = ["start", "contents", ...chapterSlugs, ...backKeys];
const beatMs = 6500;
let numbered: Promise<number> | undefined;
const numberBook = () =>
  (numbered ??= Promise.all(chapterSlugs.map((slug) => bookChapters[slug]())).then((all) => {
    setBookRoute(all);
    return numberPages(Object.fromEntries(all.map((c) => [c.slug, 1 + c.steps.length + c.workbench.length + reviewCount(c.slug) + chapterExercises(c.slug).length])));
  }));

type Meta = { key: string; local: number; slug?: string; exhibit: number | null };
type Place = { closed: true } | { closed: false; key: string; at: string };
type Saved = { key: string; at: string; page: number; name: string };

const ideasQuery = "view=ideas";
const ideasKey = "every-byte:view";
const ideasName = "ideas only";
const placeKey = "every-byte:place";
const urlOf = (key: string, short: boolean, ideas: boolean, at = "") => {
  const [path, hash] = bookHref(key).split("#");
  const query = [short ? shortPathQuery : ideas ? ideasQuery : "", at && `at=${encodeURIComponent(at)}`].filter(Boolean).join("&");
  return `${path}${query ? `?${query}` : ""}${hash ? `#${hash}` : ""}`;
};

function atOf(m: Meta | undefined, def: ChapterDef | undefined): string {
  if (!m) return "";
  if (!m.slug) return m.local > 0 ? String(m.local + 1) : "";
  if (!def || m.local < 0) return "";
  const k = m.local - def.steps.length;
  if (k < 0) return def.steps[m.local].id;
  if (k < def.workbench.length) return `try-${k + 1}`;
  const r = k - def.workbench.length - reviewCount(def.slug);
  return r < 0 ? "review" : `build-${r + 1}`;
}

function localOf(key: string, at: string, def: ChapterDef | undefined): number | undefined {
  if (!at) return undefined;
  if (!chapterSlugs.includes(key)) return Number(at) - 1;
  if (!def) return undefined;
  const step = def.steps.findIndex((x) => x.id === at);
  if (step >= 0) return step;
  const [kind, n] = at.split("-");
  const k = kind === "try" ? Number(n) - 1 : kind === "review" ? def.workbench.length : kind === "build" ? def.workbench.length + reviewCount(def.slug) + Number(n) - 1 : NaN;
  return def.steps.length + k;
}

function readSaved(): Saved | undefined {
  try {
    const v = JSON.parse(localStorage.getItem(placeKey) ?? "null") as Partial<Saved> | null;
    return v && typeof v.key === "string" && sectionKeys.includes(v.key) && typeof v.at === "string" && typeof v.page === "number" && typeof v.name === "string" ? (v as Saved) : undefined;
  } catch {
    return undefined;
  }
}
const sectionName = (key: string) => (key === "start" ? "start here" : chapterSlugs.includes(key) ? chapterTitle(key) : key === "beyond" ? beyondTitle : key === "estimates" ? estimatesTitle : key === "errata" ? errataTitle : "contents");
const inUrl = (query: string) => window.location.search.slice(1).split("&").includes(query);
const ideasSaved = () => {
  try {
    return localStorage.getItem(ideasKey) === "ideas";
  } catch {
    return false;
  }
};

function parsePlace(): Place {
  const path = window.location.pathname.replace(/\/$/, "");
  if (path === BOOK) return { closed: true };
  const slug = path.split("/")[2];
  const at = new URLSearchParams(window.location.search).get("at") ?? "";
  if (slug && chapterSlugs.includes(slug)) return { closed: false, key: slug, at };
  if (slug === "start") return { closed: false, key: "start", at };
  const back = window.location.hash.slice(1);
  return { closed: false, key: backKeys.includes(back) ? back : "contents", at };
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
  const keys = ["start", "contents", ...chapterSlugs.filter((s) => raw[s]), ...backKeys];
  const stops: Stop[] = [];
  const meta: Meta[] = [];
  for (const key of keys) {
    const slug = chapterSlugs.includes(key) ? key : undefined;
    raw[key].forEach((s, i) => {
      meta.push({ key, local: slug ? i - 1 : i, slug, exhibit: slug ? s.exhibit : null });
      stops.push({ ...s, spread: stops.length, exhibit: slug && s.exhibit !== null ? chapterSlugs.indexOf(slug) : null });
    });
  }
  return { stops, meta };
}

const base: Record<string, Stop[]> = { start: startStops(), contents: [contentsStop()], beyond: beyondStops(0), engines: engineStops(0), estimates: estimateStops(0), errata: errataStops(0), glossary: glossaryStops(0) };
const initial = compose(base);

export function EveryByte() {
  const reduced = usePrefersReducedMotion();
  const router = useRouter();
  const pathname = usePathname();
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
  const [short, setShort] = useState(false);
  const [lensOn, setLensOn] = useState(true);
  const shortRef = useRef(false);
  const [ideas, setIdeas] = useState(false);
  const [saved, setSaved] = useState<Saved>();
  const ideasRef = useRef(false);

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

  const setMode = useCallback((on: boolean) => {
    shortRef.current = on;
    setShort(on);
  }, []);

  const setIdeasMode = useCallback((on: boolean) => {
    ideasRef.current = on;
    setIdeas(on);
    setIdeasView(on);
    scene.current?.repaint();
  }, []);

  useEffect(() => {
    const sync = () => {
      setMode(inUrl(shortPathQuery));
      setIdeasMode(inUrl(ideasQuery) || ideasSaved());
      setSaved(readSaved());
    };
    sync();
    window.addEventListener("popstate", sync);
    return () => window.removeEventListener("popstate", sync);
  }, [setMode, setIdeasMode]);

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
      const chapterRaw = [introStop(c), ...chapterStops(c, text), ...reviewStops(c), ...exerciseStops(c)];
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
    const m = book.current.meta[index];
    if (!m) return;
    const url = urlOf(m.key, shortRef.current, ideasRef.current, atOf(m, defsRef.current[m.key]));
    const here = window.location;
    if (here.pathname + here.search + here.hash === url) return;
    const [path, hash] = bookHref(m.key).split("#");
    const moved = here.pathname.replace(/\/$/, "") !== path || here.hash !== (hash ? `#${hash}` : "");
    window.history[push && moved ? "pushState" : "replaceState"](null, "", url);
  }, []);

  const preload = useCallback(
    (index: number) => {
      const slug = book.current.meta[index]?.slug;
      const at = slug ? chapterSlugs.indexOf(slug) : backKeys.includes(book.current.meta[index]?.key ?? "") ? chapterSlugs.length : -1;
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
    async (key: string, riffle: boolean, push = true, last = false, at = "") => {
      await Promise.all([ensure(key), numberBook()]);
      const local = localOf(key, at, defsRef.current[key]);
      const found = local === undefined ? -1 : book.current.meta.findIndex((m) => m.key === key && m.local === local);
      const index = found >= 0 ? found : firstOf(key, last);
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
      setSaved(readSaved());
      stopRef.current = 0;
      setStop(0);
      void scene.current.close();
      if (push && window.location.pathname !== BOOK) window.history.pushState(null, "", BOOK);
    },
    [],
  );

  useEffect(() => {
    if (pathname.replace(/\/$/, "") === BOOK) closeBook(false);
  }, [pathname, closeBook]);

  const goStop = useCallback(
    async (key: string, local: number) => {
      await Promise.all([ensure(key), numberBook()]);
      const index = book.current.meta.findIndex((m) => m.key === key && m.local === local);
      const from = book.current.meta[stopRef.current];
      if (index >= 0) go(index, !from || Math.abs(routePage(key, local) - routePage(from.key, from.local)) > 1);
    },
    [ensure, go],
  );

  const resume = useCallback(
    async (place: Saved) => {
      await Promise.all([ensure(place.key), numberBook()]);
      const local = localOf(place.key, place.at, defsRef.current[place.key]);
      const found = book.current.meta.findIndex((m) => m.key === place.key && m.local === local);
      const index = found >= 0 ? found : firstOf(place.key);
      const s = scene.current;
      if (!s) return;
      if (!s.isOpen) {
        stopRef.current = 0;
        setOpen(true);
        void s.open(0);
      }
      go(index, true);
    },
    [ensure, go],
  );

  const enterShort = useCallback(() => {
    setMode(true);
    void goKey("start", true);
  }, [setMode, goKey]);

  const leaveShort = useCallback(() => {
    setMode(false);
    setUrl(stopRef.current, false);
  }, [setMode, setUrl]);

  const toggleIdeas = useCallback(
    (on: boolean) => {
      setIdeasMode(on);
      try {
        localStorage.setItem(ideasKey, on ? "ideas" : "all");
      } catch {}
      const m = book.current.meta[stopRef.current];
      if (scene.current?.isOpen) setUrl(stopRef.current, false);
      if (!on || shortRef.current || !m || !isDeep(m.key, m.local)) return;
      const to = routeStep(m.key, m.local, 1, ideasRoute());
      const index = to ? book.current.meta.findIndex((x) => x.key === to.key && x.local === to.local) : -1;
      if (index >= 0) go(index);
    },
    [setIdeasMode, go, setUrl],
  );

  const step = useCallback(
    (delta: 1 | -1) => {
      const i = stopRef.current;
      const m = book.current.meta[i];
      const order = shortRef.current ? bookRoute().order : ideasRef.current ? ideasRoute() : [];
      if (order.length) {
        const to = routeStep(m.key, m.local, delta, order);
        if (to) void goStop(to.key, to.local);
        else if (i === 0 && delta < 0) closeBook();
        return;
      }
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
    [go, goKey, goStop, closeBook],
  );

  const jump = useCallback(
    (delta: 1 | -1) => {
      const key = book.current.meta[stopRef.current]?.key ?? "contents";
      const next = routeSection(key, delta, shortRef.current);
      if (next) void goKey(next, true);
    },
    [goKey],
  );

  const barRef = useRef<HTMLDivElement>(null);
  const [tip, setTip] = useState<{ label: string; x: number; on: boolean }>();
  const tipProps = (label: string) => {
    const show = (e: { currentTarget: HTMLElement }) => {
      const box = e.currentTarget.getBoundingClientRect();
      const frame = e.currentTarget.closest("nav")?.parentElement?.getBoundingClientRect();
      setTip({ label, x: box.left + box.width / 2 - (frame?.left ?? 0), on: true });
    };
    const hide = () => setTip((t) => t && { ...t, on: false });
    return { onPointerEnter: show, onFocus: show, onPointerLeave: hide, onBlur: hide };
  };
  const [barWidth, setBarWidth] = useState<number>();
  useEffect(() => {
    const el = barRef.current;
    if (!el) return;
    const observer = new ResizeObserver(() => setBarWidth(el.offsetWidth + 2));
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const turnTo = useCallback(
    (slug: string, k: number) => {
      const to = ideasRef.current && !shortRef.current && isDeep(slug, k) ? routeStep(slug, k, 1, ideasRoute()) : undefined;
      if (to) void goStop(to.key, to.local);
      else go(firstOf(slug) + 1 + k, k > 0);
    },
    [go, goStop],
  );

  const focusLink = useCallback((target: Element) => {
    const el = target.closest<HTMLElement>("[data-book-link]");
    scene.current?.focusLink(el?.matches(":focus-visible") ? (el.dataset.bookLink ?? null) : null, el?.dataset.bookLabel);
  }, []);

  const onLinkHref = useCallback(
    (href: string) => {
      if (href === "#path") return enterShort();
      const [back, n] = href.startsWith("#") ? href.slice(1).split(":") : [];
      if (back && backKeys.includes(back)) return n ? go(firstOf(back) + (back === "beyond" ? beyondSpreadOf(Number(n)) : Number(n)), true) : void goKey(back, true);
      if (/^https?:\/\//.test(href)) return void window.open(href, "_blank", "noopener,noreferrer");
      const [path, at] = href.split("#");
      const slug = path.startsWith(`${BOOK}/`) ? path.slice(BOOK.length + 1) : "";
      if (at && chapterSlugs.includes(slug)) return turnTo(slug, Number(at));
      if (chapterSlugs.includes(slug)) return void goKey(slug, true);
      router.push(href);
    },
    [enterShort, go, goKey, router, turnTo],
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
          pageOf: (spread, side) => {
            const key = book.current.meta[spread]?.key;
            const first = key === undefined ? undefined : chapterPage(key);
            return first === undefined ? undefined : 2 * (first + spread - firstOf(key!)) - (side === "left" ? 2 : 1);
          },
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
      if (!place.closed) void goKey(place.key, false, false, false, place.at);
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
      else void goKey(place.key, true, false, false, place.at);
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

  useEffect(() => {
    if (!open && window.location.pathname.replace(/\/$/, "") !== BOOK) return;
    document.title = `${open ? `${sectionName(segment)} · ` : ""}Every Byte · llm-inference-engine`;
  }, [open, segment]);

  useEffect(() => {
    const m = layout.meta[stop];
    if (!open || !m || total === 0) return;
    try {
      localStorage.setItem(placeKey, JSON.stringify({ key: m.key, at: atOf(m, defs[m.key]), page: Math.max(1, leftPage(page) ?? 1), name: sectionName(m.key) }));
    } catch {}
  }, [open, stop, layout, defs, page, total]);

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
  const beyond = segment === "beyond" && meta ? beyondSpreads[meta.local] : null;
  const problem = segment === "estimates" && meta ? estimates[meta.local - 1] : undefined;
  const chapterStep = def && meta ? def.steps[meta.local] : undefined;
  const bench = def && meta ? def.workbench[meta.local - def.steps.length] : undefined;
  const reviewAt = def && meta ? meta.local - def.steps.length - def.workbench.length : -1;
  const review = def && reviewAt >= 0 && reviewAt < reviewCount(def.slug) ? chapterReview(def.slug) : undefined;
  const exercise = def && meta ? chapterExercises(def.slug)[reviewAt - reviewCount(def.slug)] : undefined;
  const exerciseNumpy = exercise ? numpyOf(exercise) : undefined;
  const otherHref = def ? `/kernels/${def.slug}${chapterStep && meta && meta.local >= (def.lead ?? 0) ? `#step-${chapterStep.id}` : ""}` : "/kernels";
  const route = bookRoute();
  const shortOn = short && open && total > 0 && route.order.length > 0;
  const ideasOn = ideas && !short && open && total > 0;
  const order = shortOn ? route.order : ideasOn ? ideasRoute() : [];
  const routeAt = meta ? routeIndex(meta.key, meta.local, order) : -1;
  const lastPage = order.length && meta ? !routeStep(meta.key, meta.local, 1, order) : stop === stops.length - 1;
  const ideasTip = short ? `the ${shortPathName} is on: leave it to read ${ideasName}` : ideas ? "show every page" : `${ideasName}: skip ${deepGroup}`;
  const hourButton = (
    <button type="button" data-book-link="#path" onClick={enterShort} className="link-underline mt-2 block text-left text-text">
      {`If you have an hour: read the ${shortPathName} →`}
      <span className="block text-xs text-muted">{routeLine()}</span>
    </button>
  );

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
        <Magnifier lens={lensOn ? lens : null} />
        <PointGloss at={term} />
        <motion.nav
          aria-label="book"
          initial={false}
          animate={{ width: barWidth }}
          transition={{ duration: reduced ? 0 : 0.7, ease: "easeInOut" }}
          className="absolute bottom-4 left-1/2 z-20 -translate-x-1/2 overflow-hidden rounded-full border border-[var(--pill-edge)] bg-surface/55 tabular-nums text-xs text-muted shadow-lg shadow-[color:var(--pill-shadow)] backdrop-blur-xl backdrop-saturate-150"
        >
          <div ref={barRef} className="flex w-max max-w-[calc(100vw-2rem)] items-center gap-1.5 py-1 pl-1.5 pr-1.5">
            <button type="button" aria-label="previous chapter" disabled={!open || !routeSection(segment, -1, shortOn)} onClick={() => jump(-1)} className="pressable rounded-full p-1.5 enabled:hover:text-text disabled:opacity-30">
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
                <Scramble key={page} text={`${open ? `${leftPage(page)}–${page * 2 - 1}` : 1} / ${total * 2 - 1}`} numeric durationMs={700} />
              </span>
            )}
            <span className="min-w-0 max-w-[22rem] truncate">
              {!open && saved ? (
                <button type="button" data-continue onClick={() => void resume(saved)} className="link-underline max-w-full truncate text-left text-text">
                  {`continue at page ${saved.page}, ${saved.name} →`}
                </button>
              ) : (
                <Scramble key={caption} text={caption} durationMs={900} />
              )}
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
            <button type="button" aria-label={open ? "next page" : "open the book"} disabled={open && lastPage} onClick={() => (open ? step(1) : void goKey("start", false))} className="pressable rounded-full p-1.5 enabled:hover:text-text disabled:opacity-30">
              <Icon d="M6 3l5 5-5 5" />
            </button>
            <button type="button" aria-label="next chapter" disabled={!open || !routeSection(segment, 1, shortOn)} onClick={() => jump(1)} className="pressable rounded-full p-1.5 enabled:hover:text-text disabled:opacity-30">
              <Icon d="M3 3l5 5-5 5M8 3l5 5-5 5" />
            </button>
            <span className="ml-1 flex shrink-0 items-center gap-1.5 border-l border-border pl-2">
              <button type="button" aria-label="table of contents" {...tipProps("table of contents")} onClick={() => void goKey("contents", true)} className="pressable rounded-full p-1.5 hover:text-text">
                <Icon d="M8 4.5C6.5 3.5 4.5 3 2 3v9.5c2.5 0 4.5.5 6 1.5 1.5-1 3.5-1.5 6-1.5V3c-2.5 0-4.5.5-6 1.5zM8 4.5V14" />
              </button>
              <button type="button" aria-label="magnifier" {...tipProps("magnifier")} aria-pressed={lensOn} onClick={() => setLensOn((on) => !on)} className={clsx("pressable rounded-full p-1.5 hover:text-text max-sm:hidden", lensOn && "text-accent")}>
                <Icon d="M7 2.5a4.5 4.5 0 1 0 0 9 4.5 4.5 0 1 0 0-9zM10.3 10.3 14 14" />
              </button>
              <button
                type="button"
                aria-label={ideasName}
                aria-pressed={ideas && !short}
                aria-disabled={short || undefined}
                {...tipProps(ideasTip)}
                onClick={(e) => {
                  if (short) return;
                  toggleIdeas(!ideas);
                  if (e.currentTarget.matches(":hover, :focus-visible")) setTip((t) => t && { ...t, label: ideas ? `${ideasName}: skip ${deepGroup}` : "show every page" });
                }}
                className={clsx("pressable rounded-full p-1.5 transition-[color,opacity] duration-500 ease-in-out motion-reduce:transition-none", short ? "cursor-default opacity-30" : "hover:text-text", ideas && !short && "text-accent")}
              >
                <svg aria-hidden viewBox="0 0 16 16" className="size-4 fill-none stroke-current" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
                  <path d="M6.5 14h3M6 12h4M8 3.5a3.5 3.5 0 0 0-2 6.4V12h4V9.9a3.5 3.5 0 0 0-2-6.4z" />
                  <path data-rays d="M8 .8v.9M3 5.8h.9M12.1 5.8h.9M4.3 2.3l.6.6M11.7 2.3l-.6.6" className={clsx("origin-center transition-[opacity,scale] duration-500 ease-in-out motion-reduce:transition-none", ideas && !short ? "scale-100 opacity-100" : "scale-50 opacity-0")} />
                </svg>
              </button>
              <Link href={otherHref} aria-label={def ? "reading view" : "list view"} {...tipProps(def ? "reading view" : "list view")} className="pressable rounded-full p-1.5 hover:text-text max-sm:hidden">
                <Icon d={def ? "M4 2.5h8v11H4zM6 5.5h4M6 8h4M6 10.5h2.5" : "M6 4h8M6 8h8M6 12h8M2.5 4h.01M2.5 8h.01M2.5 12h.01"} />
              </Link>
            </span>
          </div>
        </motion.nav>
        <span
          aria-hidden
          style={{ left: tip?.x ?? "50%" }}
          className={clsx("pointer-events-none absolute bottom-[3.75rem] z-20 -translate-x-1/2 whitespace-nowrap rounded-md border border-border bg-surface/95 px-2.5 py-1.5 text-xs text-muted shadow-lg shadow-[color:var(--pill-shadow)] backdrop-blur-md transition-[opacity,translate] duration-300 ease-out motion-reduce:transition-none", tip?.on ? "translate-y-0 opacity-100" : "translate-y-1 opacity-0")}
        >
          {tip?.label}
        </span>
        <AnimatePresence>
          {(shortOn || ideasOn) && (
            <motion.div
              key={shortOn ? "short" : "ideas"}
              data-short-path={shortOn || undefined}
              data-ideas-only={ideasOn || undefined}
              aria-label={shortOn ? shortPathName : ideasName}
              initial={{ opacity: 0, y: reduced ? 0 : -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: reduced ? 0 : -10 }}
              transition={{ duration: reduced ? 0 : 0.7, ease: "easeInOut" }}
              className="absolute left-1/2 top-3 z-20 flex w-max max-w-[calc(100vw-2rem)] -translate-x-1/2 items-center gap-2 rounded-full border border-[var(--pill-edge)] bg-surface/55 py-1 pl-4 pr-1.5 tabular-nums text-xs text-muted shadow-lg shadow-[color:var(--pill-shadow)] backdrop-blur-xl backdrop-saturate-150"
            >
              <span className="text-text">{shortOn ? shortPathName : ideasName}</span>
              <span>{routeAt >= 0 ? `${routeAt + 1} of ${order.length}` : shortOn ? "off the path, turn the page to rejoin" : "a deep dive, turn the page to skip it"}</span>
              <button type="button" onClick={shortOn ? leaveShort : () => toggleIdeas(false)} className="pressable rounded-full border-l border-border px-2.5 py-0.5 hover:text-text">
                {shortOn ? "leave" : "show every page"}
              </button>
            </motion.div>
          )}
        </AnimatePresence>
        {overlay && narrow && (
          <div data-theme="light" className={clsx("book-overlay absolute inset-0 touch-auto overflow-y-auto bg-[var(--book-paper)] p-5 text-[var(--text)] [scrollbar-width:none]", (shortOn || ideasOn) && "pt-14")}>
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
        {def && review && overlay?.kind === reviewOverlay && pageEl && createPortal(<ReviewPage key={def.slug} number={def.number} review={review} />, pageEl)}
      </div>
      <div className="relative border-border bg-surface max-[720px]:border-t">
        <GlossProvider key={stop} off={!narrow}>
        <article aria-live="polite" onFocus={(e) => focusLink(e.target)} onBlur={() => scene.current?.focusLink(null)} className="max-h-[32dvh] overflow-y-auto px-4 py-3 text-sm leading-relaxed text-muted min-[721px]:sr-only">
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
                    <button type="button" data-book-link={`${bookHref(def.slug)}#${k}`} onClick={() => turnTo(def.slug, k)} className="link-underline text-left text-text">
                      {t}
                    </button>
                    <span className="ml-auto tabular-nums">{leftPage((chapterPage(def.slug) ?? 0) + k + 1)}</span>
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
          ) : review ? (
            <>
              <h2 className="text-xs">review</h2>
              <h1 className="font-serif text-2xl text-text">what to remember</h1>
              <p className="mt-2 font-serif text-lg text-accent">{review.lesson}</p>
              <p className="mt-2 text-text">{gloss(review.why)}</p>
              <ol className="mt-2 list-decimal space-y-1 pl-4">
                {review.takeaways.map((t) => (
                  <li key={t}>{gloss(t)}</li>
                ))}
              </ol>
            </>
          ) : exercise ? (
            <>
              <h2 className="text-xs">build it yourself</h2>
              <h1 className="font-serif text-2xl text-text">{exercise.title}</h1>
              <p className="mt-1 text-xs">{runsLabel[exercise.runs]}</p>
              <p className="mt-2">{gloss(exercise.task)}</p>
              {exerciseNumpy && (
                <p className="mt-2">
                  Try it in NumPy first, on any machine:{" "}
                  <a href={numpyUrl(exerciseNumpy)} data-book-link={numpyUrl(exerciseNumpy)} target="_blank" rel="noreferrer" className="link-underline font-mono text-xs text-text">
                    {numpyCommand(exerciseNumpy)}
                  </a>
                </p>
              )}
              <Link href={`/exercises#${exercise.id}`} data-book-link={`/exercises#${exercise.id}`} className="link-underline mt-2 inline-block text-text">
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
                {route.order.length > 0 && hourButton}
              </>
            ) : meta.local === 1 ? (
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
                        <Link href={bookHref(r.slugs[0])} data-book-link={bookHref(r.slugs[0])} data-book-label={r.label} onClick={(e) => inBook(e, r.slugs[0])} className="link-underline text-text">
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
            ) : (
              <>
                <h1 className="font-serif text-2xl text-text">serving in one spread</h1>
                <div className="mt-2 space-y-2">
                  {startText.serving.map((p) => (
                    <p key={p}>{gloss(p)}</p>
                  ))}
                </div>
                <h2 className="mt-3 font-serif text-xl text-text">three requests</h2>
                <div className="mt-1 space-y-2">
                  {startText.servingFigure.map((p) => (
                    <p key={p}>{gloss(p)}</p>
                  ))}
                </div>
                <p className="mt-1 text-xs">
                  Chapter {bookOrder.indexOf("chunked-prefill") + 1}, chunked prefill, is about the stalls.{" "}
                  <Link href={bookHref("chunked-prefill")} onClick={(e) => inBook(e, "chunked-prefill")} className="link-underline text-text">
                    Open it
                  </Link>
                </p>
              </>
            )
          ) : segment === "estimates" && meta ? (
            problem ? (
              <>
                <h2 className="text-xs">{`${estimatesTitle} · problem ${meta.local} of ${estimates.length}`}</h2>
                <h1 className="font-serif text-2xl text-text">{problem.title}</h1>
                <p className="mt-2 text-text">{gloss(problem.question)}</p>
                <p className="mt-2">{gloss(problem.first)}</p>
                <ul className="mt-2 text-xs">
                  {problem.sources.map((s) => (
                    <li key={s}>{s}</li>
                  ))}
                </ul>
                <details className="mt-2">
                  <summary className="cursor-pointer text-text">the working and the answer</summary>
                  <ol className="mt-1 list-decimal space-y-1 pl-4">
                    {problem.steps.map((s) => (
                      <li key={s}>{gloss(s)}</li>
                    ))}
                  </ol>
                  <p className="mt-2 font-serif text-lg text-accent">{gloss(problem.answer)}</p>
                  <p className="mt-1 text-text">{gloss(problem.teaches)}</p>
                </details>
                {problem.beyond && (
                  <button type="button" data-book-link={`#beyond:${beyondTopics.findIndex((t) => t.id === problem.beyond)}`} onClick={() => onLinkHref(`#beyond:${beyondTopics.findIndex((t) => t.id === problem.beyond)}`)} className="link-underline mt-2 block text-left text-text">
                    {`Beyond this engine: ${problem.beyond} →`}
                  </button>
                )}
              </>
            ) : (
              <>
                <h1 className="font-serif text-2xl text-text">{estimatesTitle}</h1>
                <div className="mt-2 space-y-2">
                  {estimatesIntro.map((p) => (
                    <p key={p}>{p}</p>
                  ))}
                </div>
                <h2 className="mt-3 font-serif text-xl text-text">the problems</h2>
                <ol className="mt-1 list-decimal pl-4">
                  {estimates.map((e) => (
                    <li key={e.id}>{e.title}</li>
                  ))}
                </ol>
              </>
            )
          ) : beyond ? (
            <>
              {[beyond.left, beyond.right].map((p) =>
                p.kind === "intro" ? (
                  <section key="intro">
                    <h1 className="font-serif text-2xl text-text">{beyondTitle}</h1>
                    <div className="mt-2 space-y-2">
                      {beyondIntro.map((text) => (
                        <p key={text}>{gloss(text)}</p>
                      ))}
                    </div>
                  </section>
                ) : p.kind === "units" ? (
                  <section key="units" className="mt-3">
                    <h2 className="font-serif text-xl text-text">{beyondUnitsTitle}</h2>
                    <ul className="mt-1">
                      {beyondTopics.map((t) => (
                        <li key={t.id}>
                          <span className="text-text">{t.title}</span>
                          {`: ${t.unit}`}
                        </li>
                      ))}
                    </ul>
                    <p className="mt-1 text-xs">{beyondUnitsNote}</p>
                  </section>
                ) : (
                  <section key={p.topic.id} className="mt-3 space-y-2 first:mt-0">
                    <h2 className="text-xs">{`${beyondTitle} · ${p.topic.unit}`}</h2>
                    <h1 className="font-serif text-2xl text-text">{p.topic.title}</h1>
                    {p.topic.paragraphs.map((text) => (
                      <p key={text}>{gloss(text)}</p>
                    ))}
                    {estimateOf(p.topic) && (
                      <button type="button" data-book-link={estimateOf(p.topic)?.href} onClick={() => onLinkHref(estimateOf(p.topic)?.href ?? "")} className="link-underline block text-left text-text">
                        {estimateOf(p.topic)?.text}
                      </button>
                    )}
                    <a href={p.topic.source.href} data-book-link={p.topic.source.href} target="_blank" rel="noreferrer" className="link-underline block text-xs text-text">
                      {p.topic.source.label}
                    </a>
                  </section>
                ),
              )}
            </>
          ) : segment === "glossary" && meta ? (
            <GlossaryArticle spread={glossarySpreads[meta.local]} />
          ) : segment === "engines" ? (
            <EnginesList />
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
                        <Link href={chapterSlugs.includes(d.slug) ? bookHref(d.slug) : `/kernels/${d.slug}`} data-book-link={bookOrder.includes(d.slug) ? bookHref(d.slug) : `/kernels/${d.slug}`} onClick={chapterSlugs.includes(d.slug) ? (e) => inBook(e, d.slug) : undefined} className="link-underline text-accent">
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
                        <Link href={bookHref(slug)} data-book-link={bookHref(slug)} onClick={chapterSlugs.includes(slug) ? (e) => inBook(e, slug) : undefined} className="link-underline text-text">
                          {chapterTitle(slug)}
                        </Link>
                        {route.chapters.includes(slug) && <span className="text-xs text-accent">{shortPathName}</span>}
                        <span className="ml-auto tabular-nums">{leftPage(chapterPage(slug))}</span>
                      </li>
                    ))}
                  </ol>
                </div>
              ))}
              {route.order.length > 0 && hourButton}
              <button type="button" data-book-link="#beyond" onClick={() => void goKey("beyond", true)} className="link-underline mt-2 flex w-full text-text">
                Beyond this engine
                <span className="ml-auto tabular-nums text-muted">{leftPage(chapterPage("beyond"))}</span>
              </button>
              <button type="button" data-book-link="#engines" onClick={() => void goKey("engines", true)} className="link-underline mt-2 flex w-full text-text">
                Where this lives in other engines
                <span className="ml-auto tabular-nums text-muted">{leftPage(chapterPage("engines"))}</span>
              </button>
              <button type="button" data-book-link="#estimates" onClick={() => void goKey("estimates", true)} className="link-underline mt-2 flex w-full text-text">
                Estimate it yourself
                <span className="ml-auto tabular-nums text-muted">{leftPage(chapterPage("estimates"))}</span>
              </button>
              <button type="button" data-book-link="#errata" onClick={() => void goKey("errata", true)} className="link-underline mt-2 flex w-full text-text">
                {`Errata: ${errataTitle}`}
                <span className="ml-auto tabular-nums text-muted">{leftPage(chapterPage("errata"))}</span>
              </button>
              <button type="button" data-book-link="#glossary" onClick={() => void goKey("glossary", true)} className="link-underline mt-2 flex w-full text-text">
                {glossaryTitle}
                <span className="ml-auto tabular-nums text-muted">{leftPage(chapterPage("glossary"))}</span>
              </button>
            </>
          )}
        </article>
        </GlossProvider>
      </div>
    </div>
  );
}
