"use client";

import { AnimatePresence, motion } from "motion/react";
import { createContext, type CSSProperties, type ReactNode, useContext, useEffect, useId, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { type GlossEntry, glossary } from "@/lib/glossary";
import { usePrefersReducedMotion } from "@/lib/reduced-motion";

type Setter = (active: boolean) => void;
type Registry = { add: (key: string, el: HTMLElement, set: Setter) => () => void };

function createRegistry(): Registry {
  const groups = new Map<string, Map<HTMLElement, Setter>>();
  const dirty = new Set<string>();
  let queued = false;
  const flush = () => {
    queued = false;
    for (const key of dirty) {
      const group = groups.get(key);
      if (!group) continue;
      let first: HTMLElement | null = null;
      for (const el of group.keys()) {
        if (!el.isConnected || el.getClientRects().length === 0) continue;
        if (!first || first.compareDocumentPosition(el) & Node.DOCUMENT_POSITION_PRECEDING) first = el;
      }
      for (const [el, set] of group) set(el === first);
    }
    dirty.clear();
  };
  const mark = (key: string) => {
    dirty.add(key);
    if (queued) return;
    queued = true;
    queueMicrotask(flush);
  };
  return {
    add(key, el, set) {
      let group = groups.get(key);
      if (!group) groups.set(key, (group = new Map()));
      group.set(el, set);
      mark(key);
      return () => {
        group.delete(el);
        mark(key);
      };
    },
  };
}

const GlossContext = createContext<Registry | null>(null);

export function GlossProvider({ children, off = false }: { children: ReactNode; off?: boolean }) {
  const [registry] = useState(createRegistry);
  return <GlossContext.Provider value={off ? null : registry}>{children}</GlossContext.Provider>;
}

const ease = [0.23, 1, 0.32, 1] as const;
const OPEN_DELAY = 260;
const CLOSE_DELAY = 120;
const GUTTER = 16;
const CARD_W = 288;

export function GlossCard({ entry, id, style, below = true }: { entry: GlossEntry; id?: string; style?: CSSProperties; below?: boolean }) {
  const reduced = usePrefersReducedMotion();
  return (
    <motion.div
      id={id}
      role="tooltip"
      initial={reduced ? { opacity: 0 } : { opacity: 0, y: below ? -4 : 4, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={reduced ? { opacity: 0 } : { opacity: 0, y: below ? -2 : 2, scale: 0.99 }}
      transition={{ duration: reduced ? 0 : 0.32, ease }}
      style={{ width: CARD_W, maxWidth: `calc(100vw - ${GUTTER * 2}px)`, transformOrigin: below ? "top center" : "bottom center", ...style }}
      className="pointer-events-none z-50 rounded-md border border-border bg-surface/95 px-3 py-2.5 text-left text-[13px] leading-snug shadow-lg shadow-black/15 backdrop-blur-md"
    >
      <div className="text-text">
        <span className="font-medium">{entry.term}</span>
        <span className="text-muted">{` · ${entry.full}`}</span>
      </div>
      <p className="mt-1 text-xs leading-relaxed text-muted">{entry.why}</p>
    </motion.div>
  );
}

type Place = { left: number; top: number; below: boolean };

function placeNear(el: HTMLElement): Place {
  const all = el.getBoundingClientRect();
  const lines = el.getClientRects();
  const below = all.bottom + 180 < window.innerHeight || all.top < 200;
  const rect = (below ? lines[lines.length - 1] : lines[0]) ?? all;
  const vw = document.documentElement.clientWidth;
  const width = Math.min(CARD_W, vw - GUTTER * 2);
  const left = Math.min(Math.max(rect.left + rect.width / 2 - width / 2, GUTTER), vw - GUTTER - width);
  return { left, top: below ? rect.bottom + 8 : rect.top - 8, below };
}

export function Term({ k, children }: { k: string; children: ReactNode }) {
  const registry = useContext(GlossContext);
  const ref = useRef<HTMLSpanElement>(null);
  const [active, setActive] = useState(false);
  const [place, setPlace] = useState<Place | null>(null);
  const timer = useRef(0);
  const id = useId();
  const entry = glossary[k];

  useLayoutEffect(() => {
    const el = ref.current;
    if (!registry || !el || !entry) return;
    const remove = registry.add(k, el, setActive);
    return () => {
      remove();
      setActive(false);
    };
  }, [registry, k, entry]);

  const open = place !== null;
  useEffect(() => {
    if (!open) return;
    let raf = 0;
    const follow = () => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => ref.current && setPlace(placeNear(ref.current)));
    };
    const onKey = (event: KeyboardEvent) => event.key === "Escape" && setPlace(null);
    window.addEventListener("scroll", follow, { passive: true, capture: true });
    window.addEventListener("resize", follow);
    document.addEventListener("keydown", onKey);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("scroll", follow, { capture: true });
      window.removeEventListener("resize", follow);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  useEffect(() => () => window.clearTimeout(timer.current), []);

  if (!active || !entry) return <span ref={ref}>{children}</span>;

  const show = () => {
    window.clearTimeout(timer.current);
    if (ref.current) setPlace(placeNear(ref.current));
  };
  const hide = () => {
    window.clearTimeout(timer.current);
    setPlace(null);
  };
  const later = (fn: () => void, ms: number) => {
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(fn, ms);
  };

  return (
    <span
      ref={ref}
      tabIndex={0}
      aria-describedby={open ? id : undefined}
      data-term={k}
      className="gloss-term"
      onPointerEnter={(e) => e.pointerType !== "touch" && later(show, OPEN_DELAY)}
      onPointerLeave={(e) => e.pointerType !== "touch" && later(hide, CLOSE_DELAY)}
      onPointerUp={(e) => e.pointerType === "touch" && (open ? hide() : show())}
      onFocus={show}
      onBlur={hide}
    >
      {children}
      {createPortal(<AnimatePresence>{place && <GlossCard entry={entry} id={id} below={place.below} style={{ position: "fixed", left: place.left, top: place.top, translate: place.below ? undefined : "0 -100%" }} />}</AnimatePresence>, document.body)}
    </span>
  );
}

export type PointTerm = { key: string; x: number; y: number; tap: boolean };

export function PointGloss({ at }: { at: PointTerm | null }) {
  const [shown, setShown] = useState<PointTerm | null>(null);
  useEffect(() => {
    const id = window.setTimeout(() => setShown(at), at ? (at.tap ? 0 : OPEN_DELAY) : CLOSE_DELAY);
    return () => window.clearTimeout(id);
  }, [at]);
  const entry = shown ? glossary[shown.key] : undefined;
  const below = shown ? shown.y < window.innerHeight * 0.55 : true;
  return (
    <AnimatePresence>
      {shown && entry && (
        <GlossCard
          key={`${shown.key}-${shown.x}-${shown.y}`}
          entry={entry}
          below={below}
          style={{ position: "absolute", left: `clamp(${GUTTER}px, ${shown.x - CARD_W / 2}px, calc(100% - ${GUTTER + CARD_W}px))`, top: below ? shown.y + 22 : shown.y - 14, translate: below ? undefined : "0 -100%" }}
        />
      )}
    </AnimatePresence>
  );
}
