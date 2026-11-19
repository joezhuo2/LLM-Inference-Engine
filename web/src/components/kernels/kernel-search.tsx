"use client";

import clsx from "clsx";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { type KeyboardEvent as KeyEvent, type ReactNode, useEffect, useId, useRef, useState } from "react";
import { bookOrder } from "@/content/book-order";

type Doc = { href: string; where: string; title: string; text: string; head: string };
type Hit = Doc & { rank: number };

const LIMIT = 12;

async function loadDocs(): Promise<Doc[]> {
  const [{ default: index }, { glossaryEntries }] = await Promise.all([import("@/content/search-index.json"), import("@/lib/glossary")]);
  const pages = index as Record<string, { title: string; entries: { id: string; title: string; text: string }[] }>;
  const terms = glossaryEntries.map((e): Doc => ({ href: `/kernels/glossary#${e.key}`, where: "glossary", title: e.term, text: `${e.full}. ${e.why}`, head: [e.term, ...e.aliases].join(" ").toLowerCase() }));
  const steps = bookOrder.flatMap((slug) => (pages[slug]?.entries ?? []).map((e): Doc => ({ href: `/kernels/${slug}#${e.id}`, where: `${bookOrder.indexOf(slug) + 1} · ${pages[slug].title}`, title: e.title, text: e.text, head: e.title.toLowerCase() })));
  return [...terms, ...steps];
}

function searchDocs(docs: Doc[], query: string): Hit[] {
  const words = query.toLowerCase().split(/\s+/).filter(Boolean);
  if (!words.length) return [];
  const hits: Hit[] = [];
  for (const d of docs) {
    const all = `${d.head} ${d.text.toLowerCase()}`;
    if (words.every((w) => all.includes(w))) hits.push({ ...d, rank: words.every((w) => d.head.includes(w)) ? 0 : 1 });
  }
  return hits.sort((a, b) => a.rank - b.rank);
}

const escape = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

const marked = (text: string, words: string[]) => text.split(new RegExp(`(${words.map(escape).join("|")})`, "gi")).map((p, i) => (i % 2 ? <mark key={i} className="rounded-sm bg-accent-dim text-text">{p}</mark> : p));

function excerpt(text: string, words: string[]): ReactNode {
  const lower = text.toLowerCase();
  const found = words.map((w) => lower.indexOf(w)).filter((i) => i >= 0);
  const at = found.length ? Math.min(...found) : 0;
  let start = Math.max(0, at - 60);
  let end = Math.min(text.length, at + 110);
  if (start > 0) start = text.indexOf(" ", start) + 1;
  if (end < text.length) end = Math.max(text.lastIndexOf(" ", end), at + 1);
  return (
    <>
      {start > 0 && "… "}
      {marked(text.slice(start, end), words)}
      {end < text.length && " …"}
    </>
  );
}

export function KernelSearch() {
  const router = useRouter();
  const input = useRef<HTMLInputElement>(null);
  const docs = useRef<Promise<Doc[]> | null>(null);
  const [loaded, setLoaded] = useState<Doc[] | null>(null);
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const id = useId();
  const words = query.toLowerCase().split(/\s+/).filter(Boolean);
  const hits = loaded ? searchDocs(loaded, query) : [];
  const shown = hits.slice(0, LIMIT);

  const load = () => void (docs.current ??= loadDocs()).then(setLoaded);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "/" || e.metaKey || e.ctrlKey || e.altKey) return;
      const el = document.activeElement as HTMLElement | null;
      if (el && (el.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(el.tagName))) return;
      e.preventDefault();
      input.current?.focus();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, []);

  const onKeyDown = (e: KeyEvent<HTMLInputElement>) => {
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      if (shown.length) setActive((a) => (a + (e.key === "ArrowDown" ? 1 : shown.length - 1)) % shown.length);
    } else if (e.key === "Enter" && shown[active]) {
      e.preventDefault();
      router.push(shown[active].href);
    } else if (e.key === "Escape") {
      e.preventDefault();
      if (query) setQuery("");
      else input.current?.blur();
    }
  };

  const open = words.length > 0 && loaded !== null;

  useEffect(() => {
    if (open) document.getElementById(`${id}-${active}`)?.scrollIntoView({ block: "nearest" });
  }, [open, active, id]);
  return (
    <div className="relative max-w-xl">
      <div className="flex items-center gap-2 rounded border border-border bg-surface px-3 py-2 focus-within:border-accent">
        <svg aria-hidden viewBox="0 0 16 16" className="size-4 shrink-0 fill-none stroke-current text-muted" strokeWidth={1.5} strokeLinecap="round">
          <circle cx="7" cy="7" r="4.5" />
          <path d="m10.5 10.5 3 3" />
        </svg>
        <input
          ref={input}
          type="search"
          role="combobox"
          aria-label="search every chapter"
          aria-expanded={open}
          aria-controls={`${id}-list`}
          aria-autocomplete="list"
          aria-activedescendant={open && shown[active] ? `${id}-${active}` : undefined}
          value={query}
          onFocus={load}
          onChange={(e) => {
            setQuery(e.target.value);
            setActive(0);
          }}
          onKeyDown={onKeyDown}
          placeholder="search every chapter and the glossary"
          spellCheck={false}
          autoComplete="off"
          style={{ outline: "none" }}
          className="min-w-0 flex-1 bg-transparent text-text placeholder:text-muted focus:outline-none [&::-webkit-search-cancel-button]:hidden"
        />
        {query ? (
          <button type="button" onClick={() => (setQuery(""), input.current?.focus())} aria-label="clear the search" className="pressable -mr-1 rounded px-1.5 text-muted hover:text-text">
            ×
          </button>
        ) : (
          <kbd className="rounded border border-border px-1.5 text-xs text-muted">/</kbd>
        )}
      </div>
      {open && (
        <div className="absolute inset-x-0 top-full z-30 mt-2 rounded-md border border-border bg-surface/95 shadow-lg shadow-black/15 backdrop-blur-md transition-[opacity,translate] duration-300 ease-[var(--ease-out)] starting:-translate-y-1 starting:opacity-0 motion-reduce:transition-none">
          <p aria-live="polite" className="border-b border-border px-3 py-1.5 tabular-nums text-xs text-muted">
            {hits.length === 0 ? `nothing found for "${query.trim()}"` : hits.length > LIMIT ? `${hits.length} matches, the first ${LIMIT} shown` : `${hits.length} ${hits.length === 1 ? "match" : "matches"}`}
          </p>
          {shown.length > 0 && (
            <ul id={`${id}-list`} role="listbox" aria-label="search results" className="max-h-[60svh] overflow-y-auto py-1">
              {shown.map((h, i) => (
                <li key={h.href} id={`${id}-${i}`} role="option" aria-selected={i === active}>
                  <Link href={h.href} onPointerEnter={() => setActive(i)} className={clsx("block px-3 py-2", i === active && "bg-accent-dim")}>
                    <span className="block text-xs text-muted">{h.where}</span>
                    <span className="block text-text">{marked(h.title, words)}</span>
                    <span className="block text-sm text-muted">{excerpt(h.text, words)}</span>
                  </Link>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}
