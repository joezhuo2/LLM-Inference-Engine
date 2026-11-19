"use client";

import Link from "next/link";
import { useDeferredValue, useRef, useState } from "react";
import type { GlossEntry } from "@/lib/glossary";

export type GlossaryRow = GlossEntry & { first?: { href: string; number: number; title: string } };

const letterOf = (term: string) => (/^[a-z]/i.test(term) ? term[0].toUpperCase() : "#");

export function GlossaryList({ rows }: { rows: GlossaryRow[] }) {
  const [query, setQuery] = useState("");
  const deferred = useDeferredValue(query);
  const input = useRef<HTMLInputElement>(null);
  const words = deferred.toLowerCase().split(/\s+/).filter(Boolean);
  const shown = rows.filter((r) => {
    const hay = [r.term, r.full, ...r.aliases].join(" ").toLowerCase();
    return words.every((w) => hay.includes(w));
  });
  const letters = [...new Set(shown.map((r) => letterOf(r.term)))];
  const clear = () => {
    setQuery("");
    input.current?.focus();
  };
  return (
    <div className="space-y-10">
      <div className="sticky top-16 z-10 max-w-xl">
        <div className="flex items-center gap-2 rounded border border-border bg-surface/95 px-3 py-2 shadow-sm backdrop-blur-md focus-within:border-accent">
          <svg aria-hidden viewBox="0 0 16 16" className="size-4 shrink-0 fill-none stroke-current text-muted" strokeWidth={1.5} strokeLinecap="round">
            <circle cx="7" cy="7" r="4.5" />
            <path d="m10.5 10.5 3 3" />
          </svg>
          <input
            ref={input}
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Escape" && query && (e.preventDefault(), setQuery(""))}
            placeholder="filter terms, full names and other spellings"
            aria-label="filter the glossary"
            aria-controls="glossary-terms"
            spellCheck={false}
            autoComplete="off"
            style={{ outline: "none" }}
            className="min-w-0 flex-1 bg-transparent text-text placeholder:text-muted focus:outline-none [&::-webkit-search-cancel-button]:hidden"
          />
          <span aria-live="polite" className="shrink-0 tabular-nums text-xs text-muted">
            {words.length ? `${shown.length} of ${rows.length}` : `${rows.length} terms`}
          </span>
          {query && (
            <button type="button" onClick={clear} aria-label="clear the filter" className="pressable -mr-1 rounded px-1.5 text-muted hover:text-text">
              ×
            </button>
          )}
        </div>
      </div>
      <div id="glossary-terms" className="space-y-10">
        {letters.map((letter) => (
          <section key={letter} aria-labelledby={`letter-${letter}`} className="grid gap-4 sm:grid-cols-[3rem_minmax(0,1fr)]">
            <h2 id={`letter-${letter}`} className="font-serif text-3xl text-accent">
              {letter}
            </h2>
            <dl className="max-w-2xl space-y-6">
              {shown
                .filter((r) => letterOf(r.term) === letter)
                .map((r) => (
                  <div key={r.key} id={r.key} className="scroll-mt-24 rounded transition-colors duration-700 target:bg-accent-dim target:outline target:outline-8 target:outline-accent-dim motion-reduce:transition-none">
                    <dt>
                      <span className="font-serif text-xl text-text">{r.term}</span>
                      <span className="text-muted">{` · ${r.full}`}</span>
                    </dt>
                    <dd className="mt-1 text-muted">{r.why}</dd>
                    {r.first && (
                      <dd className="mt-1 text-xs text-muted">
                        first used in{" "}
                        <Link href={r.first.href} transitionTypes={["nav-forward"]} className="link-underline text-accent">
                          {`chapter ${r.first.number}, ${r.first.title}`}
                        </Link>
                      </dd>
                    )}
                  </div>
                ))}
            </dl>
          </section>
        ))}
        {shown.length === 0 && (
          <p className="text-muted">
            {`No term matches "${deferred.trim()}". `}
            <button type="button" onClick={clear} className="link-underline text-accent">
              clear the filter
            </button>
          </p>
        )}
      </div>
    </div>
  );
}
