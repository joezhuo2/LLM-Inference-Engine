"use client";

import { useEffect, useRef, useState } from "react";
import { useIntroDone } from "@/lib/intro";

const digits = "0123456789";
const everyMs = 45;

function pool(ch: string, numeric: boolean) {
  if (/[0-9]/.test(ch)) return digits;
  if (numeric) return null;
  if (/[ijlftr]/.test(ch)) return "ijlftr";
  if (/[mw]/.test(ch)) return "mw";
  if (/[a-z]/.test(ch)) return "abcdeghknopqsuvxyz";
  if (/[A-Z]/.test(ch)) return "ABCDEGHKNOPRSTUVXYZ";
  return null;
}

function splitWords(text: string) {
  return text.match(/\S+?-(?=\S)|\S+|\s+/g) ?? [];
}

function isWord(part: string) {
  return /\S/.test(part);
}

export function Scramble({
  text,
  numeric = false,
  durationMs = 900,
  delayMs = 0,
  trigger = "load",
  replay = 0,
  className,
}: {
  text: string;
  numeric?: boolean;
  durationMs?: number;
  delayMs?: number;
  trigger?: "load" | "view";
  replay?: number;
  className?: string;
}) {
  const introDone = useIntroDone();
  const ref = useRef<HTMLSpanElement>(null);
  const [seen, setSeen] = useState(false);
  const start = introDone && (trigger === "load" || seen);

  useEffect(() => {
    const el = ref.current;
    if (!el || trigger !== "view" || seen) return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) setSeen(true);
      },
      { rootMargin: "0px 0px -10% 0px" },
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [trigger, seen]);

  useEffect(() => {
    const el = ref.current;
    if (!el || !start) return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      el.querySelectorAll<HTMLElement>("[data-scramble]").forEach((span, w) => {
        span.textContent = splitWords(text).filter(isWord)[w];
      });
      return;
    }
    const spans = [...el.querySelectorAll<HTMLElement>("[data-scramble]")];
    const words = splitWords(text).filter(isWord);
    const total = [...words.join("")].length;
    let seen = 0;
    const parts = words.map((word) => {
      const chars = [...word];
      const resolveAt = chars.map((_, i) => delayMs + (durationMs * (seen + i + 1)) / total);
      seen += chars.length;
      return { word, chars, resolveAt };
    });
    const t0 = performance.now();
    let last = -Infinity;
    let raf = 0;
    const tick = (now: number) => {
      const t = now - t0;
      if (t - last >= everyMs) {
        last = t;
        parts.forEach(({ chars, resolveAt }, w) => {
          spans[w].textContent = chars
            .map((ch, i) => {
              const set = pool(ch, numeric);
              return !set || t >= resolveAt[i] ? ch : set[Math.floor(Math.random() * set.length)];
            })
            .join("");
        });
      }
      if (t < delayMs + durationMs) raf = requestAnimationFrame(tick);
      else parts.forEach(({ word }, w) => (spans[w].textContent = word));
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [start, text, numeric, durationMs, delayMs, replay]);

  return (
    <span ref={ref} className={className}>
      <span className="sr-only">{text}</span>
      <span aria-hidden="true">
        {splitWords(text).map((part, i) =>
          isWord(part) ? (
            <span key={i} className="relative inline-block whitespace-nowrap">
              <span className="invisible">{part}</span>
              <span data-scramble className="absolute top-0 left-0">
                {part}
              </span>
            </span>
          ) : (
            part
          ),
        )}
      </span>
    </span>
  );
}
