"use client";

import { useEffect, useRef, useState } from "react";
import { useIntroDone } from "@/lib/intro";

const letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789#$%&*";
const digits = "0123456789";
const everyMs = 45;

function pool(ch: string, numeric: boolean) {
  if (numeric) return /[0-9]/.test(ch) ? digits : null;
  return /[a-z0-9]/i.test(ch) ? letters : null;
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
      el.textContent = text;
      return;
    }
    const chars = [...text];
    const resolveAt = chars.map((_, i) => delayMs + (durationMs * (i + 1)) / chars.length);
    const t0 = performance.now();
    let last = -Infinity;
    let raf = 0;
    const tick = (now: number) => {
      const t = now - t0;
      if (t - last >= everyMs) {
        last = t;
        el.textContent = chars
          .map((ch, i) => {
            const set = pool(ch, numeric);
            return !set || t >= resolveAt[i] ? ch : set[Math.floor(Math.random() * set.length)];
          })
          .join("");
      }
      if (t < delayMs + durationMs) raf = requestAnimationFrame(tick);
      else el.textContent = text;
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [start, text, numeric, durationMs, delayMs, replay]);

  return (
    <span className={className}>
      <span className="sr-only">{text}</span>
      <span ref={ref} aria-hidden="true">
        {text}
      </span>
    </span>
  );
}
