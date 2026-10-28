"use client";

import { useMotionValue } from "motion/react";
import { type RefObject, useEffect, useRef, useState } from "react";

function midline(panel: HTMLElement | null, first: HTMLElement | null | undefined) {
  const p = panel?.getBoundingClientRect();
  const f = first?.getBoundingClientRect();
  const above = p && f && p.left < f.right && f.left < p.right;
  return ((above ? Math.max(p.bottom, 0) : 0) + window.innerHeight) / 2;
}

function centers(refs: (HTMLElement | null)[]) {
  return refs.map((el) => {
    const r = el?.getBoundingClientRect();
    return r ? (r.top + r.bottom) / 2 : NaN;
  });
}

function lerp(c: number[], t: number) {
  const i = Math.max(0, Math.min(Math.floor(t), c.length - 2));
  return c[i] + (c[i + 1] - c[i]) * (t - i);
}

export function useActiveStep(panel: RefObject<HTMLElement | null>) {
  const [active, setActive] = useState(0);
  const progress = useMotionValue(0);
  const refs = useRef<(HTMLElement | null)[]>([]);

  useEffect(() => {
    let frame = 0;
    const pick = () => {
      frame = 0;
      const mid = midline(panel.current, refs.current[0]);
      const c = centers(refs.current);
      let best = 0;
      let dist = Infinity;
      c.forEach((y, i) => {
        const d = Math.abs(y - mid);
        if (d < dist) {
          dist = d;
          best = i;
        }
      });
      setActive(best);
      const next = mid > c[best] ? best + 1 : best - 1;
      const span = c[next] - c[best];
      const frac = Number.isFinite(span) && span !== 0 ? (mid - c[best]) / span : 0;
      progress.set(Math.max(0, Math.min(best + Math.sign(next - best) * Math.max(0, Math.min(frac, 1)), c.length - 1)));
    };
    const schedule = () => {
      if (!frame) frame = requestAnimationFrame(pick);
    };
    pick();
    window.addEventListener("scroll", schedule, { passive: true });
    window.addEventListener("resize", schedule);
    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener("scroll", schedule);
      window.removeEventListener("resize", schedule);
    };
  }, [panel, progress]);

  const scrollTo = (t: number, behavior: ScrollBehavior) => {
    const c = centers(refs.current);
    if (c.length > 1) window.scrollBy({ top: lerp(c, t) - midline(panel.current, refs.current[0]), behavior });
  };

  return { active, progress, refs, scrollTo };
}
