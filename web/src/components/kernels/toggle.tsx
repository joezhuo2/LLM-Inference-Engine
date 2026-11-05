"use client";

import clsx from "clsx";
import { useLayoutEffect, useRef, useState } from "react";
import { radioKeyDown } from "@/lib/radio-keys";

export function Toggle({ label, choices, value, onChange }: { label: string; choices: string[]; value: string; onChange: (v: string) => void }) {
  const group = useRef<HTMLDivElement>(null);
  const [clip, setClip] = useState<string | null>(null);
  const index = choices.indexOf(value);
  const labels = choices.join("\n");

  useLayoutEffect(() => {
    const el = group.current;
    if (!el) return;
    const measure = () => {
      const button = el.querySelectorAll<HTMLElement>('[role="radio"]')[index];
      if (!button) return setClip(null);
      const g = el.getBoundingClientRect();
      const b = button.getBoundingClientRect();
      const scale = g.width / el.offsetWidth || 1;
      const left = (b.left - g.left) / scale - el.clientLeft;
      setClip(`inset(0 ${el.clientWidth - left - b.width / scale}px 0 ${left}px)`);
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect();
  }, [index, labels]);

  return (
    <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
      <span className="w-14 shrink-0 text-muted">{label}</span>
      <div ref={group} role="radiogroup" aria-label={label} onKeyDown={radioKeyDown(choices, value, onChange)} className="relative flex overflow-hidden rounded border border-border">
        {choices.map((c) => (
          <button
            key={c}
            type="button"
            role="radio"
            aria-checked={value === c}
            tabIndex={value === c ? 0 : -1}
            onClick={() => onChange(c)}
            className={clsx("pressable px-2.5 py-1 tabular-nums", value === c && clip === null ? "bg-accent-dim text-accent" : "text-muted")}
          >
            {c}
          </button>
        ))}
        {clip !== null && (
          <div aria-hidden className="toggle-thumb pointer-events-none absolute inset-0 flex bg-accent-dim text-accent" style={{ clipPath: clip }}>
            {choices.map((c) => (
              <span key={c} className="px-2.5 py-1 tabular-nums">
                {c}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
