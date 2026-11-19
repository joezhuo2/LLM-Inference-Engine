"use client";

import type { ReactNode } from "react";
import { SmoothRange } from "../smooth-range";

export function Mirror({ from, children }: { from: string; children: ReactNode }) {
  return (
    <div role="group" aria-label={`controls from ${from}, mirrored here`} className="space-y-1.5 rounded border border-dashed border-border px-2.5 py-2">
      <div className="text-muted">{`the controls from ${from}, mirrored: change them here and watch below`}</div>
      {children}
    </div>
  );
}

export function MirrorRange({ id, label, min, max, step, value, onValue, text, disabled }: { id: string; label: string; min: number; max: number; step?: number; value: number; onValue: (v: number) => void; text: string; disabled?: boolean }) {
  return (
    <div className="grid grid-cols-[3.5rem_minmax(0,1fr)_4.5rem] items-center gap-x-2">
      <label htmlFor={id} className="text-muted">
        {label}
      </label>
      <SmoothRange id={id} min={min} max={max} step={step} value={value} onValue={onValue} disabled={disabled} aria-valuetext={text} className="w-full accent-[var(--accent)] disabled:opacity-40" />
      <span className="text-right text-text tabular-nums">{text}</span>
    </div>
  );
}
