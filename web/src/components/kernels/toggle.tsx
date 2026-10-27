"use client";

import clsx from "clsx";

export function Toggle({ label, choices, value, onChange }: { label: string; choices: string[]; value: string; onChange: (v: string) => void }) {
  return (
    <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
      <span className="w-14 shrink-0 text-muted">{label}</span>
      <div role="radiogroup" aria-label={label} className="flex overflow-hidden rounded border border-border">
        {choices.map((c) => (
          <button
            key={c}
            type="button"
            role="radio"
            aria-checked={value === c}
            onClick={() => onChange(c)}
            className={clsx("pressable px-2.5 py-1 tabular-nums", value === c ? "bg-accent-dim text-accent" : "text-muted")}
          >
            {c}
          </button>
        ))}
      </div>
    </div>
  );
}
