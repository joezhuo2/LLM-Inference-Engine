import type { KeyboardEvent } from "react";

const steps: Record<string, number> = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 };

export function radioKeyDown<T extends string>(choices: readonly T[], value: T, onChange: (v: T) => void) {
  return (e: KeyboardEvent<HTMLElement>) => {
    const current = choices.indexOf(value);
    const next = e.key === "Home" ? 0 : e.key === "End" ? choices.length - 1 : e.key in steps ? (current + steps[e.key] + choices.length) % choices.length : -1;
    if (next < 0) return;
    e.preventDefault();
    onChange(choices[next]);
    e.currentTarget.querySelectorAll<HTMLElement>('[role="radio"]')[next]?.focus();
  };
}
