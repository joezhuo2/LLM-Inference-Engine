"use client";

import type { MouseEvent } from "react";
import { useTheme } from "next-themes";

export function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme();
  const toggle = (event: MouseEvent<HTMLButtonElement>) => {
    const next = resolvedTheme === "dark" ? "light" : "dark";
    const root = document.documentElement;
    if (!document.startViewTransition || matchMedia("(prefers-reduced-motion: reduce)").matches) return setTheme(next);
    const { left, top, width, height } = event.currentTarget.getBoundingClientRect();
    const x = left + width / 2;
    const y = top + height / 2;
    const radius = Math.hypot(Math.max(x, innerWidth - x), Math.max(y, innerHeight - y));
    root.classList.add("theme-switch");
    const transition = document.startViewTransition(() => {
      root.setAttribute("data-theme", next);
      root.style.colorScheme = next;
      setTheme(next);
    });
    void transition.ready.then(() => root.animate({ clipPath: [`circle(0px at ${x}px ${y}px)`, `circle(${radius}px at ${x}px ${y}px)`] }, { duration: 700, easing: "cubic-bezier(0.65, 0, 0.35, 1)", pseudoElement: "::view-transition-new(root)" }));
    void transition.finished.finally(() => root.classList.remove("theme-switch"));
  };
  return (
    <button type="button" onClick={toggle} className="pressable rounded p-1.5 text-muted hover:text-text" aria-label="Toggle light and dark theme">
      <svg aria-hidden viewBox="0 0 16 16" className="size-4 fill-none stroke-current" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
        <path className="theme-icon-light" d="M8 5a3 3 0 1 0 0 6 3 3 0 1 0 0-6zM8 1v1.5M8 13.5V15M1 8h1.5M13.5 8H15M3.05 3.05l1.06 1.06M11.89 11.89l1.06 1.06M3.05 12.95l1.06-1.06M11.89 4.11l1.06-1.06" />
        <path className="theme-icon-dark" d="M13.5 9.5A5.5 5.5 0 1 1 6.5 2.5a4.5 4.5 0 0 0 7 7z" />
      </svg>
    </button>
  );
}
