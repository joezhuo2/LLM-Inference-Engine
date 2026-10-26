"use client";

import { useTheme } from "next-themes";

export function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme();
  return (
    <button
      type="button"
      onClick={() => setTheme(resolvedTheme === "dark" ? "light" : "dark")}
      className="pressable rounded border border-border px-2 py-1 font-mono text-xs text-muted"
      aria-label="Toggle light and dark theme"
    >
      <span className="theme-icon-light">light</span>
      <span className="theme-icon-dark">dark</span>
    </button>
  );
}
