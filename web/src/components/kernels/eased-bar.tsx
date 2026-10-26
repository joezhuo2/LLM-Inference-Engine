import clsx from "clsx";

export function EasedBar({ fraction, className }: { fraction: number; className: string }) {
  const pct = Math.min(100, Math.max(0, fraction * 100));
  return (
    <div
      className={clsx("bar-ease h-full rounded-full", className)}
      style={{ clipPath: `inset(0 calc(100% - max(0.5rem, ${pct.toFixed(3)}%)) 0 0 round 999px)` }}
    />
  );
}
