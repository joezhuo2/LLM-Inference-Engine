const W = 560;
const H = 400;

export function Frame({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-full w-full font-mono [font-variant-ligatures:none]" role="img" aria-label={label}>
      {children}
    </svg>
  );
}

export function Label({ x, y, children, anchor, className = "fill-muted" }: { x: number; y: number; children: React.ReactNode; anchor?: "end" | "middle"; className?: string }) {
  return (
    <text x={x} y={y} textAnchor={anchor} className={`${className} text-[11px]`}>
      {children}
    </text>
  );
}
