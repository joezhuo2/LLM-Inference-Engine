import clsx from "clsx";

export function Skeleton({ className }: { className?: string }) {
  return <div aria-hidden="true" className={clsx("animate-pulse rounded bg-surface motion-reduce:animate-none", className)} />;
}
