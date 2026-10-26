import clsx from "clsx";
import { Scramble } from "@/components/scramble";

export function PageTitle({ text, className }: { text: string; className?: string }) {
  return (
    <h1 className={clsx("font-mono text-2xl sm:text-3xl", className)}>
      <Scramble text={text} durationMs={1600} />
    </h1>
  );
}
