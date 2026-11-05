import clsx from "clsx";
import { Scramble } from "@/components/scramble";

export function PageTitle({ text, className }: { text: string; className?: string }) {
  return (
    <h1 className={clsx("font-serif text-4xl leading-tight sm:text-5xl", className)}>
      <Scramble text={text} durationMs={1600} />
    </h1>
  );
}
