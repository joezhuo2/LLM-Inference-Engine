import Link from "next/link";
import { kernelCards } from "@/content/kernels";

const button = "pressable hover-grow flex min-w-0 flex-col items-center rounded border border-border bg-surface px-3 py-2 text-center";

export function StoryNav({ slug }: { slug: string }) {
  const i = kernelCards.findIndex((k) => k.slug === slug);
  const prev = i > 0 ? kernelCards[i - 1] : undefined;
  const next = i >= 0 && i < kernelCards.length - 1 ? kernelCards[i + 1] : undefined;
  return (
    <nav aria-label="kernel stories" className="mx-auto grid max-w-2xl grid-cols-3 gap-3 font-mono">
      {prev ? (
        <Link href={`/kernels/${prev.slug}`} transitionTypes={["nav-back"]} rel="prev" className={button}>
          <span className="text-xs text-muted">← previous</span>
          <span className="mt-1 text-sm text-accent">{prev.title}</span>
        </Link>
      ) : (
        <div />
      )}
      <Link href="/kernels" transitionTypes={["nav-back"]} className={button}>
        <span className="text-xs text-muted">↑ all</span>
        <span className="mt-1 text-sm text-accent">kernels</span>
      </Link>
      {next ? (
        <Link href={`/kernels/${next.slug}`} transitionTypes={["nav-forward"]} rel="next" className={button}>
          <span className="text-xs text-muted">next →</span>
          <span className="mt-1 text-sm text-accent">{next.title}</span>
        </Link>
      ) : (
        <div />
      )}
    </nav>
  );
}
