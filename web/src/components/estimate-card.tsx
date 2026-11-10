import Link from "next/link";
import { chapterNumber } from "@/content/book-order";
import type { Estimate } from "@/content/estimates";
import { kernelStory } from "@/content/kernels";
import { Disclosure } from "./disclosure";
import { gloss } from "./glossed";

export function EstimateCard({ estimate: e, n }: { estimate: Estimate; n: number }) {
  return (
    <article id={e.id} aria-labelledby={`${e.id}-title`} className="scroll-mt-20 space-y-4 rounded border border-border p-4 sm:p-5">
      <header className="space-y-2">
        <h3 id={`${e.id}-title`} className="font-serif text-2xl text-text">
          <span className="mr-3 tabular-nums text-accent">{n}</span>
          {e.title}
        </h3>
        <p className="max-w-2xl text-text">{gloss(e.question)}</p>
        <p className="max-w-2xl text-muted">{gloss(e.first)}</p>
        <ul className="max-w-2xl space-y-0.5 text-xs text-muted">
          {e.sources.map((s) => (
            <li key={s}>{s}</li>
          ))}
        </ul>
      </header>
      <Disclosure summary="show the working and the answer">
        <div className="max-w-2xl space-y-3">
          <ol className="space-y-2 text-muted">
            {e.steps.map((s, i) => (
              <li key={i} className="grid grid-cols-[1.5rem_minmax(0,1fr)] gap-x-2">
                <span className="tabular-nums text-accent">{i + 1}</span>
                <span>{gloss(s)}</span>
              </li>
            ))}
          </ol>
          <div className="space-y-1 border-l-2 border-accent pl-4">
            <div className="text-xs text-muted">the answer</div>
            <p className="font-serif text-xl text-accent">{gloss(e.answer)}</p>
          </div>
          <p className="text-text">{gloss(e.teaches)}</p>
          {e.chapters.map((slug) => (
            <Link key={slug} href={`/kernels/${slug}`} className="link-underline inline-block text-xs text-accent">
              {`read chapter ${chapterNumber(slug)}, ${kernelStory(slug)?.title ?? slug}`}
            </Link>
          ))}
        </div>
      </Disclosure>
    </article>
  );
}
