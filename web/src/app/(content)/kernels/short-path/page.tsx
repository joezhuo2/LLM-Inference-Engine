import type { Metadata } from "next";
import Link from "next/link";
import { startSpreadCount, startWords } from "@/components/book/front-matter";
import { BOOK } from "@/components/book/paths";
import { PageTitle } from "@/components/page-title";
import { PageTransition } from "@/components/page-transition";
import { bookOrder, chapterNumber } from "@/content/book-order";
import { kernelStory } from "@/content/kernels";
import { countedChapter, defaultCap, leadWords, pathMinutes, reading, roundMinutes, shortPathName, shortPath, shortPathQuery, shortRoute, stepWords } from "@/content/short-path";

const description = "A route through Every Byte for a reader with an hour: start here, then the lead-in and the idea steps of the chapters that carry the book's argument about bytes per token.";

export const metadata: Metadata = { title: shortPathName, description };

const uncounted = (slug: string) => Array.from({ length: shortPath.find((c) => c.slug === slug)?.cap ?? defaultCap }, (_, i) => ({ id: `${i}`, title: "" }));
const stepsOf = (slug: string) => countedChapter(slug)?.steps ?? uncounted(slug);
const route = shortRoute(bookOrder.map((slug) => ({ slug, steps: stepsOf(slug) })), startWords, startSpreadCount);
const bookLink = `${BOOK}/start?${shortPathQuery}`;

export default function ShortPath() {
  return (
    <PageTransition>
      <article className="space-y-10">
        <header className="space-y-4">
          <div className="tabular-nums text-xs text-muted">
            <Link href="/kernels" transitionTypes={["nav-back"]} className="link-underline">
              kernels
            </Link>{" "}
            / {shortPathName}
          </div>
          <PageTitle text={shortPathName} />
          <p className="max-w-2xl text-muted">
            The book has one argument: a token costs the bytes read for it, so an engine gets faster by reading fewer bytes and wasting less time between kernels. This route follows only that argument. It takes the lead-in and the idea steps of {route.chapters.length} chapters and skips the build history, the sandboxes, the reviews and the exercises.
          </p>
          <p className="max-w-2xl tabular-nums text-text">
            {`${route.chapters.length} chapters, ${route.spreads} spreads in the book, ${route.words.toLocaleString("en-US")} words: about ${roundMinutes(route.minutes)} minutes.`}
          </p>
          <p className="max-w-2xl text-sm text-muted">
            {`The time is an estimate (calculated): ${reading.wordsPerMinute} words a minute, plus ${reading.secondsPerExhibit} seconds for each figure and ${reading.secondsPerTitle} seconds for each chapter opening. The words are counted from the built pages by web/scripts/short-path.mjs.`}
          </p>
          <Link href={bookLink} className="link-underline inline-block text-sm text-accent">
            read it in Every Byte, the 3D book, with the skipped pages turned for you
          </Link>
        </header>
        <ol className="max-w-2xl space-y-8">
          <li className="space-y-1">
            <div className="tabular-nums text-xs text-muted">{`book only · about ${Math.round(pathMinutes(startWords, startSpreadCount, 0))} min`}</div>
            <Link href={bookLink} className="link-underline font-serif text-2xl text-accent">
              start here
            </Link>
            <p className="text-sm text-muted">Who the book is for, the life of a token and serving in one spread. These three spreads exist only in the book.</p>
          </li>
          {route.chapters.map((c) => {
            const steps = stepsOf(c.slug);
            const counted = Boolean(countedChapter(c.slug));
            const words = c.picks.reduce((n, i) => n + stepWords(c.slug, steps[i].id), leadWords(c.slug));
            return (
              <li key={c.slug} className="space-y-1">
                <div className="tabular-nums text-xs text-muted">{`chapter ${chapterNumber(c.slug)} · about ${Math.round(pathMinutes(words, 1 + c.picks.length, 1))} min`}</div>
                <Link href={`/kernels/${c.slug}`} className="link-underline font-serif text-2xl text-accent">
                  {kernelStory(c.slug)?.title ?? c.slug}
                </Link>
                <ol className="space-y-0.5 text-muted">
                  <li>
                    <Link href={`/kernels/${c.slug}#about`} className="link-underline text-text">
                      what it is
                    </Link>
                  </li>
                  {!counted && (
                    <li>
                      <Link href={`/kernels/${c.slug}`} className="link-underline text-text">
                        {`the first ${c.picks.length} steps`}
                      </Link>
                    </li>
                  )}
                  {counted && c.picks.map((i) => (
                    <li key={steps[i].id}>
                      <Link href={`/kernels/${c.slug}#step-${steps[i].id}`} className="link-underline text-text">
                        {steps[i].title}
                      </Link>
                    </li>
                  ))}
                </ol>
              </li>
            );
          })}
        </ol>
      </article>
    </PageTransition>
  );
}
