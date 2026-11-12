import type { Metadata } from "next";
import Link from "next/link";
import { gloss } from "@/components/glossed";
import { StoryNav } from "@/components/kernels/story-nav";
import { PageTitle } from "@/components/page-title";
import { PageTransition } from "@/components/page-transition";
import { bookHref } from "@/components/book/paths";
import { beyondIntro, beyondTitle, beyondTopics } from "@/content/beyond";
import { estimates } from "@/content/estimates";

const description = "What this engine leaves out, in the book's own unit of bytes per token: quantization, a draft model for speculative decoding, prefix caching and multiple GPUs. None of them is implemented or measured here.";

export const metadata: Metadata = { title: beyondTitle, description };

export default function Beyond() {
  return (
    <PageTransition>
      <article className="space-y-14">
        <header className="space-y-4">
          <div className="tabular-nums text-xs text-muted">
            <Link href="/kernels" transitionTypes={["nav-back"]} className="link-underline">
              kernels
            </Link>{" "}
            / beyond
          </div>
          <PageTitle text={beyondTitle} />
          {beyondIntro.map((p) => (
            <p key={p} className="max-w-2xl text-muted">
              {gloss(p)}
            </p>
          ))}
          <ul className="max-w-2xl space-y-1 text-sm">
            {beyondTopics.map((t) => (
              <li key={t.id} className="flex flex-wrap gap-x-3">
                <a href={`#${t.id}`} className="link-underline text-accent">
                  {t.title}
                </a>
                <span className="text-muted">{t.unit}</span>
              </li>
            ))}
          </ul>
          <Link href={bookHref("beyond")} className="link-underline inline-block text-sm text-accent">
            read this in Every Byte, the 3D book
          </Link>
        </header>
        {beyondTopics.map((t) => {
          const n = estimates.findIndex((e) => e.id === t.estimate) + 1;
          return (
            <section key={t.id} aria-labelledby={t.id} className="space-y-4">
              <div>
                <div className="text-xs text-muted">{t.unit}</div>
                <h2 id={t.id} className="mt-1 scroll-mt-20 font-serif text-3xl text-text">
                  {t.title}
                </h2>
              </div>
              {t.paragraphs.map((p) => (
                <p key={p} className="max-w-2xl text-muted">
                  {gloss(p)}
                </p>
              ))}
              {n > 0 && (
                <Link href={`/exercises#${t.estimate}`} className="link-underline block max-w-2xl text-sm text-accent">
                  {`estimate ${n}: ${estimates[n - 1].title}`}
                </Link>
              )}
              <p className="max-w-2xl text-sm text-muted">
                The source to read:{" "}
                <a href={t.source.href} target="_blank" rel="noreferrer" className="link-underline text-text">
                  {t.source.label}
                </a>
              </p>
            </section>
          );
        })}
        <StoryNav slug="beyond" />
      </article>
    </PageTransition>
  );
}
