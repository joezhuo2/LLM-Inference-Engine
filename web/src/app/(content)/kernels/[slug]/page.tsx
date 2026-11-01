import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import type { ReactNode } from "react";
import { KernelTimeline } from "@/components/kernels/kernel-timeline";
import { StoryNav } from "@/components/kernels/story-nav";
import { KernelLeadView, KernelStoryView } from "@/components/kernels/story-view";
import { PageTitle } from "@/components/page-title";
import { PageTransition } from "@/components/page-transition";
import { Scramble } from "@/components/scramble";
import { repoUrl } from "@/content/build-log";
import { kernelStories, kernelStory } from "@/content/kernels";
import { docPath, docSections, environment, profilingTable } from "@/lib/data";
import { loadExcerpt } from "@/lib/excerpt";
import { ogImage } from "@/lib/og";

export const dynamicParams = false;

export function generateStaticParams() {
  return kernelStories.map((k) => ({ slug: k.slug }));
}

export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }): Promise<Metadata> {
  const story = kernelStory((await params).slug);
  if (!story) return {};
  return {
    title: story.title,
    description: story.short,
    openGraph: { siteName: "llm-inference-engine", type: "article", title: `${story.title} · llm-inference-engine`, description: story.short, images: [ogImage(story.slug, `${story.title}: ${story.headline.value} ${story.headline.label}`)] },
  };
}

function Section({ id, title, children }: { id: string; title: string; children: ReactNode }) {
  return (
    <section aria-labelledby={id} className="space-y-6">
      <h2 id={id} className="font-mono text-xl text-text">
        {title}
      </h2>
      {children}
    </section>
  );
}

const numeric = /^[\d,.\s/x%n/a-]+$|^\d/;

export default async function KernelPage({ params }: { params: Promise<{ slug: string }> }) {
  const story = kernelStory((await params).slug);
  if (!story) notFound();
  const excerpts = await Promise.all(story.excerpts.map(loadExcerpt));
  return (
    <PageTransition>
      <article className="space-y-20">
        <header className="space-y-4">
          <div className="font-mono text-xs text-muted">
            <Link href="/kernels" transitionTypes={["nav-back"]} className="link-underline">
              kernels
            </Link>{" "}
            / {String(story.index).padStart(2, "0")}
          </div>
          <PageTitle text={story.title} />
          <p className="max-w-2xl text-muted">{story.short}</p>
          <div className="flex flex-wrap items-end gap-x-10 gap-y-4 font-mono">
            <div>
              <div className="text-xs text-muted">computes</div>
              <div className="mt-1 text-text">{story.formula}</div>
            </div>
            <div>
              <div className="text-xs text-muted">{story.headline.label}</div>
              <div className="mt-1 text-2xl text-accent tabular-nums">
                <Scramble text={story.headline.value} numeric delayMs={300} durationMs={1400} />
              </div>
            </div>
          </div>
        </header>

        {story.lead && (
          <Section id="lead" title={story.lead.title}>
            <p className="max-w-2xl text-muted">{story.lead.intro}</p>
            <KernelLeadView slug={story.slug} />
          </Section>
        )}

        <Section id="how" title="how it works">
          <KernelStoryView slug={story.slug} />
        </Section>

        <Section id="timeline" title="timeline">
          <p className="max-w-2xl text-muted">
            Every change to the kernel and its wiring, from the spec to today. Numbers are as measured at the time.
          </p>
          <KernelTimeline entries={story.timeline} />
        </Section>

        <Section id="measurements" title="measurements">
          <p className="max-w-2xl text-muted">
            Copied at build time from {[...new Set(story.measurements.map((m) => docPath[m.doc ?? "profiling"]))].join(" and ")}, measured on the {String(environment.gpu ?? "RTX 5060 Laptop GPU")}.
          </p>
          {story.measurements.map((m) => {
            const t = profilingTable(m.section, m.table, m.doc);
            const title = docSections(m.doc ?? "profiling").find((s) => s.id === m.section)?.title;
            return (
              <figure key={`${m.section}-${m.table}`} className="space-y-2">
                <figcaption className="text-sm text-muted">
                  <span className="font-mono text-xs text-text">{title}</span> · {m.caption}
                </figcaption>
                <div className="overflow-x-auto rounded border border-border">
                  <table className="w-full min-w-[32rem] font-mono text-xs tabular-nums">
                    <thead className="bg-surface text-left text-muted">
                      <tr>
                        {t.header.map((h, i) => (
                          <th key={i} className="whitespace-nowrap px-3 py-2 font-normal">
                            {h}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {t.rows.map((r, i) => (
                        <tr key={i} className="border-t border-border">
                          {r.map((c, j) => (
                            <td key={j} className={numeric.test(c) && j > 0 ? "whitespace-nowrap px-3 py-1.5 text-right" : "px-3 py-1.5"}>
                              {c}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <pre className="overflow-x-auto rounded border border-border bg-surface px-3 py-2 font-mono text-[11px] text-muted">
                  {m.command}
                </pre>
              </figure>
            );
          })}
        </Section>

        <Section id="tested" title="how it is tested">
          <p className="max-w-2xl text-muted">{story.testing}</p>
        </Section>

        <Section id="code" title="the code">
          {excerpts.map((e) => (
            <figure key={`${e.file}-${e.startLine}`} className="space-y-2">
              <figcaption className="flex flex-wrap items-baseline justify-between gap-2 text-sm text-muted">
                <span>{e.caption}</span>
                <a href={e.url} className="link-underline font-mono text-xs text-text">
                  {e.file}:{e.startLine}–{e.endLine}
                </a>
              </figcaption>
              <div className="code-excerpt overflow-hidden rounded border border-border bg-surface" dangerouslySetInnerHTML={{ __html: e.html }} />
            </figure>
          ))}
          <div className="grid gap-6 font-mono text-xs sm:grid-cols-2">
            {[
              { title: "sources", items: story.sources },
              { title: "tests", items: story.tests },
            ].map((group) => (
              <div key={group.title}>
                <div className="text-muted">{group.title}</div>
                <ul className="mt-2 space-y-1">
                  {group.items.map((s) => (
                    <li key={s.path}>
                      <a href={`${repoUrl}/blob/main/${s.path}`} className="link-underline text-text">
                        {s.path}
                      </a>
                      <span className="text-muted"> · {s.label}</span>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </Section>

        <StoryNav slug={story.slug} />
      </article>
    </PageTransition>
  );
}
