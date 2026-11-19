import type { Metadata } from "next";
import Link from "next/link";
import { GlossaryList, type GlossaryRow } from "@/components/kernels/glossary-list";
import { PageTitle } from "@/components/page-title";
import { PageTransition } from "@/components/page-transition";
import { kernelStories } from "@/content/kernels";
import searchIndex from "@/content/search-index.json";
import { findTerms, glossaryEntries } from "@/lib/glossary";
import { ogImage } from "@/lib/og";

const description = "Every term the chapters explain on hover, in one list: what it stands for, why it matters here, and the chapter that first uses it.";

export const metadata: Metadata = {
  title: "glossary",
  description,
  openGraph: { siteName: "llm-inference-engine", type: "article", title: "glossary · llm-inference-engine", description, images: [ogImage("glossary", "glossary: every term the kernels explain")] },
};

const index = searchIndex as Record<string, { title: string; entries: { id: string; title: string; text: string }[] }>;

function firstUses() {
  const first: Record<string, GlossaryRow["first"]> = {};
  for (const story of kernelStories)
    for (const entry of index[story.slug]?.entries ?? [])
      for (const m of findTerms(entry.text)) first[m.key] ??= { href: `/kernels/${story.slug}#${entry.id}`, number: story.index, title: story.title };
  return first;
}

export default function Glossary() {
  const first = firstUses();
  const rows: GlossaryRow[] = [...glossaryEntries].sort((a, b) => a.term.localeCompare(b.term, "en", { sensitivity: "base" })).map((e) => ({ ...e, first: first[e.key] }));
  return (
    <PageTransition>
      <article className="space-y-10">
        <header className="space-y-4">
          <div className="tabular-nums text-xs text-muted">
            <Link href="/kernels" transitionTypes={["nav-back"]} className="link-underline">
              kernels
            </Link>{" "}
            / glossary
          </div>
          <PageTitle text="glossary" />
          <p className="max-w-2xl text-muted">
            {`The ${rows.length} terms the chapters explain when you hover them, each with what it stands for and why it matters in this engine.`} The
            chapter after each one is the first in book order whose step text uses it, found in the built pages by web/scripts/search-index.mjs.
          </p>
        </header>
        <GlossaryList rows={rows} />
      </article>
    </PageTransition>
  );
}
