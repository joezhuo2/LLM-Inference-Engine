import type { Metadata } from "next";
import Link from "next/link";
import { DeadEndList } from "@/components/kernels/dead-end-list";
import { StoryNav } from "@/components/kernels/story-nav";
import { PageTitle } from "@/components/page-title";
import { PageTransition } from "@/components/page-transition";
import { deadEnds } from "@/content/dead-ends";
import { ogImage } from "@/lib/og";

const description = "The changes that did not do what they were meant to: slower drafts, a broken bit, rules with a cliff and a profile read wrong, each with what was kept.";

export const metadata: Metadata = {
  title: "what didn't work",
  description,
  openGraph: { siteName: "llm-inference-engine", type: "article", title: "what didn't work · llm-inference-engine", description, images: [ogImage("dead-ends", "what didn't work: the dead ends of the kernels")] },
};

export default function DeadEnds() {
  return (
    <PageTransition>
      <article className="space-y-10">
        <header className="space-y-4">
          <div className="font-mono text-xs text-muted">
            <Link href="/kernels" transitionTypes={["nav-back"]} className="link-underline">
              kernels
            </Link>{" "}
            / dead ends
          </div>
          <PageTitle text="what didn't work" />
          <p className="max-w-2xl text-muted">
            {`${deadEnds.length} changes that did not do what they were meant to, from the profiling notes, the changelog and the PRs.`} Each one says
            what was tried, what the measurement or the test showed, and what was kept. Numbers are as measured at the time.
          </p>
        </header>
        <DeadEndList />
        <StoryNav slug="dead-ends" />
      </article>
    </PageTransition>
  );
}
