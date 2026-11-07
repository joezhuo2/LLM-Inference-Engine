import type { Metadata } from "next";
import Link from "next/link";
import { ExerciseCard } from "@/components/exercise-card";
import { PageTitle } from "@/components/page-title";
import { PageTransition } from "@/components/page-transition";
import { cloneCommand, exercises } from "@/content/exercises";
import { ogImage } from "@/lib/og";

const description = "Rebuild the engine piece by piece: check out the commit where each stub landed, make its tests pass, then compare with the code that shipped.";

export const metadata: Metadata = {
  title: "exercises",
  description,
  openGraph: { siteName: "llm-inference-engine", type: "article", title: "exercises · llm-inference-engine", description, images: [ogImage("exercises", "exercises: rebuild the engine piece by piece")] },
};

const groups = [
  { id: "cpu", title: "start here: no GPU needed", intro: "The block manager and the scheduler are plain C++ and their tests run on any machine with a C++20 compiler, CMake 3.28 and Ninja. They are the best place to start even if you have a GPU: the logic is the hard part, and a failure is easy to debug.", items: exercises.filter((e) => !e.gpu) },
  { id: "gpu", title: "on the GPU", intro: "These need an NVIDIA GPU and CUDA 12.8 or newer. The commands build for whatever GPU you have; the engine was written for an RTX 5060 Laptop GPU, so your times will differ, but the tests check correctness, not speed. Exercises marked as loading TinyLlama need the model downloaded as the README's Building section describes.", items: exercises.filter((e) => e.gpu) },
];

export default function Exercises() {
  return (
    <PageTransition>
      <article className="space-y-14">
        <header className="space-y-4">
          <PageTitle text="exercises" />
          <p className="max-w-2xl text-muted">
            Reading about a kernel is not the same as being able to write one. Most core pieces of this engine started as a stub: a function that throws, a spec of what it must do, and tests that fail until it does. Joe wrote nearly all of them against those tests; split attention was drafted with Claude at his request. The commits are still in the history, so you can start from exactly where he did.
          </p>
          <ol className="max-w-2xl list-decimal space-y-1 pl-5 text-muted">
            <li>
              Read the chapter, on its{" "}
              <Link href="/kernels" className="link-underline text-text">
                kernels page
              </Link>{" "}
              or in{" "}
              <Link href="/every-byte" className="link-underline text-text">
                Every Byte
              </Link>
              .
            </li>
            <li>Check out the stub&apos;s commit and run its tests. They fail.</li>
            <li>Write the function until they pass, using the spec and the tests as your only guide.</li>
            <li>Compare with the code that shipped, and read the chapter&apos;s timeline and what didn&apos;t work for the mistakes made on the way.</li>
          </ol>
          <p className="max-w-2xl text-muted">Clone the repo once:</p>
          <pre className="max-w-2xl overflow-x-auto rounded border border-border bg-surface px-3 py-2 font-mono text-[11px] text-text">{cloneCommand}</pre>
          <p className="max-w-2xl text-sm text-muted">
            GEMM calls cuBLAS and sampling was written without a stub, so those two chapters have no exercise. Each exercise switches to an old commit; <span className="font-mono text-[0.85em]">git switch main</span> brings you back.
          </p>
        </header>
        {groups.map((g) => (
          <section key={g.id} aria-labelledby={g.id} className="space-y-6">
            <div className="space-y-2">
              <h2 id={g.id} className="font-serif text-3xl text-text">
                {g.title}
              </h2>
              <p className="max-w-2xl text-muted">{g.intro}</p>
            </div>
            {g.items.map((e) => (
              <ExerciseCard key={e.id} exercise={e} chapterLink />
            ))}
          </section>
        ))}
      </article>
    </PageTransition>
  );
}
