import Link from "next/link";
import type { ReactNode } from "react";
import { repoUrl } from "@/content/build-log";
import specs from "@/content/exercise-specs.json";
import { compareCommand, type Exercise, exerciseCommands, implLabel, prUrl } from "@/content/exercises";
import { gloss } from "./glossed";

function Commands({ lines }: { lines: string[] }) {
  return <pre className="overflow-x-auto rounded border border-border bg-surface px-3 py-2 font-mono text-[11px] leading-relaxed text-text">{lines.join("\n")}</pre>;
}

function Step({ n, title, children }: { n: number; title: string; children: ReactNode }) {
  return (
    <li className="grid grid-cols-[1.5rem_minmax(0,1fr)] gap-x-2">
      <span className="tabular-nums text-muted">{n}</span>
      <div className="space-y-2">
        <div className="text-text">{title}</div>
        {children}
      </div>
    </li>
  );
}

function Tag({ children }: { children: ReactNode }) {
  return <span className="rounded border border-border px-1.5 py-0.5 text-[11px] text-muted">{children}</span>;
}

type SpecParagraph = { label: string; text: string };

const exerciseSpec = (id: string): SpecParagraph[] => (specs as Record<string, SpecParagraph[]>)[id] ?? [];

const fileLink = (path: string, sha = "main") => `${repoUrl}/blob/${sha}/${path}`;

export function ExerciseCard({ exercise: e, chapterLink = false }: { exercise: Exercise; chapterLink?: boolean }) {
  return (
    <article id={e.id} aria-labelledby={`${e.id}-title`} className="scroll-mt-20 space-y-4 rounded border border-border p-4 sm:p-5">
      <header className="space-y-2">
        <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
          <h3 id={`${e.id}-title`} className="font-serif text-2xl text-text">
            {e.title}
          </h3>
          <div className="flex flex-wrap gap-1.5">
            <Tag>{e.gpu ? "needs an NVIDIA GPU" : "no GPU needed"}</Tag>
            {e.kind === "stub" && e.model && <Tag>some tests load TinyLlama</Tag>}
            {e.kind === "open" && <Tag>no tests, open ended</Tag>}
          </div>
        </div>
        <p className="max-w-2xl text-muted">{gloss(e.task)}</p>
        {chapterLink && (
          <Link href={`/kernels/${e.chapter}`} className="link-underline inline-block text-xs text-accent">
            read the chapter first
          </Link>
        )}
      </header>
      {e.kind === "stub" ? <StubSteps e={e} /> : <OpenSteps e={e} />}
      <details className="group text-sm">
        <summary className="cursor-pointer text-muted hover:text-text">a nudge, if you are stuck</summary>
        <p className="mt-2 max-w-2xl text-muted">{gloss(e.nudge)}</p>
      </details>
    </article>
  );
}

function StubSteps({ e }: { e: Extract<Exercise, { kind: "stub" }> }) {
  const [checkout, build, test] = exerciseCommands(e);
  const spec = exerciseSpec(e.id);
  return (
    <>
      <ol className="space-y-4 text-sm">
        <Step n={1} title="Go back to the day the stub landed">
          <p className="text-muted">
            Commit <span className="font-mono text-[0.85em] text-text">{e.start}</span> is the repo just before the implementation: the function throws <span className="font-mono text-[0.85em]">unimplemented</span>, the tests are written, and every piece the engine had by then is in place around it.
          </p>
          <Commands lines={[checkout, build]} />
        </Step>
        <Step n={2} title="Run the tests and watch them fail">
          <Commands lines={[test]} />
        </Step>
        <Step n={3} title="Make them pass">
          <p className="text-muted">
            Write it in{" "}
            <a href={fileLink(e.file, e.start)} className="link-underline font-mono text-[0.85em] text-text">
              {e.file}
            </a>
            , reading the spec below and the tests in{" "}
            <a href={fileLink(e.tests, e.start)} className="link-underline font-mono text-[0.85em] text-text">
              {e.tests}
            </a>
            . Rerun step 2 until nothing fails.
          </p>
        </Step>
        <Step n={4} title="Compare">
          <p className="text-muted">
            Then read {implLabel(e)} in{" "}
            {e.implPrs.map((n, i) => (
              <span key={n}>
                {i > 0 && ", "}
                <a href={prUrl(n)} className="link-underline text-text">
                  PR {n}
                </a>
              </span>
            ))}
            , or diff it against the stub:
          </p>
          <Commands lines={[compareCommand(e)]} />
        </Step>
      </ol>
      {spec.length > 0 && (
        <details className="group text-sm">
          <summary className="cursor-pointer text-muted hover:text-text">the spec, as Joe got it</summary>
          <div className="mt-3 max-w-2xl space-y-3 text-muted">
            {spec.map((p, i) => (
              <p key={i}>
                {p.label && <span className="text-text">{p.label}. </span>}
                {p.text}
              </p>
            ))}
            <p className="text-xs">
              Copied from the TODO(JOE) block of {e.file} at {e.start}, written in{" "}
              {e.stubPrs.map((n) => (
                <a key={n} href={prUrl(n)} className="link-underline text-text">
                  PR {n}
                </a>
              ))}
              .
            </p>
          </div>
        </details>
      )}
    </>
  );
}

function OpenSteps({ e }: { e: Extract<Exercise, { kind: "open" }> }) {
  return (
    <ol className="space-y-4 text-sm">
      <Step n={1} title="Write it from scratch">
        <p className="text-muted">Any file will do: a single .cu built with nvcc is enough. Nothing in the engine needs to change.</p>
        <Commands lines={["nvcc -O3 -arch=native copy.cu -o copy && ./copy"]} />
      </Step>
      <Step n={2} title="Check it">
        <p className="text-muted">{e.check}</p>
      </Step>
      <Step n={3} title="Compare">
        <ul className="space-y-1 text-muted">
          {e.compare.map((c) => (
            <li key={c.path}>
              <a href={fileLink(c.path)} className="link-underline font-mono text-[0.85em] text-text">
                {c.path}
              </a>{" "}
              · {c.label}
            </li>
          ))}
        </ul>
      </Step>
    </ol>
  );
}
