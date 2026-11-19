import Link from "next/link";
import { glossary } from "@/lib/glossary";
import { type GlossarySpread, glossaryIntro, glossaryTitle } from "./front-matter";

export function GlossaryArticle({ spread }: { spread: GlossarySpread }) {
  return (
    <>
      {spread.left.kind === "title" && (
        <>
          <h1 className="font-serif text-2xl text-text">{glossaryTitle}</h1>
          <p className="mt-1">{glossaryIntro[0]}</p>
          <Link href="/kernels/glossary" className="link-underline mt-1 block text-text">
            the glossary page, with a filter →
          </Link>
        </>
      )}
      <dl className="mt-2 space-y-2">
        {[spread.left, spread.right].flatMap((p) => (p.kind === "terms" ? p.keys : [])).map((key) => (
          <div key={key}>
            <dt>
              <span className="font-serif text-lg text-text">{glossary[key].term}</span>
              {` · ${glossary[key].full}`}
            </dt>
            <dd>{glossary[key].why}</dd>
          </div>
        ))}
      </dl>
    </>
  );
}
