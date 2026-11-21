import Link from "next/link";
import { chapterNumber } from "@/content/book-order";
import { type EngineCell, type EngineKey, engineHref, engineKeys, engineRow, engineRows, engines, enginesChecked, enginesIntro, enginesTitle } from "@/content/engines-map";
import { kernelStory } from "@/content/kernels";

const title = (slug: string) => kernelStory(slug)?.title ?? slug;

function Cell({ k, cell }: { k: EngineKey; cell: EngineCell }) {
  if ("none" in cell) return <p className="text-muted">{cell.none}</p>;
  return (
    <div className="space-y-1">
      <a href={engineHref(k, cell)} target="_blank" rel="noreferrer" className="link-underline break-words font-mono text-[13px] text-accent">
        {cell.symbol}
      </a>
      <div className="break-all font-mono text-[11px] text-muted">{`${cell.path}:${cell.line}`}</div>
      <p className="text-muted">{cell.note}</p>
    </div>
  );
}

export function EnginesMap() {
  return (
    <section aria-labelledby="engines" className="space-y-4">
      <div>
        <div className="text-xs text-muted">what to open next</div>
        <h2 id="engines" className="mt-1 scroll-mt-20 font-serif text-3xl text-text">
          {enginesTitle}
        </h2>
      </div>
      {enginesIntro.map((p) => (
        <p key={p} className="max-w-2xl text-muted">
          {p}
        </p>
      ))}
      <table className="w-full border-collapse text-sm max-md:block">
        <caption className="pb-3 text-left text-xs text-muted max-md:block">{`Checked on ${enginesChecked} at ${engineKeys.map((k) => `${engines[k].name} ${engines[k].tag}`).join(" and ")}. Each link opens the file at that version, at the line named.`}</caption>
        <thead className="max-md:hidden">
          <tr className="text-left text-xs text-muted">
            <th className="w-[22%] pb-2 pr-4 font-normal">chapter</th>
            {engineKeys.map((k) => (
              <th key={k} className="w-[39%] pb-2 pr-4 font-normal">
                {engines[k].name}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="max-md:block max-md:space-y-4">
          {engineRows.map((r) => (
            <tr key={r.slug} className="border-t border-border align-top max-md:block max-md:rounded max-md:border max-md:p-3">
              <th scope="row" className="py-3 pr-4 text-left font-normal max-md:block max-md:pt-0">
                <Link href={`/kernels/${r.slug}`} className="link-underline font-serif text-base text-text">
                  <span className="tabular-nums text-muted">{chapterNumber(r.slug)}</span> {title(r.slug)}
                </Link>
                <div className="mt-1 text-xs text-muted">{r.idea}</div>
              </th>
              {engineKeys.map((k) => (
                <td key={k} className="py-3 pr-4 max-md:block max-md:border-t max-md:border-border max-md:pr-0">
                  <div className="mb-1 text-xs text-muted md:hidden">{engines[k].name}</div>
                  <Cell k={k} cell={r[k]} />
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

export function EnginesLine({ slug }: { slug: string }) {
  const row = engineRow(slug);
  if (!row) return null;
  return (
    <p className="max-w-2xl text-sm text-muted">
      {"in other engines: "}
      {engineKeys.map((k, i) => {
        const cell = row[k];
        return (
          <span key={k}>
            {i > 0 && ", "}
            {`${engines[k].name} `}
            {"none" in cell ? (
              <span>{cell.none}</span>
            ) : (
              <a href={engineHref(k, cell)} target="_blank" rel="noreferrer" className="link-underline font-mono text-[13px] text-accent">
                {cell.symbol}
              </a>
            )}
          </span>
        );
      })}
      {" · "}
      <Link href="/kernels/beyond#engines" className="link-underline text-text">
        the whole table
      </Link>
    </p>
  );
}

export function EnginesList() {
  return (
    <section className="space-y-2">
      <h1 className="font-serif text-2xl text-text">{enginesTitle}</h1>
      <p className="text-xs">{`${engineKeys.map((k) => `${engines[k].name} ${engines[k].tag}`).join(", ")}, checked ${enginesChecked}`}</p>
      <ol className="space-y-2">
        {engineRows.map((r) => (
          <li key={r.slug}>
            <span className="text-text">{`${chapterNumber(r.slug)} ${title(r.slug)}`}</span>
            {engineKeys.map((k) => {
              const cell = r[k];
              return (
                <span key={k} className="block text-xs">
                  {`${engines[k].name}: `}
                  {"none" in cell ? cell.none : (
                    <a href={engineHref(k, cell)} data-book-link={engineHref(k, cell)} target="_blank" rel="noreferrer" className="link-underline text-text">
                      {cell.symbol}
                    </a>
                  )}
                </span>
              );
            })}
          </li>
        ))}
      </ol>
    </section>
  );
}
