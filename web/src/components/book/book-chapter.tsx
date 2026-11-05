"use client";

import { useEffect, useState } from "react";
import { BookView } from "./book-view";
import type { ChapterDef } from "./chapter";
import { bookChapters } from "./chapters";

export function BookChapter({ slug }: { slug: string }) {
  const [chapter, setChapter] = useState<ChapterDef | null>(null);
  useEffect(() => {
    let live = true;
    void bookChapters[slug]?.().then((c) => {
      if (live) setChapter(c);
    });
    return () => {
      live = false;
    };
  }, [slug]);
  if (!chapter) return <div aria-busy className="h-[calc(100dvh-3rem)] bg-[var(--book-desk)]" />;
  return <BookView key={slug} chapter={chapter} />;
}
