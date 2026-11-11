"use client";

import { useState } from "react";
import { Disclosure } from "@/components/disclosure";
import { recallOf } from "@/content/reviews";
import type { ChapterReview } from "@/content/reviews/types";

export function ReviewPage({ number, review }: { number: number; review: ChapterReview }) {
  const [open, setOpen] = useState<number | null>(null);
  return (
    <div className="space-y-4 pb-2 pt-4">
      <div className="text-[10px] text-muted">{`chapter ${number} · review`}</div>
      <h2 className="font-serif text-[2.6rem] leading-none text-text">check yourself</h2>
      <p className="text-sm text-muted">Decide your answer, then open it.</p>
      <ol className="space-y-4">
        {review.questions.map((q, i) => {
          const recall = recallOf(q);
          return (
            <li key={q.question} className="grid grid-cols-[1.5rem_minmax(0,1fr)] gap-x-2">
              <span className="font-serif text-xl leading-6 text-accent">{i + 1}</span>
              <div className="space-y-1.5">
                <p className="text-text">{q.question}</p>
                {recall && <p className="text-xs text-muted">{recall.line}</p>}
                <Disclosure summary={open === i ? "hide the answer" : "show the answer"} open={open === i} onToggle={() => setOpen(open === i ? null : i)}>
                  <p className="font-serif text-lg leading-snug text-accent">{q.answer}</p>
                  <p className="mt-1 text-muted">{q.why}</p>
                </Disclosure>
              </div>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
