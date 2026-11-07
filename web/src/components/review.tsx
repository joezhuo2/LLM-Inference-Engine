import type { KernelStory } from "@/content/kernels/types";
import type { ChapterReview } from "@/content/reviews/types";
import { Disclosure } from "./disclosure";
import { gloss } from "./glossed";

export function ReviewView({ story, review }: { story: KernelStory; review: ChapterReview }) {
  return (
    <div className="grid gap-10 md:grid-cols-2">
      <div className="space-y-6">
        <div className="space-y-2 rounded border border-border p-4">
          <div className="text-xs text-muted">the number that matters</div>
          <div className="text-3xl tabular-nums text-accent">{story.headline.value}</div>
          <div className="text-sm text-text">{story.headline.label}</div>
          <p className="text-sm text-muted">{gloss(review.why)}</p>
        </div>
        <div className="space-y-3">
          <h3 className="font-serif text-2xl text-text">what to remember</h3>
          <ol className="space-y-3">
            {review.takeaways.map((t, i) => (
              <li key={i} className="grid grid-cols-[1.5rem_minmax(0,1fr)] gap-x-2 text-muted">
                <span className="tabular-nums text-accent">{i + 1}</span>
                <span>{gloss(t)}</span>
              </li>
            ))}
          </ol>
        </div>
      </div>
      <div className="space-y-3">
        <h3 className="font-serif text-2xl text-text">check yourself</h3>
        <p className="text-sm text-muted">Decide your answer before you open it.</p>
        <ol className="space-y-5">
          {review.questions.map((q, i) => (
            <li key={i} className="space-y-2 rounded border border-border p-4">
              <p className="text-text">
                <span className="mr-2 tabular-nums text-accent">{i + 1}</span>
                {gloss(q.question)}
              </p>
              <Disclosure summary="show the answer">
                <p className="text-text">{gloss(q.answer)}</p>
                <p className="mt-1 text-muted">{gloss(q.why)}</p>
              </Disclosure>
            </li>
          ))}
        </ol>
      </div>
    </div>
  );
}
