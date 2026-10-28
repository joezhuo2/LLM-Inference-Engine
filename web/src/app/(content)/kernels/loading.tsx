import { PageTransition } from "@/components/page-transition";
import { Skeleton } from "@/components/skeleton";

export default function Loading() {
  return (
    <PageTransition>
      <div role="status" aria-label="loading" className="space-y-10">
        <div className="space-y-3">
          <Skeleton className="h-9 w-40" />
          <Skeleton className="h-5 w-full max-w-2xl" />
        </div>
        <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_18rem] lg:items-start">
          <Skeleton className="h-96 w-full" />
          <div className="space-y-2">
            {Array.from({ length: 6 }, (_, i) => (
              <Skeleton key={i} className="h-11 w-full" />
            ))}
          </div>
        </div>
      </div>
    </PageTransition>
  );
}
