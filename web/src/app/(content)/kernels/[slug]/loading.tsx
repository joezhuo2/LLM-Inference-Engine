import { PageTransition } from "@/components/page-transition";
import { Skeleton } from "@/components/skeleton";

export default function Loading() {
  return (
    <PageTransition>
      <div role="status" aria-label="loading" className="space-y-20">
        <div className="space-y-4">
          <Skeleton className="h-4 w-24" />
          <Skeleton className="h-9 w-72" />
          <Skeleton className="h-5 w-full max-w-2xl" />
          <div className="flex gap-10">
            <Skeleton className="h-12 w-48" />
            <Skeleton className="h-12 w-32" />
          </div>
        </div>
        <div className="space-y-6">
          <Skeleton className="h-6 w-40" />
          <Skeleton className="h-80 w-full" />
        </div>
      </div>
    </PageTransition>
  );
}
