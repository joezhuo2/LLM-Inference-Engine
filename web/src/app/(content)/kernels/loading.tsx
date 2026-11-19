import { PageTransition } from "@/components/page-transition";
import { Skeleton } from "@/components/skeleton";

export default function Loading() {
  return (
    <PageTransition>
      <div role="status" aria-label="loading" className="min-h-dvh">
        <div className="fixed left-[clamp(22px,4.7vw,88px)] top-1/2 w-[min(510px,39vw)] -translate-y-1/2 space-y-4 max-[720px]:w-[min(76vw,400px)]">
          <Skeleton className="h-[clamp(68px,7vw,126px)] w-3/5" />
          <div className="space-y-2">
            <Skeleton className="h-5 w-full" />
            <Skeleton className="h-5 w-4/5" />
          </div>
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-10 w-full max-w-xl" />
        </div>
      </div>
    </PageTransition>
  );
}
