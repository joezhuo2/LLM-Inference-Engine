import type { Metadata } from "next";
import { Dashboard } from "@/components/dashboard/dashboard";
import { PageTitle } from "@/components/page-title";
import { preferredTrace } from "@/lib/data";

export const metadata: Metadata = { title: "dashboard" };

function recordingLabel(): string {
  const t = preferredTrace();
  if (!t) return "no recorded run in results/traces";
  const date = String(t.meta.date ?? "").slice(0, 10);
  const commit = t.commit ? `, commit ${t.commit}` : "";
  return `${date} on ${String(t.meta.gpu)}${commit}: ${t.requests} chat requests at ${String(t.meta.request_rate)} per second, up to ${String(t.meta.max_batch)} at once${t.meta.graphs ? ", CUDA Graphs" : ""}`;
}

export default function DashboardPage() {
  return (
    <div className="space-y-4">
      <header className="space-y-2">
        <PageTitle text="dashboard" />
        <p className="max-w-3xl text-muted">The engine serving requests one scheduler step at a time: who ran, which KV blocks they held, what they wrote.</p>
      </header>
      <Dashboard label={recordingLabel()} />
    </div>
  );
}
