import type { Metadata } from "next";
import { format, traces } from "@/lib/data";

export const metadata: Metadata = { title: "dashboard" };

export default function Dashboard() {
  return (
    <div className="space-y-6">
      <h1 className="font-mono text-2xl">dashboard</h1>
      <p className="max-w-2xl text-muted">Recorded serving runs available for replay.</p>
      <ul className="space-y-3 font-mono text-sm">
        {traces.map((t) => (
          <li key={t.file} className="rounded border border-border bg-surface p-4">
            <div className="text-accent">{t.file.replace("/traces/", "")}</div>
            <div className="mt-2 text-muted tabular-nums">
              {t.requests} requests · {t.steps} steps · {format(t.duration_s)} s · up to {t.max_batch_seen} sequences at once ·{" "}
              {String(t.meta.gpu ?? "")}
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}
