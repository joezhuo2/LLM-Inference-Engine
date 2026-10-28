import { PageTitle } from "@/components/page-title";
import type { Metadata } from "next";
import { batches, environment, format, medianOf, systems } from "@/lib/data";

export const metadata: Metadata = { title: "results" };

export default function Results() {
  return (
    <div className="space-y-6">
      <PageTitle text="results" />
      <p className="max-w-2xl text-muted">
        Output tokens per second and median time per output token, median of the recorded runs. Workload:{" "}
        {String(environment.workload ?? "unknown")}.
      </p>
      {batches.length === 0 ? (
        <p className="rounded border border-border bg-surface p-4 font-mono text-sm text-muted">
          No benchmark runs in results/ yet: run scripts/reproduce.sh and rebuild.
        </p>
      ) : (
        <div className="overflow-x-auto rounded border border-border">
          <table className="w-full min-w-[36rem] font-mono text-sm tabular-nums">
            <thead className="bg-surface text-left text-muted">
              <tr>
                <th className="px-3 py-2 font-normal">system</th>
                {batches.map((b) => (
                  <th key={b} className="whitespace-nowrap px-3 py-2 text-right font-normal">
                    batch {b}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {systems.map((sys) => (
                <tr key={sys.id} className="border-t border-border">
                  <td className={sys.id.startsWith("engine") ? "px-3 py-2 text-accent" : "px-3 py-2"}>{sys.label}</td>
                  {batches.map((b) => {
                    const tps = medianOf(sys.id, b, "output_throughput");
                    const tpot = medianOf(sys.id, b, "tpot_s_p50");
                    return (
                      <td key={b} className="px-3 py-2 text-right">
                        {format(tps, 0)}
                        <span className="block whitespace-nowrap text-xs text-muted">{tpot === undefined ? "" : `${format(tpot * 1e3, 2)} ms`}</span>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
