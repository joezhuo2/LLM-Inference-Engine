import type { Metadata } from "next";
import { PageTitle } from "@/components/page-title";
import { PageTransition } from "@/components/page-transition";
import { ResultsStory } from "@/components/results/story";
import { repoUrl } from "@/content/build-log";
import { benchRun } from "@/content/results";
import { batches, environment, format, medianOf, systems } from "@/lib/data";
import { ogImage } from "@/lib/og";

const description = "Throughput, time per token, the decode step profile and the roofline of the engine against HuggingFace and llama.cpp, all measured on one laptop GPU.";

export const metadata: Metadata = {
  title: "results",
  description,
  openGraph: { siteName: "llm-inference-engine", type: "article", title: "results · llm-inference-engine", description, images: [ogImage("results", "results: the engine against HuggingFace and llama.cpp")] },
};

export default function Results() {
  return (
    <PageTransition>
      <div className="space-y-20">
        <header className="space-y-4">
          <PageTitle text="results" />
          <p className="max-w-2xl text-muted">
            What the engine measures on {benchRun.gpu}, against HuggingFace transformers and llama.cpp. Scroll through the
            steps; hover or focus a chart for its exact values. Every number is read from <code className="font-mono text-text">results/</code> and{" "}
            <a href={`${repoUrl}/blob/main/docs/results.md`} className="link-underline text-accent">
              docs/results.md
            </a>{" "}
            when the site is built.
          </p>
        </header>
        <ResultsStory />
        <section aria-labelledby="runs" className="space-y-4">
          <h2 id="runs" className="font-mono text-xl text-text">
            every run
          </h2>
          <p className="max-w-2xl text-muted">
            Output tokens per second and, below each, the median time per output token, median of the recorded runs.
            Workload: {String(environment.workload ?? "unknown")}.
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
                            {format(tps, 1)}
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
        </section>
        <section aria-labelledby="reproduce" className="space-y-4">
          <h2 id="reproduce" className="font-mono text-xl text-text">
            reproduce
          </h2>
          <p className="max-w-2xl text-muted">
            One script rebuilds the benchmark CSVs, the three decode step profiles, the roofline and the generated parts of
            docs/results.md, in about 10 minutes, plugged in and with no other GPU program running.
          </p>
          <pre className="overflow-x-auto rounded border border-border bg-surface p-4 font-mono text-sm text-text">scripts/reproduce.sh</pre>
        </section>
      </div>
    </PageTransition>
  );
}
