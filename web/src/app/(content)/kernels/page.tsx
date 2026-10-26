import type { Metadata } from "next";

export const metadata: Metadata = { title: "kernels" };

const stories = [
  { title: "paged attention: v1 to v2 to split", text: "Scores in shared memory, then an online softmax, then one long context cut across thread blocks and merged." },
  { title: "CUDA Graphs", text: "Capturing the decode step once per batch size and split count to remove launch gaps." },
  { title: "continuous batching and the paged KV cache", text: "Admitting, preempting and recomputing requests over a block-allocated cache." },
  { title: "fused add + RMSNorm", text: "128-bit loads, values kept in registers and warp shuffles in one kernel." },
];

export default function Kernels() {
  return (
    <div className="space-y-6">
      <h1 className="font-mono text-2xl">kernels</h1>
      <ol className="space-y-3">
        {stories.map((s, i) => (
          <li key={s.title} className="rounded border border-border bg-surface p-4">
            <div className="font-mono text-sm">
              <span className="text-muted">{String(i + 1).padStart(2, "0")} </span>
              <span className="text-accent">{s.title}</span>
            </div>
            <p className="mt-2 text-sm text-muted">{s.text}</p>
          </li>
        ))}
      </ol>
    </div>
  );
}
