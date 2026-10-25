import { environment } from "@/lib/data";

export function Footer() {
  return (
    <footer className="mx-auto w-full max-w-5xl border-t border-border px-4 py-6 font-mono text-xs text-muted sm:px-6">
      measured on {String(environment.gpu ?? "unknown GPU")} · commit {String(environment.commit ?? "unknown")} ·{" "}
      {String(environment.date ?? "").slice(0, 10)}
    </footer>
  );
}
