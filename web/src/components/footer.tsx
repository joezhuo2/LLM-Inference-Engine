import Link from "next/link";
import { shortPathName } from "@/content/short-path";
import { environment } from "@/lib/data";

export function Footer() {
  return (
    <footer className="mx-auto w-full max-w-5xl border-t border-border px-4 py-6 tabular-nums text-xs text-muted sm:px-6">
      measured on {String(environment.gpu ?? "unknown GPU")} · commit {String(environment.commit ?? "unknown")} ·{" "}
      {String(environment.date ?? "").slice(0, 10)}
      <span hidden>
        {" · "}
        <Link href="/every-byte" className="link-underline">
          read the kernels as Every Byte, a 3D book
        </Link>
        {" · "}
        <Link href="/kernels/short-path" className="link-underline">
          {`if you have an hour: the ${shortPathName}`}
        </Link>
      </span>
    </footer>
  );
}
