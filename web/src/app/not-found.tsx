import type { Metadata } from "next";
import Link from "next/link";
import { PageTitle } from "@/components/page-title";

export const metadata: Metadata = { title: "not found", robots: { index: false } };

const pages = [
  { href: "/", label: "overview" },
  { href: "/results", label: "results" },
  { href: "/kernels", label: "kernels" },
];

export default function NotFound() {
  return (
    <div className="mx-auto w-full max-w-5xl space-y-6 px-4 py-20 sm:px-6">
      <div className="font-mono text-xs text-muted">error 404</div>
      <PageTitle text="page not found" />
      <p className="max-w-2xl text-muted">Nothing is served at this address. It may have moved when the site was reorganized.</p>
      <ul className="flex flex-wrap gap-x-6 gap-y-2 font-mono text-sm">
        {pages.map((p) => (
          <li key={p.href}>
            <Link href={p.href} className="link-underline text-accent">
              {p.label}
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
}
