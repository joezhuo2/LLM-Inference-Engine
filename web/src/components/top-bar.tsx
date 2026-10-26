"use client";

import clsx from "clsx";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { ThemeToggle } from "@/components/theme-toggle";

const links = [
  { href: "/", label: "overview" },
  { href: "/results", label: "results" },
  { href: "/kernels", label: "kernels" },
  { href: "/dashboard", label: "dashboard" },
];

export function TopBar() {
  const pathname = usePathname();
  return (
    <header className="sticky top-0 z-10 border-b border-border bg-bg/90 backdrop-blur">
      <nav className="mx-auto flex h-12 w-full max-w-5xl items-center gap-6 px-4 font-mono text-sm sm:px-6">
        <Link href="/" className="text-accent">
          llmie
        </Link>
        <ul className="flex min-w-0 flex-1 gap-4 overflow-x-auto">
          {links.map(({ href, label }) => {
            const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
            return (
              <li key={href}>
                <Link
                  href={href}
                  aria-current={active ? "page" : undefined}
                  className={clsx("link-hover", active ? "text-text" : "text-muted")}
                >
                  {label}
                </Link>
              </li>
            );
          })}
        </ul>
        <ThemeToggle />
      </nav>
    </header>
  );
}
