"use client";

import clsx from "clsx";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { ThemeToggle } from "@/components/theme-toggle";

const links = [
  { href: "/", label: "overview" },
  { href: "/kernels", label: "kernels" },
];

export function TopBar() {
  const pathname = usePathname();
  return (
    <header style={{ viewTransitionName: "site-header" }} className="sticky top-0 z-10 border-b border-border bg-bg/90 backdrop-blur">
      <nav className="mx-auto flex h-12 w-full max-w-5xl items-center gap-4 px-4 font-mono text-[13px] sm:gap-6 sm:text-sm sm:px-6">
        <Link href="/" className="text-accent">
          llmie
        </Link>
        <ul className="flex min-w-0 flex-1 gap-3 overflow-x-auto sm:gap-4">
          {links.map(({ href, label }) => {
            const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
            return (
              <li key={href} className={clsx(href === "/" && "max-sm:hidden")}>
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
