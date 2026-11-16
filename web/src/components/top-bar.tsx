"use client";

import clsx from "clsx";
import Link from "next/link";
import { usePathname } from "next/navigation";
import type { ReactNode } from "react";
import { ThemeToggle } from "@/components/theme-toggle";

const links = [
  { href: "/", label: "overview" },
  { href: "/results", label: "results" },
  { href: "/kernels", label: "kernels" },
  { href: "/dashboard", label: "dashboard" },
];

function Tip({ label, children }: { label: ReactNode; children: ReactNode }) {
  return (
    <span className="group relative flex">
      {children}
      <span aria-hidden className="pointer-events-none absolute right-0 top-full z-50 mt-2 -translate-y-1 whitespace-nowrap rounded-md border border-border bg-surface/95 px-2.5 py-1.5 text-xs text-muted opacity-0 shadow-lg shadow-black/15 backdrop-blur-md transition-[opacity,translate] duration-300 ease-out group-hover:translate-y-0 group-hover:opacity-100 group-has-[:focus-visible]:translate-y-0 group-has-[:focus-visible]:opacity-100 motion-reduce:transition-none">
        {label}
      </span>
    </span>
  );
}

export function TopBar() {
  const pathname = usePathname();
  return (
    <header style={{ viewTransitionName: "site-header" }} className="sticky top-0 z-10 border-b border-border bg-bg/90 backdrop-blur">
      <nav className="mx-auto flex h-12 w-full max-w-5xl items-center gap-4 px-4 tabular-nums text-[13px] sm:gap-6 sm:text-sm sm:px-6">
        <Link href="/" className="text-accent">
          llmie
        </Link>
        <ul className="flex min-w-0 flex-1 gap-3 overflow-x-auto [scrollbar-width:none] sm:gap-4">
          {links.map(({ href, label }) => {
            const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
            return (
              <li key={href} className={clsx("whitespace-nowrap", href === "/" && "max-sm:hidden")}>
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
        <div className="flex items-center gap-1">
          <Tip label="view the code on GitHub">
            <a href="https://github.com/joezhuo2/LLM-Inference-Engine" target="_blank" rel="noopener noreferrer" aria-label="GitHub repository" className="pressable rounded p-1.5 text-muted hover:text-text">
              <svg aria-hidden viewBox="0 0 16 16" className="size-4 fill-current">
                <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z" />
              </svg>
            </a>
          </Tip>
          <Tip
            label={
              <>
                <span className="theme-icon-dark">switch to dark mode</span>
                <span className="theme-icon-light">switch to light mode</span>
              </>
            }
          >
            <ThemeToggle />
          </Tip>
        </div>
      </nav>
    </header>
  );
}
