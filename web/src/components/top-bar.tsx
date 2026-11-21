"use client";

import clsx from "clsx";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { type MouseEvent, type ReactNode, useState } from "react";
import { ThemeToggle } from "@/components/theme-toggle";

const bookCover = "/every-byte";
const sweepEase = "cubic-bezier(0.65, 0, 0.35, 1)";

function bookShown() {
  return window.location.pathname.startsWith(bookCover) && (document.querySelector<HTMLElement>("main canvas[role='img']")?.dataset.ready !== undefined || !!document.querySelector("main h1")?.textContent?.includes("WebGL"));
}

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
  const router = useRouter();
  const [opening, setOpening] = useState(false);
  const inBook = pathname.startsWith("/every-byte");
  const bookOpen = inBook || opening;
  const enterBook = (event: MouseEvent<HTMLAnchorElement>) => {
    if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    const host = document.querySelector("main canvas[role='img']")?.parentElement;
    if (inBook && host) {
      event.preventDefault();
      host.dispatchEvent(new CustomEvent("bookclose"));
      return;
    }
    if (opening) return event.preventDefault();
    if (inBook || matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    event.preventDefault();
    const icon = event.currentTarget.getBoundingClientRect();
    const top = event.currentTarget.closest("header")?.getBoundingClientRect().bottom ?? 0;
    const x = icon.left + icon.width / 2;
    const y = icon.top + icon.height / 2 - top;
    const radius = Math.hypot(Math.max(x, innerWidth - x), Math.max(Math.abs(y), innerHeight - top - y));
    const sheet = document.createElement("div");
    sheet.setAttribute("aria-hidden", "true");
    sheet.dataset.bookSheet = "";
    sheet.className = "fixed inset-x-0 bottom-0 z-40 flex items-center justify-center bg-[var(--book-desk)]";
    sheet.style.top = `${top}px`;
    const name = document.createElement("span");
    name.className = "font-serif text-3xl text-[var(--book-faint)] opacity-0";
    name.textContent = "Every Byte";
    sheet.append(name);
    document.body.append(sheet);
    setOpening(true);
    name.animate({ opacity: [0, 1] }, { duration: 700, delay: 500, easing: sweepEase, fill: "forwards" });
    void sheet.animate({ clipPath: [`circle(0px at ${x}px ${y}px)`, `circle(${radius}px at ${x}px ${y}px)`] }, { duration: 700, easing: sweepEase }).finished.then(() => {
      router.push(bookCover);
      const since = performance.now();
      const wait = () => {
        if (!bookShown() && performance.now() - since < 10000) return void requestAnimationFrame(wait);
        setOpening(false);
        void sheet.animate({ opacity: [1, 0] }, { duration: 700, easing: sweepEase, fill: "forwards" }).finished.then(() => sheet.remove());
      };
      requestAnimationFrame(wait);
    });
  };
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
          <Tip label="read Every Byte, the 3D book">
            <Link href={bookCover} onClick={enterBook} aria-label="Every Byte, the 3D book" aria-current={inBook ? "page" : undefined} className={clsx("pressable rounded p-1.5 hover:text-text", inBook ? "text-text" : "text-muted")}>
              <svg aria-hidden viewBox="0 0 16 16" className="size-4 fill-none stroke-current" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
                <path className={clsx("transition-opacity duration-300 ease-in-out motion-reduce:transition-none", bookOpen && "opacity-0")} d="M2.5 13V3.5A1.5 1.5 0 0 1 4 2h9.5v10H4a1.5 1.5 0 0 0 0 3h9.5" />
                <path className={clsx("transition-opacity duration-300 ease-in-out motion-reduce:transition-none", !bookOpen && "opacity-0")} d="M8 4.5C6.5 3.3 4.3 3 1.5 3.2v9.3c2.8-.2 5 .1 6.5 1.3 1.5-1.2 3.7-1.5 6.5-1.3V3.2C11.7 3 9.5 3.3 8 4.5zM8 4.5v9.3" />
              </svg>
            </Link>
          </Tip>
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
