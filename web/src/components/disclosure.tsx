"use client";

import { AnimatePresence, motion } from "motion/react";
import { type ReactNode, useId, useState } from "react";
import { usePrefersReducedMotion } from "@/lib/reduced-motion";

const ease = [0.65, 0, 0.35, 1] as const;

export function Disclosure({ summary, children }: { summary: string; children: ReactNode }) {
  const [open, setOpen] = useState(false);
  const reduced = usePrefersReducedMotion();
  const id = useId();
  const duration = reduced ? 0 : 0.6;
  return (
    <div className="text-sm">
      <button type="button" aria-expanded={open} aria-controls={id} onClick={() => setOpen((o) => !o)} className="flex items-center gap-1.5 text-left text-muted transition-colors duration-200 hover:text-text">
        <motion.svg viewBox="0 0 10 10" className="size-2.5 shrink-0 fill-current" initial={false} animate={{ rotate: open ? 90 : 0 }} transition={{ duration: reduced ? 0 : 0.4, ease }} aria-hidden>
          <path d="M3 1.5 7.5 5 3 8.5z" />
        </motion.svg>
        {summary}
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            id={id}
            key="content"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1, transition: { height: { duration, ease }, opacity: { duration: duration * 0.8, ease, delay: duration * 0.2 } } }}
            exit={{ height: 0, opacity: 0, transition: { height: { duration, ease }, opacity: { duration: duration * 0.6, ease } } }}
            className="overflow-hidden"
          >
            <div className="pt-2">{children}</div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
