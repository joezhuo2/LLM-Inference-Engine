"use client";

import clsx from "clsx";
import { AnimatePresence, motion } from "motion/react";
import { useState } from "react";
import { usePrefersReducedMotion } from "@/lib/reduced-motion";

export type Prediction = { question: string; choices: string[]; answer: string; why: string };

export function Predict({ question, choices, answer, why }: Prediction) {
  const [guess, setGuess] = useState<string | null>(null);
  const reduced = usePrefersReducedMotion();
  const right = guess === answer;
  return (
    <div className="space-y-2 rounded border border-border p-2.5 text-xs leading-relaxed">
      <div className="text-text">
        <span className="text-accent">predict first: </span>
        {question}
      </div>
      <div className="flex flex-wrap gap-1.5" role="group" aria-label="your prediction">
        {choices.map((c) => (
          <button key={c} type="button" aria-pressed={guess === c} onClick={() => setGuess(c)} className={clsx("pressable rounded border px-2 py-0.5", guess === c ? "border-accent text-accent" : "border-border text-text")}>
            {c}
          </button>
        ))}
      </div>
      <div role="status" aria-live="polite">
        <AnimatePresence initial={false}>
          {guess && (
            <motion.p key="why" initial={reduced ? false : { opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} transition={{ duration: reduced ? 0 : 0.5, ease: "easeInOut" }} className="overflow-hidden text-muted">
              <span className={right ? "text-accent" : "text-warn"}>{right ? "Right. " : `Not quite: ${answer}. `}</span>
              {why}
            </motion.p>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
