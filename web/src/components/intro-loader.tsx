"use client";

import { useEffect, useRef, useState } from "react";
import { traces } from "@/lib/data";
import { finishIntro } from "@/lib/intro";
import { usePrefersReducedMotion } from "@/lib/reduced-motion";

const kvBlocks = Number((traces.find((t) => t.meta.prompts && t.meta.prompts !== "random") ?? traces[0])?.meta.kv_blocks);

const stages = [
  "mapping safetensors header",
  "uploading 2.20 GB of weights",
  Number.isFinite(kvBlocks) && kvBlocks > 0 ? `allocating ${kvBlocks.toLocaleString("en-US")} KV blocks` : "allocating the KV cache",
  "capturing CUDA graphs",
  "replaying a recorded run",
];

const loadMs = 2400;
const exitMs = 600;
const hexEveryMs = 60;

function randomHex() {
  let s = "0x";
  for (let i = 0; i < 16; i++) s += "0123456789abcdef"[Math.floor(Math.random() * 16)];
  return s;
}

export function IntroLoader() {
  const reduced = usePrefersReducedMotion();
  const [phase, setPhase] = useState<"loading" | "exiting" | "gone">("loading");
  const fill = useRef<HTMLDivElement>(null);
  const percent = useRef<HTMLSpanElement>(null);
  const hash = useRef<HTMLDivElement>(null);
  const stage = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    const root = document.documentElement;
    if (reduced) {
      finishIntro();
      return;
    }
    root.style.overflow = "hidden";
    const start = performance.now();
    let lastHex = 0;
    let raf = 0;
    let timer = 0;
    const tick = (now: number) => {
      const t = Math.min((now - start) / loadMs, 1);
      const p = 1 - Math.pow(1 - t, 3);
      if (fill.current) fill.current.style.transform = `scaleX(${p})`;
      if (percent.current) percent.current.textContent = `${String(Math.round(p * 100)).padStart(3, "0")}%`;
      if (stage.current) stage.current.textContent = stages[Math.min(stages.length - 1, Math.floor(p * stages.length))];
      if (hash.current && now - lastHex > hexEveryMs) {
        hash.current.textContent = randomHex();
        lastHex = now;
      }
      if (t < 1) {
        raf = requestAnimationFrame(tick);
        return;
      }
      root.style.overflow = "";
      finishIntro();
      setPhase("exiting");
      timer = window.setTimeout(() => setPhase("gone"), exitMs);
    };
    raf = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(raf);
      clearTimeout(timer);
      root.style.overflow = "";
    };
  }, [reduced]);

  if (reduced || phase === "gone") return null;
  return (
    <div
      aria-hidden="true"
      className="intro-loader fixed inset-0 z-50 flex items-center justify-center bg-bg px-4"
      style={{
        opacity: phase === "exiting" ? 0 : 1,
        transition: `opacity ${exitMs}ms var(--ease-out)`,
        pointerEvents: phase === "exiting" ? "none" : undefined,
      }}
    >
      <div className="w-full max-w-sm font-mono text-xs">
        <div className="text-muted uppercase tracking-wide">
          <span className="text-accent">&gt;</span> <span ref={stage}>{stages[0]}</span>
        </div>
        <div ref={hash} className="mt-4 text-sm text-text tabular-nums">
          0x0000000000000000
        </div>
        <div className="mt-4 h-px bg-border">
          <div ref={fill} className="h-full origin-left bg-accent" style={{ transform: "scaleX(0)" }} />
        </div>
        <div className="mt-3 text-right text-muted tabular-nums">
          <span ref={percent}>000%</span>
        </div>
      </div>
    </div>
  );
}
