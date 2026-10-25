"use client";

import { useEffect, useRef, useState } from "react";
import { usePrefersReducedMotion } from "@/lib/reduced-motion";

export type Palette = {
  bg: string;
  text: string;
  muted: string;
  accent: string;
  warn: string;
  border: string;
  mono: string;
};

export function readPalette(): Palette {
  const s = getComputedStyle(document.documentElement);
  const v = (name: string) => s.getPropertyValue(name).trim();
  return {
    bg: v("--bg"),
    text: v("--text"),
    muted: v("--text-muted"),
    accent: v("--accent"),
    warn: v("--warn"),
    border: v("--border"),
    mono: v("--font-jetbrains") || "monospace",
  };
}

export type Frame = { ctx: CanvasRenderingContext2D; width: number; height: number; elapsedS: number; palette: Palette };

export function useCanvasLoop(draw: (frame: Frame) => void, stillAtS: number) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const reduced = usePrefersReducedMotion();
  const [playRequested, setPlayRequested] = useState(false);
  const playing = !reduced || playRequested;
  const drawRef = useRef(draw);

  useEffect(() => {
    drawRef.current = draw;
  });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    let width = 0;
    let height = 0;
    let visible = true;
    let raf = 0;
    let last = performance.now();
    let elapsed = playing ? 0 : stillAtS;
    let palette = readPalette();

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      width = canvas.clientWidth;
      height = canvas.clientHeight;
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      palette = readPalette();
      if (!playing) drawRef.current({ ctx, width, height, elapsedS: elapsed, palette });
    };

    const tick = (now: number) => {
      raf = 0;
      if (!visible || document.hidden) return;
      elapsed += Math.min((now - last) / 1000, 0.1);
      last = now;
      drawRef.current({ ctx, width, height, elapsedS: elapsed, palette });
      raf = requestAnimationFrame(tick);
    };

    const start = () => {
      if (!playing || raf || !visible || document.hidden) return;
      last = performance.now();
      raf = requestAnimationFrame(tick);
    };

    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(canvas);
    const intersection = new IntersectionObserver(([entry]) => {
      visible = entry.isIntersecting;
      start();
    });
    intersection.observe(canvas);
    const themeObserver = new MutationObserver(() => {
      palette = readPalette();
      if (!playing) drawRef.current({ ctx, width, height, elapsedS: elapsed, palette });
    });
    themeObserver.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    const onVisibility = () => start();
    document.addEventListener("visibilitychange", onVisibility);
    resize();
    start();
    return () => {
      cancelAnimationFrame(raf);
      resizeObserver.disconnect();
      intersection.disconnect();
      themeObserver.disconnect();
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [playing, stillAtS]);

  return { canvasRef, playing, play: () => setPlayRequested(true) };
}
