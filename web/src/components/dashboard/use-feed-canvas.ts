"use client";

import { useEffect, useRef } from "react";
import { type Colors, type Feed, readColors } from "./feed";

export type Paint = { ctx: CanvasRenderingContext2D; width: number; height: number; colors: Colors; dt: number };

export function useFeedCanvas(feed: Feed, draw: (paint: Paint) => boolean) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const drawRef = useRef(draw);
  const kickRef = useRef<() => void>(() => {});

  useEffect(() => {
    drawRef.current = draw;
    kickRef.current();
  });

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    let raf = 0;
    let last = 0;
    let width = 0;
    let height = 0;
    let colors = readColors();

    const frame = (now: number) => {
      raf = 0;
      if (!width || !height) return;
      const dt = last ? Math.min((now - last) / 1000, 0.1) : 0;
      last = now;
      ctx.clearRect(0, 0, width, height);
      if (drawRef.current({ ctx, width, height, colors, dt })) raf = requestAnimationFrame(frame);
      else last = 0;
    };
    const kick = () => {
      if (!raf) raf = requestAnimationFrame(frame);
    };
    kickRef.current = kick;

    const resize = new ResizeObserver(() => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      width = canvas.clientWidth;
      height = canvas.clientHeight;
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      kick();
    });
    resize.observe(canvas);
    const theme = new MutationObserver(() => {
      colors = readColors();
      kick();
    });
    theme.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    const unsubscribe = feed.subscribe(kick);
    return () => {
      cancelAnimationFrame(raf);
      resize.disconnect();
      theme.disconnect();
      unsubscribe();
      kickRef.current = () => {};
    };
  }, [feed]);

  return { canvasRef, kick: () => kickRef.current() };
}
