"use client";

import { useEffect, useRef } from "react";
import { type Colors, type Feed, readColors } from "./feed";

export type Align = "left" | "center" | "right";

export type Label = (text: string, x: number, y: number, color: string, align?: Align) => void;

export type Paint = { ctx: CanvasRenderingContext2D; width: number; height: number; colors: Colors; dt: number; label: Label };

function labelCache(fontPx: number, mono: string, dpr: number) {
  const cache = new Map<string, HTMLCanvasElement>();
  const measure = document.createElement("canvas").getContext("2d")!;
  measure.font = `${fontPx}px ${mono}`;
  const h = Math.ceil(fontPx * 1.4);
  return (text: string, color: string) => {
    const key = `${color}|${text}`;
    let sprite = cache.get(key);
    if (!sprite) {
      if (cache.size > 2000) cache.clear();
      const w = Math.ceil(measure.measureText(text).width) + 2;
      sprite = document.createElement("canvas");
      sprite.width = Math.ceil(w * dpr);
      sprite.height = Math.ceil(h * dpr);
      const c = sprite.getContext("2d")!;
      c.scale(dpr, dpr);
      c.font = measure.font;
      c.textBaseline = "middle";
      c.fillStyle = color;
      c.fillText(text, 1, h / 2);
      cache.set(key, sprite);
    }
    return { sprite, w: sprite.width / dpr, h };
  };
}

export function useFeedCanvas(feed: Feed, draw: (paint: Paint) => boolean, fontPx: number) {
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
    let sprites = labelCache(fontPx, colors.mono, 1);

    const label: Label = (text, x, y, color, align = "left") => {
      const { sprite, w, h } = sprites(text, color);
      const left = align === "left" ? x - 1 : align === "right" ? x - w + 1 : x - w / 2;
      ctx.drawImage(sprite, Math.round(left), Math.round(y - h / 2), w, h);
    };

    const frame = (now: number) => {
      raf = 0;
      if (!width || !height) return;
      const dt = last ? Math.min((now - last) / 1000, 0.1) : 1 / 60;
      last = now;
      ctx.clearRect(0, 0, width, height);
      if (drawRef.current({ ctx, width, height, colors, dt, label })) raf = requestAnimationFrame(frame);
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
      sprites = labelCache(fontPx, colors.mono, dpr);
      kick();
    });
    resize.observe(canvas);
    const theme = new MutationObserver(() => {
      colors = readColors();
      sprites = labelCache(fontPx, colors.mono, Math.min(window.devicePixelRatio || 1, 2));
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
  }, [feed, fontPx]);

  return { canvasRef, kick: () => kickRef.current() };
}
