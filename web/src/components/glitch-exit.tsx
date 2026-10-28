"use client";

import { type ReactNode, useEffect, useRef } from "react";
import { readPalette } from "@/lib/canvas";
import { usePrefersReducedMotion } from "@/lib/reduced-motion";

const inMs = 450;
const outMs = 450;
const frameMs = 40;

function noise(n: number) {
  const x = Math.sin(n * 12.9898) * 43758.5453;
  return x - Math.floor(x);
}

export function GlitchExit({ to, children }: { to?: string; children: ReactNode }) {
  const layer = useRef<HTMLDivElement>(null);
  const overlay = useRef<HTMLCanvasElement>(null);
  const turb = useRef<SVGFETurbulenceElement>(null);
  const disp = useRef<SVGFEDisplacementMapElement>(null);
  const reduce = usePrefersReducedMotion();

  useEffect(() => {
    const hero = layer.current;
    const canvas = overlay.current;
    const ctx = canvas?.getContext("2d");
    if (!hero || !canvas || !ctx) return;
    let lastY = window.scrollY;
    let lockY = 0;
    let start = -1;
    let jumped = false;
    let raf = 0;
    let palette = readPalette();

    const tearHero = (e: number, seed: number) => {
      if (reduce || e === 0) {
        hero.style.filter = hero.style.transform = "";
        return;
      }
      const r = noise(seed + 1);
      turb.current?.setAttribute("seed", String(seed));
      disp.current?.setAttribute("scale", String(e * 160 * (0.4 + r)));
      hero.style.filter = "url(#hero-glitch)";
      hero.style.transform = `translateX(${(r - 0.5) * 30 * e}px)`;
    };

    const draw = (e: number, seed: number) => {
      const w = window.innerWidth;
      const h = window.innerHeight;
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = palette.bg;
      if (reduce) {
        ctx.globalAlpha = e;
        ctx.fillRect(0, 0, w, h);
        ctx.globalAlpha = 1;
        return;
      }
      ctx.globalAlpha = Math.min(1, Math.max(0, (e - 0.55) / 0.35));
      ctx.fillRect(0, 0, w, h);
      const colors = ["#ffffff", palette.accent, palette.muted];
      const bars = Math.round(40 + e * 260);
      for (let i = 0; i < bars; i++) {
        const r = (k: number) => noise(seed * 131 + i * 7 + k);
        ctx.globalAlpha = 0.35 + r(5) * 0.65 * e;
        ctx.fillStyle = colors[Math.floor(r(6) * colors.length)];
        ctx.fillRect(r(3) * w, r(1) * h, 4 + r(4) ** 2 * 60, 2 + r(2) ** 3 * 10);
      }
      ctx.globalAlpha = 0.08 * e;
      ctx.fillStyle = palette.muted;
      for (let y = 0; y < h; y += 3) ctx.fillRect(0, y, w, 1);
      ctx.globalAlpha = 1;
    };

    const tick = (now: number) => {
      const ms = now - start;
      const seed = Math.floor(ms / frameMs);
      const e = ms < inMs ? ms / inMs : Math.max(0, 1 - (ms - inMs) / outMs);
      if (ms >= inMs && !jumped) {
        jumped = true;
        const target = to ? document.querySelector<HTMLElement>(to) : null;
        if (target) {
          const margin = parseFloat(getComputedStyle(target).scrollMarginTop) || 0;
          lockY = target.getBoundingClientRect().top + window.scrollY - margin;
          window.scrollTo({ top: lockY, behavior: "instant" });
        }
      }
      tearHero(jumped ? 0 : e, seed);
      draw(e, seed);
      if (ms < inMs + outMs) {
        raf = requestAnimationFrame(tick);
        return;
      }
      canvas.style.display = "none";
      start = -1;
      lastY = window.scrollY;
    };

    const play = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.round(window.innerWidth * dpr);
      canvas.height = Math.round(window.innerHeight * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      canvas.style.display = "block";
      palette = readPalette();
      lockY = window.scrollY;
      jumped = false;
      start = performance.now();
      raf = requestAnimationFrame(tick);
    };

    const onScroll = () => {
      const y = window.scrollY;
      if (start >= 0) {
        if (y !== lockY) window.scrollTo({ top: lockY, behavior: "instant" });
        return;
      }
      const rect = hero.getBoundingClientRect();
      const heroMid = rect.top + y + rect.height / 2;
      if (y > lastY && lastY < heroMid && y - lastY < window.innerHeight) play();
      lastY = y;
    };

    window.addEventListener("scroll", onScroll, { passive: true });
    return () => {
      window.removeEventListener("scroll", onScroll);
      cancelAnimationFrame(raf);
      hero.style.filter = hero.style.transform = "";
      canvas.style.display = "none";
    };
  }, [reduce, to]);

  return (
    <>
      <svg aria-hidden="true" className="absolute h-0 w-0">
        <filter id="hero-glitch" x="-5%" y="0%" width="110%" height="100%" colorInterpolationFilters="sRGB">
          <feTurbulence ref={turb} type="fractalNoise" baseFrequency="0.00001 0.03" numOctaves={1} seed={0} />
          <feComponentTransfer result="bands">
            <feFuncR type="discrete" tableValues="0.5 0.5 0 0.5 1 0.5 0.5 0.8 0.5 0.2 0.5" />
            <feFuncG type="table" tableValues="0.5 0.5" />
            <feFuncA type="table" tableValues="1 1" />
          </feComponentTransfer>
          <feDisplacementMap ref={disp} in="SourceGraphic" in2="bands" scale={0} xChannelSelector="R" yChannelSelector="G" />
        </filter>
      </svg>
      <div className="overflow-x-clip">
        <div ref={layer}>{children}</div>
      </div>
      <canvas ref={overlay} aria-hidden="true" className="pointer-events-none fixed inset-0 z-50 hidden h-full w-full" />
    </>
  );
}
