"use client";

import { animate, type AnimationPlaybackControls } from "motion";
import { useEffect, useRef, type ComponentProps } from "react";
import { usePrefersReducedMotion } from "@/lib/reduced-motion";

type Props = Omit<ComponentProps<"input">, "type" | "onChange"> & { onValue: (v: number) => void };

export function SmoothRange({ onValue, onPointerDown, onPointerUp, onPointerCancel, ...props }: Props) {
  const reduced = usePrefersReducedMotion();
  const down = useRef(0);
  const from = useRef(0);
  const target = useRef<number | null>(null);
  const tween = useRef<AnimationPlaybackControls | null>(null);
  const held = () => performance.now() - down.current > 250;

  useEffect(() => () => tween.current?.stop(), []);

  const glide = (el: HTMLInputElement, to: number) => {
    const min = Number(el.min || 0);
    const step = el.step === "any" ? 0 : Number(el.step || 1);
    const snap = (x: number) => (step ? min + Math.round((x - min) / step) * step : x);
    let last = from.current;
    tween.current = animate(from.current, to, {
      duration: 0.6,
      ease: [0.77, 0, 0.175, 1],
      onUpdate: (x) => {
        const v = snap(x);
        if (v === last) return;
        last = v;
        el.value = String(v);
        onValue(v);
      },
    });
  };

  return (
    <input
      {...props}
      type="range"
      onPointerDown={(e) => {
        tween.current?.stop();
        down.current = performance.now();
        from.current = Number(e.currentTarget.value);
        target.current = null;
        onPointerDown?.(e);
      }}
      onChange={(e) => {
        const v = Number(e.currentTarget.value);
        if (!down.current || held() || reduced) {
          target.current = null;
          return onValue(v);
        }
        target.current = v;
        e.currentTarget.value = String(from.current);
      }}
      onPointerUp={(e) => {
        const to = target.current;
        if (to !== null && to !== from.current) {
          if (held()) onValue(to);
          else glide(e.currentTarget, to);
        }
        down.current = 0;
        target.current = null;
        onPointerUp?.(e);
      }}
      onPointerCancel={(e) => {
        down.current = 0;
        target.current = null;
        onPointerCancel?.(e);
      }}
    />
  );
}
