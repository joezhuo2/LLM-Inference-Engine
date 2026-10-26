"use client";

import type { MotionProps, MotionStyle } from "motion/react";
import { usePrefersReducedMotion } from "@/lib/reduced-motion";

type Props = MotionProps & { style?: MotionStyle };

const edge = 0.025;

function windowTimes(start: number, end: number) {
  const s = Math.max(0, Math.min(start, 0.96));
  const e = Math.max(s + edge, Math.min(end, 0.97));
  return [0, s, Math.min(s + edge, e), e, Math.min(e + edge, 1), 1];
}

export function useLoop(duration: number) {
  const reduced = usePrefersReducedMotion();
  const loop = { duration, repeat: Infinity, ease: "linear" as const };

  const on = (start: number, end = 0.94, still = true): Props =>
    reduced
      ? { initial: false, animate: { opacity: still ? 1 : 0 } }
      : { initial: { opacity: 0 }, animate: { opacity: [0, 0, 1, 1, 0, 0] }, transition: { ...loop, times: windowTimes(start, end) } };

  const grow = (start: number, end: number, axis: "x" | "y" = "x", hold = 0.94): Props => {
    const style: MotionStyle = axis === "x" ? { transformBox: "fill-box", originX: 0, originY: 0.5 } : { transformBox: "fill-box", originX: 0.5, originY: 1 };
    if (reduced) return { initial: false, style: { ...style, transform: "none" } };
    const zero = `scale${axis.toUpperCase()}(0)`;
    const one = `scale${axis.toUpperCase()}(1)`;
    const s = Math.max(0, Math.min(start, 0.9));
    const e = Math.max(s + edge, Math.min(end, hold));
    return {
      style,
      initial: { transform: zero },
      animate: { transform: [zero, zero, one, one, zero] },
      transition: { ...loop, times: [0, s, e, Math.max(e, hold), 1] },
    };
  };

  const pulse = (offset: number): Props =>
    reduced
      ? { initial: false, animate: { opacity: 1 } }
      : {
          initial: { opacity: 0.35 },
          animate: { opacity: [0.35, 1, 0.35] },
          transition: { duration: 1.4, repeat: Infinity, ease: "easeInOut", delay: offset },
        };

  const move = (start: number, end: number, dx: number, dy: number, still = true): Props => {
    if (reduced) return { initial: false, animate: { x: still ? dx : 0, y: still ? dy : 0, opacity: still ? 1 : 0 } };
    const t = windowTimes(start, end);
    return {
      initial: { x: 0, y: 0, opacity: 0 },
      animate: { x: [0, 0, 0, dx, dx, dx], y: [0, 0, 0, dy, dy, dy], opacity: [0, 0, 1, 1, 0, 0] },
      transition: { ...loop, times: t, ease: ["linear", "linear", "easeInOut", "linear", "linear"] },
    };
  };

  return { on, grow, pulse, move, reduced };
}
