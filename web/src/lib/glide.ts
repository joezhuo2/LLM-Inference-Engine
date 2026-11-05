import { animate } from "motion";
import { useSyncExternalStore } from "react";

export const glideEase = [0.77, 0, 0.175, 1] as const;
export const glideSeconds = 0.6;

export type Tween = { stop: () => void };

let holds = 0;
const listeners = new Set<() => void>();

function subscribe(onChange: () => void) {
  listeners.add(onChange);
  return () => {
    listeners.delete(onChange);
  };
}

function notify() {
  document.documentElement.toggleAttribute("data-gliding", holds > 0);
  listeners.forEach((l) => l());
}

export function hold(): () => void {
  if (holds++ === 0) notify();
  let held = true;
  return () => {
    if (!held) return;
    held = false;
    if (--holds === 0) notify();
  };
}

export function useGliding(): boolean {
  return useSyncExternalStore(
    subscribe,
    () => holds > 0,
    () => false,
  );
}

export function glide(from: number, to: number, onUpdate: (v: number) => void, onComplete?: () => void): Tween {
  const release = hold();
  const run = animate(from, to, {
    duration: glideSeconds,
    ease: glideEase,
    onUpdate,
    onComplete: () => {
      onComplete?.();
      requestAnimationFrame(release);
    },
  });
  return {
    stop: () => {
      run.stop();
      release();
    },
  };
}
