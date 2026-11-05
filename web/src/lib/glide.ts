import { animate } from "motion";

export const glideEase = [0.77, 0, 0.175, 1] as const;
export const glideSeconds = 0.6;

export type Tween = { stop: () => void };

export function glide(from: number, to: number, onUpdate: (v: number) => void, onComplete?: () => void): Tween {
  return animate(from, to, { duration: glideSeconds, ease: glideEase, onUpdate, onComplete });
}
