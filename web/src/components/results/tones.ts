import type { Tone } from "@/content/results";

export const strokeOf: Record<Tone, string> = {
  accent: "stroke-accent",
  "series-2": "stroke-series-2",
  "series-3": "stroke-series-3",
  baseline: "stroke-baseline",
};

export const fillOf: Record<Tone, string> = {
  accent: "fill-accent",
  "series-2": "fill-series-2",
  "series-3": "fill-series-3",
  baseline: "fill-baseline",
};

export const easeInOut = [0.65, 0, 0.35, 1] as const;

export const charWidth = 6.6;
