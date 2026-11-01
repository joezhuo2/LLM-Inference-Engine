import { DashModel } from "@/lib/dashboard/model";
import type { TraceStep } from "@/lib/trace";

export const SLOTS = 8;

export type Colors = { text: string; muted: string; border: string; warn: string; bg: string; surface: string; baseline: string; accent: string; prefill: string; decode: string; mono: string; slots: string[] };

export function readColors(): Colors {
  const s = getComputedStyle(document.documentElement);
  const v = (n: string) => s.getPropertyValue(n).trim();
  return {
    text: v("--text"),
    muted: v("--text-muted"),
    border: v("--border"),
    warn: v("--warn"),
    bg: v("--bg"),
    surface: v("--surface"),
    baseline: v("--baseline"),
    accent: v("--accent"),
    prefill: v("--series-2"),
    decode: v("--series-1"),
    mono: v("--font-jetbrains") || "monospace",
    slots: Array.from({ length: SLOTS }, (_, i) => v(`--req-${i + 1}`)),
  };
}

export function slotColor(slot: number): string {
  return slot >= 0 ? `var(--req-${slot + 1})` : "var(--baseline)";
}

export class Feed {
  model: DashModel;
  pending: TraceStep | null = null;
  progress = 0;
  selected: number | null = null;
  mine: number | null = null;
  playing = false;
  live = false;
  version = 0;
  private listeners = new Set<() => void>();

  constructor(model: DashModel) {
    this.model = model;
  }

  subscribe(listener: () => void) {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  emit() {
    this.version++;
    this.listeners.forEach((l) => l());
  }

  reset(model: DashModel) {
    this.model = model;
    this.pending = null;
    this.progress = 0;
    this.emit();
  }

  select(id: number | null) {
    this.selected = id === this.selected ? null : id;
    this.emit();
  }

  focus(): number | null {
    return this.mine ?? this.selected;
  }
}

export function emptyModel(totalBlocks = 12915, blockSize = 16): DashModel {
  return new DashModel({ totalBlocks, blockSize, slots: SLOTS });
}
