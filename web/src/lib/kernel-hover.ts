"use client";

import { useSyncExternalStore } from "react";

let hovered: string | null = null;
const listeners = new Set<() => void>();

function subscribe(onChange: () => void) {
  listeners.add(onChange);
  return () => listeners.delete(onChange);
}

export function setHoveredKernel(slug: string | null) {
  if (slug === hovered) return;
  hovered = slug;
  listeners.forEach((l) => l());
}

export function useHoveredKernel(): string | null {
  return useSyncExternalStore(
    subscribe,
    () => hovered,
    () => null,
  );
}
