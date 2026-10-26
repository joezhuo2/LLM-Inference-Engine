"use client";

import { useSyncExternalStore } from "react";

let done = false;
const listeners = new Set<() => void>();

export function finishIntro() {
  if (done) return;
  done = true;
  listeners.forEach((l) => l());
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function useIntroDone(): boolean {
  return useSyncExternalStore(
    subscribe,
    () => done,
    () => false,
  );
}
