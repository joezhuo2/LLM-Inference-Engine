"use client";

import { useSyncExternalStore } from "react";

const query = "(prefers-reduced-motion: reduce)";
let list: MediaQueryList | null = null;
const media = () => (list ??= window.matchMedia(query));

function subscribe(onChange: () => void) {
  media().addEventListener("change", onChange);
  return () => media().removeEventListener("change", onChange);
}

export function usePrefersReducedMotion(): boolean {
  return useSyncExternalStore(
    subscribe,
    () => media().matches,
    () => false,
  );
}
