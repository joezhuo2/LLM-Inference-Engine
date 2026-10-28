import { type ReactNode, ViewTransition } from "react";

const direction = { "nav-forward": "nav-forward", "nav-back": "nav-back", default: "none" } as const;

export function PageTransition({ children }: { children: ReactNode }) {
  return (
    <ViewTransition enter={direction} exit={direction} default="none">
      {children}
    </ViewTransition>
  );
}
