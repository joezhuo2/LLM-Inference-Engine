"use client";

import { createContext, type ReactNode, useContext } from "react";

const InBook = createContext(false);

export function BookText({ children }: { children: ReactNode }) {
  return <InBook.Provider value>{children}</InBook.Provider>;
}

export function Author({ story, book }: { story: string; book: string }) {
  return useContext(InBook) ? book : story;
}
