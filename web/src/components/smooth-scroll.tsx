"use client";

import { useEffect } from "react";
import { smoothScroll } from "@/lib/smooth-scroll";

export function SmoothScroll() {
  useEffect(() => smoothScroll(), []);
  return null;
}
