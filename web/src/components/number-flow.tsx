"use client";

import Flow, { type NumberFlowProps } from "@number-flow/react";
import { useGliding } from "@/lib/glide";

export default function NumberFlow(props: NumberFlowProps) {
  const gliding = useGliding();
  return <Flow {...props} animated={!gliding} />;
}
