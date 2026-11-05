"use client";

import Flow, { type NumberFlowProps } from "@number-flow/react";
import { useGliding } from "@/lib/glide";

const formatters = new Map<string, Intl.NumberFormat>();

function formatter(locales: NumberFlowProps["locales"], format: NumberFlowProps["format"]) {
  const key = JSON.stringify([locales, format]);
  let f = formatters.get(key);
  if (!f) {
    f = new Intl.NumberFormat(locales, format);
    formatters.set(key, f);
  }
  return f;
}

export default function NumberFlow(props: NumberFlowProps) {
  const gliding = useGliding();
  if (!gliding) return <Flow {...props} />;
  const { value, locales, format, prefix = "", suffix = "", className } = props;
  return <span className={className}>{`${prefix}${formatter(locales, format).format(value as number)}${suffix}`}</span>;
}
