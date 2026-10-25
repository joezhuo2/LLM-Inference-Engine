"use client";

import { Picker } from "../picker";
import { useTrace } from "../trace";
import { Gauge, Readout, Spark } from "./variants";

export function TilesPrototype() {
  const trace = useTrace();
  if (!trace) return null;
  return (
    <Picker
      variants={[
        { name: "Readout", render: () => <Readout trace={trace} /> },
        { name: "Spark", render: () => <Spark trace={trace} /> },
        { name: "Gauge", render: () => <Gauge trace={trace} /> },
      ]}
    />
  );
}
