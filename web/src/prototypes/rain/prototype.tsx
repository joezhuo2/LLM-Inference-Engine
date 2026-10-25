"use client";

import { Picker } from "../picker";
import { useTrace } from "../trace";
import { Cascade, Lanes, Pulse } from "./variants";

export function RainPrototype() {
  const trace = useTrace();
  if (!trace) return <div className="h-[72vh] min-h-[480px] rounded border border-border" />;
  return (
    <Picker
      replay
      variants={[
        { name: "Cascade", render: () => <Cascade trace={trace} /> },
        { name: "Lanes", render: () => <Lanes trace={trace} /> },
        { name: "Pulse", render: () => <Pulse trace={trace} /> },
      ]}
    />
  );
}
