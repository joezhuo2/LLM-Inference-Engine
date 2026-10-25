"use client";

import { Picker } from "../picker";
import { Bars, Lines, Speedup } from "./variants";

export function ChartsPrototype() {
  return (
    <Picker
      variants={[
        { name: "Lines", render: () => <Lines /> },
        { name: "Bars", render: () => <Bars /> },
        { name: "Speedup", render: () => <Speedup /> },
      ]}
    />
  );
}
