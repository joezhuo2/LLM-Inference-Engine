"use client";

import { Picker } from "../picker";
import { Inline, Stepper, Sticky } from "./variants";

export function StoryPrototype() {
  return (
    <Picker
      replay
      variants={[
        { name: "Sticky", render: () => <Sticky /> },
        { name: "Inline", render: () => <Inline /> },
        { name: "Stepper", render: () => <Stepper /> },
      ]}
    />
  );
}
