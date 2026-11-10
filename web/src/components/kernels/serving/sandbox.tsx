"use client";

import { ServingReplay } from "./replay";
import { KvVisualizer } from "./visualizer";

export function ServingSandbox() {
  return (
    <div className="space-y-12">
      <div className="space-y-4">
        <h3 className="font-serif text-2xl text-text">allocate it yourself</h3>
        <KvVisualizer />
      </div>
      <div className="space-y-4">
        <h3 className="font-serif text-2xl text-text">replay a recorded run</h3>
        <ServingReplay />
      </div>
    </div>
  );
}
