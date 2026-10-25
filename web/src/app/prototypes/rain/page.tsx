import type { Metadata } from "next";
import { RainPrototype } from "@/prototypes/rain/prototype";

export const metadata: Metadata = { title: "prototype: token rain", robots: { index: false } };

export default function RainPrototypePage() {
  return <RainPrototype />;
}
