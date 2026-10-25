import type { Metadata } from "next";
import { ChartsPrototype } from "@/prototypes/charts/prototype";

export const metadata: Metadata = { title: "prototype: results charts", robots: { index: false } };

export default function ChartsPrototypePage() {
  return <ChartsPrototype />;
}
