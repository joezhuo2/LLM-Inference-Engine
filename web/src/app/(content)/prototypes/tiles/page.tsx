import type { Metadata } from "next";
import { TilesPrototype } from "@/prototypes/tiles/prototype";

export const metadata: Metadata = { title: "prototype: metric tiles", robots: { index: false } };

export default function TilesPrototypePage() {
  return <TilesPrototype />;
}
