import type { Metadata } from "next";
import { StoryPrototype } from "@/prototypes/story/prototype";

export const metadata: Metadata = { title: "prototype: kernel story", robots: { index: false } };

export default function StoryPrototypePage() {
  return <StoryPrototype />;
}
