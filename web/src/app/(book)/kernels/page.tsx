import type { Metadata } from "next";
import { EveryByte } from "@/components/book/every-byte";

export const metadata: Metadata = { title: "Every Byte", description: "The kernels of an LLM inference engine as a 3D book, one chapter per kernel." };

export default function Kernels() {
  return <EveryByte />;
}
