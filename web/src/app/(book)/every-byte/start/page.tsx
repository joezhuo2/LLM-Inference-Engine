import type { Metadata } from "next";
import { EveryByte } from "@/components/book/every-byte";

export const metadata: Metadata = { title: "start here · Every Byte", description: "Who Every Byte is for, what to know first, and the life of one token through the engine." };

export default function Start() {
  return <EveryByte />;
}
