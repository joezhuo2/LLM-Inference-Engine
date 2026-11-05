import type { Metadata } from "next";
import { EveryByte } from "@/components/book/every-byte";

export const metadata: Metadata = { title: "contents · Every Byte" };

export default function Contents() {
  return <EveryByte />;
}
