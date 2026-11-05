import type { Metadata } from "next";
import { FrontBook } from "@/components/book/front-book";

export const metadata: Metadata = { title: "contents · Every Byte" };

export default function Contents() {
  return <FrontBook />;
}
