import type { Metadata } from "next";
import { Atlas } from "@/components/kernels/atlas";
import { PageTransition } from "@/components/page-transition";
import { atlasIntro } from "@/content/kernels-atlas";

export const metadata: Metadata = { title: "kernels", description: atlasIntro };

export default function Kernels() {
  return (
    <PageTransition>
      <Atlas />
    </PageTransition>
  );
}
