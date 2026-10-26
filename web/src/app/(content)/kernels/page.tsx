import type { Metadata } from "next";
import { KernelCards } from "@/components/kernels/kernel-cards";
import { LayerMap } from "@/components/kernels/layer-map";
import { PageTitle } from "@/components/page-title";

export const metadata: Metadata = { title: "kernels" };

export default function Kernels() {
  return (
    <div className="space-y-10">
      <header className="space-y-3">
        <PageTitle text="kernels" />
        <p className="max-w-2xl text-muted">
          How each kernel of the decode step works and how it got faster, version by version, with the measurements
          behind every change.
        </p>
      </header>
      <LayerMap />
      <KernelCards />
    </div>
  );
}
