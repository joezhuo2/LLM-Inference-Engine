import type { Metadata } from "next";
import Link from "next/link";
import { KernelCards } from "@/components/kernels/kernel-cards";
import { KernelSearch } from "@/components/kernels/kernel-search";
import { LayerMap } from "@/components/kernels/layer-map";
import { PageTitle } from "@/components/page-title";
import { PageTransition } from "@/components/page-transition";
import { shortPathName } from "@/content/short-path";

export const metadata: Metadata = { title: "kernels" };

export default function Kernels() {
  return (
    <PageTransition>
      <div className="space-y-10">
        <header className="space-y-3">
          <PageTitle text="kernels" />
          <p className="max-w-2xl text-muted">
            How each kernel of the decode step works and how it got faster, version by version, with the measurements
            behind every change.
          </p>
          <div className="flex flex-wrap gap-x-6 gap-y-2 text-sm">
            <Link href="/every-byte/start" className="link-underline text-accent">
              read them in Every Byte, the 3D book
            </Link>
            <Link href="/kernels/short-path" className="link-underline text-accent">
              {`if you have an hour: the ${shortPathName}`}
            </Link>
            <Link href="/kernels/glossary" transitionTypes={["nav-forward"]} className="link-underline text-accent">
              glossary: every term the chapters explain
            </Link>
          </div>
          <KernelSearch />
        </header>
        <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_18rem] lg:items-start">
          <LayerMap />
          <div className="lg:sticky lg:top-6">
            <KernelCards />
          </div>
        </div>
      </div>
    </PageTransition>
  );
}
