import { kernelMetadata, ReadingView } from "@/components/kernels/reading-view";
import { kernelStories } from "@/content/kernels";

export const dynamicParams = false;

export function generateStaticParams() {
  return kernelStories.map((k) => ({ slug: k.slug }));
}

export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }) {
  return kernelMetadata((await params).slug);
}

export default async function ReadPage({ params }: { params: Promise<{ slug: string }> }) {
  return <ReadingView slug={(await params).slug} />;
}
