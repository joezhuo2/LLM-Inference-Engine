import { notFound } from "next/navigation";
import { BookChapter } from "@/components/book/book-chapter";
import { kernelMetadata, ReadingView } from "@/components/kernels/reading-view";
import { kernelStories, kernelStory } from "@/content/kernels";

export const dynamicParams = false;

export function generateStaticParams() {
  return kernelStories.map((k) => ({ slug: k.slug }));
}

export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }) {
  return kernelMetadata((await params).slug);
}

export default async function KernelPage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const story = kernelStory(slug);
  if (!story) notFound();
  return story.book ? <BookChapter slug={slug} /> : <ReadingView slug={slug} />;
}
