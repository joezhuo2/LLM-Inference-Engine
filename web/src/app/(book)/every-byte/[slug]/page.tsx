import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { EveryByte } from "@/components/book/every-byte";
import { kernelStories, kernelStory } from "@/content/kernels";

export const dynamicParams = false;

export function generateStaticParams() {
  return kernelStories.filter((k) => k.book).map((k) => ({ slug: k.slug }));
}

export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }): Promise<Metadata> {
  const story = kernelStory((await params).slug);
  return story ? { title: `${story.title} · Every Byte`, description: story.short } : {};
}

export default async function BookChapterPage({ params }: { params: Promise<{ slug: string }> }) {
  const story = kernelStory((await params).slug);
  if (!story?.book) notFound();
  return <EveryByte />;
}
