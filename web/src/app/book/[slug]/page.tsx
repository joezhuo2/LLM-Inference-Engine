import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { BookView } from "@/components/book/book-view";
import { kernelStory } from "@/content/kernels";

export const dynamicParams = false;

export function generateStaticParams() {
  return [{ slug: "rope" }];
}

export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }): Promise<Metadata> {
  const story = kernelStory((await params).slug);
  if (!story) return {};
  return { title: `${story.title}, the book`, description: story.short };
}

export default async function BookPage({ params }: { params: Promise<{ slug: string }> }) {
  const story = kernelStory((await params).slug);
  if (!story?.book) notFound();
  return <BookView />;
}
