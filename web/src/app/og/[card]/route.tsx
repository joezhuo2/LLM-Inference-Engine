import { deadEnds } from "@/content/dead-ends";
import { kernelStories, kernelStory } from "@/content/kernels";
import { headlineStats } from "@/lib/data";
import { ogCard } from "@/lib/og";

export const dynamic = "force-static";
export const dynamicParams = false;

export function generateStaticParams() {
  return ["overview", "dead-ends", ...kernelStories.map((k) => k.slug)].map((name) => ({ card: `${name}.png` }));
}

export async function GET(_request: Request, { params }: { params: Promise<{ card: string }> }) {
  const name = (await params).card.replace(/\.png$/, "");
  const story = kernelStory(name);
  if (name === "dead-ends") {
    return ogCard({
      eyebrow: "kernels / dead ends",
      title: "what didn't work",
      text: "The changes that did not do what they were meant to, with the measurement that showed it and what was kept.",
      stats: [{ label: "dead ends, each with its source", value: String(deadEnds.length) }],
    });
  }
  if (!story) {
    return ogCard({
      eyebrow: "overview",
      title: "llm-inference-engine",
      text: "A TinyLlama inference engine written from scratch in C++ and CUDA: paged KV cache, continuous batching, CUDA Graphs.",
      stats: headlineStats().slice(0, 3),
    });
  }
  return ogCard({
    eyebrow: `kernels / ${String(story.index).padStart(2, "0")}`,
    title: story.title,
    text: story.short,
    stats: [{ label: story.headline.label, value: story.headline.value }],
  });
}
